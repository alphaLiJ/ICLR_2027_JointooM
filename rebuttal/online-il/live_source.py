"""Original LaCAM environments feeding the original complete-stage ring."""
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
from tasks import assign_goals


FIELDS=('expert_calls','expert_seconds','environment_seconds','planning_goal_calls',
    'completed_goals','published_stages','generated_stages','cache_refresh_rows')


def _expert(directory,env_id,name,agents,seed,mode,window,limit,ring,ready,start,stop,errors,shared):
    try:
        from bootstrap import SOURCE, input_path
        from expert.expert_running import LacamExpertPolicy,configure_cpu_thread_limits,put_maps_into_registry
        from mapf_cuda.training.topology_async import _build_pogema_topology_map_env,_build_step_data
        from scipy.ndimage import label
        configure_cpu_thread_limits();put_maps_into_registry(str(SOURCE/'maps/maps.yaml'))
        env=_build_pogema_topology_map_env(num_agents=agents,map_name=name,seed=seed+env_id,
            max_episode_steps=256,collision_system='soft')
        inner=env.env.unwrapped
        policy=LacamExpertPolicy(lib_path=str(input_path('JOINTLOOM_LACAM_LIBRARY', SOURCE/'src/real_expert_alg/lacam/liblacam.so')),timeouts=[1.,5.,10.,30.])
        policy.reset_states(env);observations=inner._obs()
        grid=inner.grid.get_obstacles(ignore_borders=True)
        components,_=label(grid==0);free=np.argwhere(grid==0).astype(np.int16)
        if len(np.unique(components[grid==0]))!=1:raise ValueError('live original Maze source must be connected')
        rng=np.random.default_rng(np.random.SeedSequence([seed,env_id,20261108]))
        candidates=rng.integers(len(free),size=(agents,2048),dtype=np.int32)
        shuffle_rng=np.random.default_rng(np.random.SeedSequence([seed+10000,env_id]))
        cursors=np.zeros(agents,dtype=np.int32);arrived=np.zeros(agents,dtype=bool)
        previous_generated=None;previous_published=None;counts=np.zeros(8,dtype=np.float64)
        ready.set();start.wait()
        while counts[5]<limit and not stop.is_set():
            block=[]
            for _ in range(min(window,limit-int(counts[5]))):
                if stop.is_set():break
                goals=np.asarray(inner.grid.get_targets_xy(ignore_borders=True),dtype=np.int16)
                has_new=previous_generated is None or np.any(goals!=previous_generated)
                previous_generated=goals.copy()
                t=time.perf_counter();actions=policy.act(observations)
                counts[0]+=1;counts[1]+=time.perf_counter()-t;counts[3]+=int(has_new)
                raw=_build_step_data(observations,actions,env_id=env_id,reset_flag=0)
                block.append(raw);counts[6]+=1
                t=time.perf_counter()
                if mode=='lifelong':
                    observations,_,_,_,_=inner.step(actions.astype(np.int64,copy=False))
                    positions=np.asarray(inner.grid.get_agents_xy(ignore_borders=True),dtype=np.int16)
                    reached=np.all(positions==goals,axis=1)
                    assign_goals(positions,goals,cursors,candidates,free,reached)
                    counts[4]=int(cursors.sum())
                    border=inner.grid_config.obs_radius
                    inner.grid.finishes_xy=[tuple(map(int,g+border)) for g in goals]
                    if reached.any():
                        relative=inner.grid.get_targets_xy_relative()
                        for agent in np.flatnonzero(reached):
                            observations[agent]['global_target_xy']=inner.grid.finishes_xy[agent]
                            observations[agent]['target_xy']=relative[agent]
                else:
                    _,_,terminated,truncated,_=env.step(actions.astype(np.int64,copy=False))
                    positions=np.asarray(inner.grid.get_agents_xy(ignore_borders=True),dtype=np.int16)
                    reached=np.all(positions==goals,axis=1)
                    counts[4]+=int((reached&~arrived).sum());arrived|=reached
                    if all(terminated) or all(truncated):
                        env.reset();policy.reset_states(env);arrived.fill(False)
                    observations=inner._obs()
                counts[2]+=time.perf_counter()-t
                shared[env_id*8:(env_id+1)*8]=counts
            order=np.arange(len(block))
            if window>1:shuffle_rng.shuffle(order)
            for index in order:
                if stop.is_set():break
                raw=block[index]
                changed=np.ones(agents,dtype=bool) if previous_published is None else np.any(raw[:,4:6]!=previous_published,axis=1)
                raw[:,7]=changed;previous_published=raw[:,4:6].copy()
                ring.reserve_and_write(raw);counts[5]+=1;counts[7]+=int(changed.sum())
                shared[env_id*8:(env_id+1)*8]=counts
        stop.wait()
    except BaseException:
        errors.put(traceback.format_exc());ready.set()


class LiveRingSource:
    def __init__(self,ring,names,agents,seed,mode,window=1,limit=1000000000):
        self.ring=ring;self.sequence=0
        self.start_event=mp.Event();self.stop_event=mp.Event();self.errors=mp.Queue()
        self.ready=[mp.Event() for _ in names]
        self.shared=mp.Array('d',len(names)*8,lock=False)
        self.processes=[];directory=str(Path(__file__).resolve().parent)
        for e,(name,ready) in enumerate(zip(names,self.ready)):
            p=mp.Process(target=_expert,args=(directory,e,name,agents,seed,mode,window,limit,ring,
                ready,self.start_event,self.stop_event,self.errors,self.shared),daemon=True)
            p.start();self.processes.append(p)
        began=time.monotonic()
        while not all(r.is_set() for r in self.ready):
            self.check()
            if time.monotonic()-began>180:raise TimeoutError('live source initialization')
            time.sleep(.01)
        self.check()

    def check(self):
        if not self.errors.empty():raise RuntimeError(self.errors.get())
        if any(p.exitcode is not None for p in self.processes):raise RuntimeError('live expert exited')

    def start(self):self.start_event.set()

    def next_rows(self):
        began=time.monotonic()
        while not self.ring.is_stage_ready(self.sequence):
            self.check()
            if time.monotonic()-began>180:raise TimeoutError('live expert failed to publish')
            time.sleep(.0005)
        lo,hi=self.ring._stage_row_bounds(self.sequence)
        rows=self.ring.cpu_buffer[lo:hi].numpy().astype(np.uint16,copy=True)
        self.ring.release_stage(self.sequence);self.sequence+=1
        self.ring.shared_compute_ptr.value=self.sequence*self.ring.stage_rows
        return rows

    def stats(self):
        values=np.array(self.shared).reshape(-1,8)
        return dict(totals={key:float(value) for key,value in zip(FIELDS,values.sum(0))},
            workers=[dict(zip(FIELDS,map(float,row))) for row in values],
            published_per_env=list(map(int,self.ring.env_write_seq)),
            scope='completed source work; concurrent workers overlap; generated/published need not equal consumed')

    def try_next_rows(self):
        return self.next_rows() if self.ring.is_stage_ready(self.sequence) else None

    def close(self):
        self.stop_event.set();self.start_event.set()
        for p in self.processes:
            p.join(1)
            if p.is_alive():p.terminate();p.join()
        self.errors.cancel_join_thread();self.errors.close()
