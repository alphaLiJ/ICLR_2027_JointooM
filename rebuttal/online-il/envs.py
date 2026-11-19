import bootstrap
import multiprocessing as mp
import os
import traceback
import time
import numpy as np
from inputs import CpuTopology,CpuMagatBuilder,combine_inputs
from tasks import assign_goals


class CpuLifelong:
    def __init__(self,tasks):
        from pogema.envs import PogemaCoopFinish
        from pogema.grid_config import GridConfig
        self.tasks=tasks;self.envs=[]
        self.topology=CpuTopology(tasks.grid)
        self.goals=tasks.goals.copy(); self.cursors=np.zeros(tasks.goals.shape[:2],dtype=np.int32)
        self.builders=[];self.steps=0;self.goal_bins=[];self.build_seconds=0.;self.step_seconds=0.
        for pos,goal in zip(tasks.positions,tasks.goals):
            cfg=GridConfig(map=tasks.grid.tolist(),agents_xy=pos.tolist(),targets_xy=goal.tolist(),
                num_agents=len(pos),obs_radius=5,on_target="nothing",collision_system="soft",
                observation_type="MAPF",max_episode_steps=4096,empty_outside=True)
            env=PogemaCoopFinish(cfg);env._initialize_grid();env.update_was_on_goal()
            self.envs.append(env);self.builders.append(CpuMagatBuilder(self.topology,goal))

    def positions(self):
        return np.asarray([e.grid.get_agents_xy(ignore_borders=True) for e in self.envs],dtype=np.int16)

    def build(self):
        started=time.perf_counter()
        pos=self.positions()
        data=combine_inputs([b.build(p,g) for b,p,g in zip(self.builders,pos,self.goals)],self.tasks.positions.shape[1])
        self.build_seconds+=time.perf_counter()-started
        return data

    def step(self,actions):
        started=time.perf_counter()
        for e,(env,action) in enumerate(zip(self.envs,actions)):
            env.move_agents(np.array(action,dtype=np.int64,copy=True))
            pos=np.asarray(env.grid.get_agents_xy(ignore_borders=True),dtype=np.int16)
            reached=np.all(pos==self.goals[e],axis=1)
            assign_goals(pos,self.goals[e],self.cursors[e],self.tasks.candidates[e],self.tasks.cells,reached)
            border=env.grid_config.obs_radius
            env.grid.finishes_xy=[tuple(map(int,g+border)) for g in self.goals[e]]
        self.steps+=1
        if self.steps%128==0:self.goal_bins.append(int(self.cursors.sum()))
        self.step_seconds+=time.perf_counter()-started

    def result(self):
        import resource
        return dict(positions=self.positions(),goals=self.goals,cursors=self.cursors,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            cached_goals=len(self.topology.cache),goal_bins=self.goal_bins,
            build_seconds=self.build_seconds,step_seconds=self.step_seconds)


def _cpu_worker(conn,tasks,affinity,backend,mode,prepare):
    try:
        os.sched_setaffinity(0,set(affinity))
        if backend=='original':
            from pogema_path import OriginalPogemaLifelong
            factory=lambda task:OriginalPogemaLifelong(task,mode=mode,prepare=prepare)
        else:factory=CpuLifelong
        env=factory(tasks);conn.send(True)
        while True:
            command,value=conn.recv()
            if command=="close":break
            if command=="reset":env=factory(tasks);conn.send(True)
            elif command=="build":conn.send(env.build())
            elif command=="step":
                actions,next_input=value;env.step(actions)
                conn.send(env.build() if next_input else True)
            elif command=="result":conn.send(env.result())
            elif command=="active":conn.send(int((~env.arrived).sum()) if mode=='standard_mapf' else int(env.goals.shape[0]*env.goals.shape[1]))
    except BaseException:
        conn.send({"error":traceback.format_exc()})
    finally:conn.close()


class CpuPool:
    def __init__(self,tasks,workers,affinity,backend='optimized',mode='lifelong',prepare=True):
        if len(tasks.positions)%workers:raise ValueError("envs must divide workers")
        self.conns=[];self.processes=[];self.workers=workers
        self.env_batch=len(tasks.positions)//workers
        ctx=mp.get_context("spawn")
        for ids in np.split(np.arange(len(tasks.positions)),workers):
            parent,child=ctx.Pipe()
            p=ctx.Process(target=_cpu_worker,args=(child,tasks.subset(ids),affinity,backend,mode,prepare))
            p.start();child.close();self.conns.append(parent);self.processes.append(p)
        for c in self.conns:self.receive(c)

    @staticmethod
    def receive(c):
        if not c.poll(120):raise TimeoutError("CPU lifelong worker timed out")
        value=c.recv()
        if isinstance(value,dict) and "error" in value:raise RuntimeError(value["error"])
        return value

    def reset(self):
        for c in self.conns:c.send(("reset",None))
        for c in self.conns:self.receive(c)

    def close(self):
        for c in self.conns:
            try:c.send(("close",None))
            except (BrokenPipeError,EOFError):pass
        for p in self.processes:
            p.join(5)
            if p.is_alive():p.terminate();p.join()


class ResidentLifelong:
    def __init__(self,tasks,device,horizon):
        import torch
        import grid_world_cpp as ext
        from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator
        self.torch=torch;self.tasks=tasks;self.device=device
        grids,extents=stack_grids_for_compiled_simulator([tasks.grid]*len(tasks.positions),device=device)
        self.sim=ext.GridWorldSimulator(grids,tasks.positions.shape[1],0,grids.numel(),20261108,"lifelong",horizon)
        self.sim.set_real_map_extents(extents)
        self.sim.pyg_builder_mode="local_gather";self.sim.pyg_local_gather_impl="auto"
        self.starts=torch.as_tensor(tasks.candidates,device=device)
        cells=np.broadcast_to(tasks.cells,(len(tasks.positions),*tasks.cells.shape)).copy()
        self.cells=torch.as_tensor(cells,device=device)
        self.counts=torch.full((len(tasks.positions),),len(tasks.cells),dtype=torch.int32,device=device)
        self.reset()

    def reset(self):
        t=self.torch;tasks=self.tasks
        self.sim.load_state(t.as_tensor(tasks.positions,device=self.device),t.as_tensor(tasks.goals,device=self.device),
            t.zeros(tasks.positions.shape[:2],dtype=t.uint8,device=self.device),
            t.zeros(len(tasks.positions),dtype=t.int32,device=self.device))
        self.sim.set_goal_candidates(self.starts,self.cells,self.counts)
        self.sim.update_derived_state()

    def result(self):
        return dict(positions=self.torch.stack((self.sim.cur_x,self.sim.cur_y),dim=-1).cpu().numpy(),
            goals=self.torch.stack((self.sim.goal_x,self.sim.goal_y),dim=-1).cpu().numpy(),
            cursors=self.sim.goal_cursors.cpu().numpy())
