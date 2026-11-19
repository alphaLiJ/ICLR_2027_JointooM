import bootstrap
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


class Topology:
    def __init__(self, grid):
        self.grid=grid;self.h,self.w=grid.shape;self.cache={}
        free=grid==0;edges=[]
        for dx,dy in ((1,0),(0,1)):
            x,y=np.nonzero(free[:self.h-dx,:self.w-dy]&free[dx:,dy:])
            edges.append((x*self.w+y,(x+dx)*self.w+y+dy))
        src=np.concatenate([x[0] for x in edges]);dst=np.concatenate([x[1] for x in edges])
        self.graph=coo_matrix((np.ones(2*len(src)),(np.r_[src,dst],np.r_[dst,src])),
                              shape=(grid.size,grid.size)).tocsr()

    def distances(self, goals):
        keys=goals[:,0].astype(np.int64)*self.w+goals[:,1]
        missing=sorted(set(keys)-self.cache.keys())
        if missing:
            ds=shortest_path(self.graph,directed=False,unweighted=True,indices=missing)
            for k,d in zip(missing,ds):
                out=np.full(d.shape,-1,dtype=np.int32)
                finite=np.isfinite(d);out[finite]=d[finite].astype(np.int32)
                self.cache[k]=out.reshape(self.h,self.w)
        return np.stack([self.cache[k] for k in keys])


def encode_numpy(positions, distances, best):
    """Vectorized implementation of released C++ generate_input."""
    n=len(positions);h,w=distances.shape[1:]
    rr=positions[:,0,None,None]+np.arange(-5,6)[None,:,None]
    cc=positions[:,1,None,None]+np.arange(-5,6)[None,None,:]
    valid=(rr>=0)&(rr<h)&(cc>=0)&(cc<w)
    raw=np.where(valid,distances[np.arange(n)[:,None,None],rr.clip(0,h-1),cc.clip(0,w-1)],-1)
    mx=np.maximum(0,raw.max((1,2)))
    transformed=np.where(raw<0,raw+5+mx[:,None,None],raw)
    transformed-=transformed.min((1,2))[:,None,None]
    denom=transformed.max((1,2))
    inv=denom[:,None,None]-transformed
    norm=inv/np.maximum(denom[:,None,None],1)
    current=distances[np.arange(n),positions[:,0],positions[:,1]]
    norm[:,5,5]=-np.minimum(1.,(current-best+1)/64.)
    occ=np.zeros((h,w),np.float32);occ[positions[:,0],positions[:,1]]=1
    local=np.where(valid,occ[rr.clip(0,h-1),cc.clip(0,w-1)],0)
    return np.stack((local,norm),axis=1).astype(np.float32)


class CpuEnvironment:
    """Pogema 1.2.2 collision implementation, replacing its observation with CostTracer."""
    def __init__(self,tasks):
        from pogema.envs import PogemaLifeLong
        from pogema.grid_config import GridConfig
        self.tasks=tasks;self.envs=[];self.topologies=[]
        self.episodes=np.zeros(tasks['envs'],np.int32)
        self.steps=np.zeros(tasks['envs'],np.int32)
        self.cursors=np.zeros((tasks['envs'],tasks['agents']),np.int32)
        self.goals=tasks['goals'][0].copy()
        self.distance=[];self.best=[]
        for e,grid in enumerate(tasks['grids']):
            config=GridConfig(size=max(grid.shape),map=grid.tolist(),agents_xy=tasks['positions'][0,e].tolist(),
                targets_xy=self.goals[e].tolist(),num_agents=tasks['agents'],obs_radius=5,
                on_target='restart',collision_system='soft',observation_type='POMAPF',
                max_episode_steps=tasks['horizon'],empty_outside=False,seed=tasks['seed']+e)
            env=PogemaLifeLong(config);env._initialize_grid();env.update_was_on_goal()
            self.envs.append(env);top=Topology(grid);self.topologies.append(top)
            d=top.distances(self.goals[e]);self.distance.append(d)
            pos=tasks['positions'][0,e]
            self.best.append(d[np.arange(tasks['agents']),pos[:,0],pos[:,1]].copy())

    def positions(self):
        return np.asarray([env.grid.get_agents_xy(ignore_borders=True) for env in self.envs],np.int16)

    def observe(self):
        return np.stack([encode_numpy(p,d,b) for p,d,b in zip(self.positions(),self.distance,self.best)])

    def step(self, actions):
        rewards=[];reached=[]
        for e,(env,act) in enumerate(zip(self.envs,actions)):
            # Original Pogema soft collision path; it mutates commands during resolution.
            # CostTracer replaces Pogema's unused POMAPF view and arrival reward.
            env.move_agents(act.astype(np.int64,copy=True))
            env.update_was_on_goal()
            pos=np.asarray(env.grid.get_agents_xy(ignore_borders=True),np.int16)
            dist=self.distance[e][np.arange(len(pos)),pos[:,0],pos[:,1]]
            rewards.append((dist<self.best[e]).astype(np.float32)*.1)
            self.best[e]=np.minimum(self.best[e],dist)
            hit=np.all(pos==self.goals[e],axis=1);reached.append(hit.astype(np.float32))
            ids=np.flatnonzero(hit)
            episode=self.episodes[e]%len(self.tasks['positions'])
            self.goals[e,ids]=self.tasks['streams'][episode,e,ids,self.cursors[e,ids]]
            self.cursors[e,ids]+=1
            if len(ids):
                self.distance[e][ids]=self.topologies[e].distances(self.goals[e,ids])
                self.best[e][ids]=self.distance[e][ids,pos[ids,0],pos[ids,1]]
            env.grid.finishes_xy=[tuple(map(int,g+5)) for g in self.goals[e]]
        self.steps+=1
        return self.observe(),np.stack(rewards),np.stack(reached),self.steps>=self.tasks['horizon']

    def reset_due(self):
        for e in np.flatnonzero(self.steps>=self.tasks['horizon']):
            self.episodes[e]+=1;episode=self.episodes[e]%len(self.tasks['positions'])
            self.steps[e]=0;self.cursors[e]=0
            pos=self.tasks['positions'][episode,e];self.goals[e]=self.tasks['goals'][episode,e]
            env=self.envs[e]
            env.grid.positions_xy=[tuple(map(int,p+5)) for p in pos]
            env.grid.finishes_xy=[tuple(map(int,g+5)) for g in self.goals[e]]
            env.grid.positions[:]=0
            for x,y in env.grid.positions_xy:env.grid.positions[x,y]=1
            env.was_on_goal=[False]*len(pos)
            self.distance[e]=self.topologies[e].distances(self.goals[e])
            self.best[e]=self.distance[e][np.arange(len(pos)),pos[:,0],pos[:,1]].copy()
        return self.observe()

    def state(self):
        return dict(positions=self.positions(),goals=self.goals.copy(),cursors=self.cursors.copy(),
                    best=np.stack(self.best),steps=self.steps.copy())


def cpu_worker(conn,tasks,affinity):
    import os,traceback
    try:
        os.sched_setaffinity(0,set(affinity))
        env=CpuEnvironment(tasks);conn.send(env.observe())
        while True:
            cmd,arg=conn.recv()
            if cmd=='close':break
            if cmd=='step':conn.send(env.step(arg))
            elif cmd=='reset':conn.send(env.reset_due())
            elif cmd=='state':conn.send(env.state())
    except BaseException:
        conn.send({'error':traceback.format_exc()})
    finally:conn.close()


class CpuPool:
    def __init__(self,tasks,workers=8):
        import multiprocessing as mp,os
        from tasks import subset
        self.tasks=tasks;self.conns=[];self.processes=[]
        self.groups=np.array_split(np.arange(tasks['envs']),min(workers,tasks['envs']))
        affinity=sorted(os.sched_getaffinity(0))[:16]
        ctx=mp.get_context('spawn')
        for ids in self.groups:
            parent,child=ctx.Pipe()
            p=ctx.Process(target=cpu_worker,args=(child,subset(tasks,ids),affinity));p.start();child.close()
            self.conns.append(parent);self.processes.append(p)
        self.obs=np.concatenate([self.receive(c) for c in self.conns])

    @staticmethod
    def receive(conn):
        if not conn.poll(120):raise TimeoutError('Pogema worker did not respond')
        result=conn.recv()
        if isinstance(result,dict) and 'error' in result:raise RuntimeError(result['error'])
        return result

    def step(self,actions):
        for conn,ids in zip(self.conns,self.groups):conn.send(('step',actions[ids]))
        results=[self.receive(c) for c in self.conns]
        return tuple(np.concatenate([r[k] for r in results]) for k in range(4))

    def reset_due(self):
        for c in self.conns:c.send(('reset',None))
        return np.concatenate([self.receive(c) for c in self.conns])

    def close(self):
        for c in self.conns:
            try:c.send(('close',None))
            except (BrokenPipeError,EOFError):pass
        for p in self.processes:
            p.join(5)
            if p.is_alive():p.terminate();p.join()


class CudaEnvironment:
    def __init__(self,tasks):
        import torch,grid_world_cpp as ext
        self.torch=torch;self.ext=ext;self.tasks=tasks
        self.n=tasks['envs'];self.a=tasks['agents'];self.horizon=tasks['horizon']
        padded=np.ones((self.n,128,128),np.int32)
        for e,g in enumerate(tasks['grids']):padded[e,:g.shape[0],:g.shape[1]]=g
        grid=torch.as_tensor(padded,device='cuda')
        self.sim=ext.GridWorldSimulator(grid,self.a,0,grid.numel(),tasks['seed'],'costtracer',self.horizon)
        self.builder=ext.MapfGPTObservationBuilder(grid,self.a)
        self.bank_pos=torch.as_tensor(tasks['positions'],device='cuda')
        self.bank_goal=torch.as_tensor(tasks['goals'],device='cuda')
        self.stream_bank=torch.as_tensor(tasks['streams'],device='cuda')
        self.episodes=torch.zeros(self.n,dtype=torch.int32,device='cuda')
        self.env_ids=torch.arange(self.n,device='cuda')
        self.stream=self.stream_bank[0].contiguous()
        self.cursor=torch.zeros((self.n,self.a),dtype=torch.int32,device='cuda')
        self.best=torch.full_like(self.cursor,-1)
        self.reward=torch.empty((self.n,self.a),device='cuda');self.reached=torch.empty_like(self.reward)
        self.obs=torch.empty((self.n,self.a,2,11,11),device='cuda')
        self.sim.load_state(self.bank_pos[0],self.bank_goal[0],
             torch.zeros((self.n,self.a),dtype=torch.uint8,device='cuda'),torch.zeros_like(self.episodes))
        self.observe()

    def observe(self):
        s=self.sim;b=self.builder
        b.refresh_distances(s.cur_x,s.cur_y,s.goal_x,s.goal_y)
        self.ext.costtracer_observe(s,b,self.best,self.obs)
        return self.obs

    def step(self,actions):
        self.sim.update_actions(actions.to(self.torch.uint8).reshape(self.n,self.a))
        self.sim.step_sim_only()
        self.ext.costtracer_reward_goals(self.sim,self.builder,self.best,self.stream,self.cursor,self.reward,self.reached)
        self.observe()
        return self.obs,self.reward,self.reached,self.sim.step_counts>=self.horizon

    def reset_due(self):
        self.ext.costtracer_reset(self.sim,self.bank_pos,self.bank_goal,self.episodes,self.cursor,
                                 self.best,self.episodes,self.horizon)
        self.stream=self.stream_bank[self.episodes.long()%len(self.bank_pos),self.env_ids].contiguous()
        return self.observe()

    def state(self):
        t=self.torch;s=self.sim
        return dict(positions=t.stack((s.cur_x,s.cur_y),-1).cpu().numpy(),
            goals=t.stack((s.goal_x,s.goal_y),-1).cpu().numpy(),cursors=self.cursor.cpu().numpy(),
            best=self.best.cpu().numpy(),steps=s.step_counts.cpu().numpy())
