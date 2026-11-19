"""Matched parallel Pogema execution with shared-memory CPU or compact inputs."""
import runtime
from collections import OrderedDict
import ctypes
import multiprocessing as mp
import traceback
import numpy as np
import torch
from torch_geometric.data import Batch, Data


DTYPES = dict(rows=np.uint16, terminated=np.uint8, x=np.float32,
              edge_index=np.int64, edge_attr=np.float32, edge_weight=np.float64,
              edge_count=np.int32)


def views(shared):
    return {k: np.frombuffer(buf, dtype=DTYPES[k]).reshape(shape)
            for k, (buf, shape) in shared.items()}


def worker(conn, shared, ids, backend, kwargs, agents):
    try:
        from pogema import pogema_v0
        from magat_plus.runtime_data_generation import get_runtime_data_generator
        torch.set_num_threads(1)
        arrays = views(shared)
        args = runtime.configuration(agents)
        envs, observations, generators, streams = {}, {}, {}, {}

        def reset_one(e, seed):
            cfg = runtime.task_config(seed, agents)
            envs[e] = pogema_v0(cfg)
            observations[e], _ = envs[e].reset()
            if backend == 'cpu':
                generators[e] = get_runtime_data_generator(cfg, args, kwargs)
            return int(cfg.seed)

        def fill(e, full=True):
            arrays['rows'][e] = runtime.packed_state(observations[e])
            arrays['terminated'][e] = np.asarray(envs[e].was_on_goal, dtype=np.uint8)
            if backend == 'cpu' and full:
                data = generators[e](observations[e], envs[e])
                count = data.edge_index.shape[1]
                arrays['x'][e] = data.x.numpy()
                arrays['edge_count'][e] = count
                arrays['edge_index'][e, :, :count] = data.edge_index.numpy()
                arrays['edge_attr'][e, :count] = data.edge_attr.numpy()
                arrays['edge_weight'][e, :count] = data.edge_weight.numpy()

        conn.send(dict(ready=True))
        while True:
            request = conn.recv()
            if request['command'] == 'close':break
            resets, episodes = [], []
            if request['command'] == 'reset':
                for e in ids:
                    seed = int(request['seeds'][e])
                    streams[e] = np.random.default_rng(seed + 777_000_001)
                    resets.append((e, reset_one(e, seed)))
                    fill(e)
            elif request['command'] == 'step':
                for e, actions in request['actions']:
                    observations[e], _, term, trunc, _ = envs[e].step(actions)
                    if all(term) or all(trunc):
                        episodes.append(dict(env=e, success=bool(all(term)),
                            arrival=float(np.mean(envs[e].was_on_goal))))
                        resets.append((e, reset_one(e, int(streams[e].integers(10**10)))))
                    fill(e, full=request.get('full', True))
            conn.send(dict(resets=resets, episodes=episodes))
    except BaseException:
        conn.send(dict(error=traceback.format_exc()))
    finally:
        conn.close()


class EnvironmentPool:
    def __init__(self, backend, agents, num_envs, workers, kwargs):
        self.backend, self.agents, self.num_envs = backend, agents, num_envs
        self.context = mp.get_context('spawn')
        n, a = num_envs, agents
        shapes = dict(rows=(n,a,8), terminated=(n,a))
        if backend == 'cpu':
            shapes.update(x=(n,a,4,13,13), edge_index=(n,2,a*a),
                edge_attr=(n,a*a,3), edge_weight=(n,a*a), edge_count=(n,))
        self.shared = {}
        for key, shape in shapes.items():
            typ = np.ctypeslib.as_ctypes_type(DTYPES[key])
            self.shared[key] = (self.context.RawArray(typ, int(np.prod(shape))), shape)
        self.arrays = views(self.shared)
        self.children, self.connections, self.ids = [], [], []
        for rank in range(min(workers, n)):
            ids = list(range(rank,n,min(workers,n)))
            parent, child = self.context.Pipe()
            process = self.context.Process(target=worker,
                args=(child, self.shared, ids, backend, kwargs, agents))
            process.start();child.close()
            self.children.append(process);self.connections.append(parent);self.ids.append(ids)
        self._receive()
        self.rngs = {}
        self.episode_steps = np.zeros(n,dtype=np.int64)
        self.completed_episodes = []

    def _receive(self):
        results=[]
        for conn in self.connections:
            if not conn.poll(120):raise TimeoutError('environment worker did not respond')
            result=conn.recv()
            if 'error' in result:raise RuntimeError(result['error'])
            results.append(result)
        return results

    def _events(self, results):
        for result in results:
            for e, seed in result.get('resets',[]):
                self.rngs[e]=np.random.default_rng(seed)
                self.episode_steps[e]=0
            self.completed_episodes.extend(result.get('episodes',[]))

    def reset(self, seeds):
        self.completed_episodes=[]
        for conn in self.connections:conn.send(dict(command='reset',seeds=seeds))
        self._events(self._receive())

    def step(self, actions, full=True):
        for conn, ids in zip(self.connections,self.ids):
            conn.send(dict(command='step',actions=[(e,actions[e]) for e in ids],full=full))
        self.episode_steps += 1
        self._events(self._receive())

    def sample_actions(self, logits):
        probs=torch.softmax(logits,dim=-1).detach().cpu().numpy().reshape(self.num_envs,self.agents,5)
        result=np.empty((self.num_envs,self.agents),dtype=np.int64)
        for e in range(self.num_envs):
            # Same categorical draws as per-agent Generator.choice(size=1),
            # vectorized equally for both paths; one uniform draw per agent.
            p=probs[e]/np.sum(probs[e],axis=-1,keepdims=True)
            cdf=np.cumsum(p,axis=-1,dtype=np.float64)
            cdf/=cdf[:,-1:]
            result[e]=np.sum(self.rngs[e].random(self.agents)[:,None]>=cdf,axis=-1)
        return result

    def close(self):
        for conn, process in zip(self.connections,self.children):
            if process.is_alive():
                try:conn.send(dict(command='close'))
                except (BrokenPipeError,EOFError):pass
        for process in self.children:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join()
        for conn in self.connections:conn.close()


class Inputs:
    def __init__(self, backend, pool, grid):
        self.backend,self.pool,self.grid=backend,pool,np.asarray(grid,dtype=np.uint8)
        self.h2d_bytes=0
        n,a=pool.num_envs,pool.agents
        self.batch=torch.arange(n,device='cuda').repeat_interleave(a)
        self.ptr=torch.arange(n+1,device='cuda')*a
        if backend=='jointloom':
            import gpu_inputs
            if not getattr(gpu_inputs.CUDAGroup,'dagger_optimized',False):
                previous=gpu_inputs.CUDAGroup
                class OptimizedGroup(previous):
                    dagger_optimized=True
                    def __init__(self,*args,**kwargs):
                        super().__init__(*args,**kwargs)
                        self.sim.pyg_local_gather_impl='warp_cooperative'
                gpu_inputs.CUDAGroup=OptimizedGroup
            from gpu_inputs import JointLoomInputs
            self.builder=JointLoomInputs()
        else:
            self.pinned={
                'x':torch.empty((n*a,4,13,13),pin_memory=True),
                'edge_index':torch.empty((2,n*a*a),dtype=torch.long,pin_memory=True),
                'edge_attr':torch.empty((n*a*a,3),pin_memory=True),
                'edge_weight':torch.empty(n*a*a,dtype=torch.float64,pin_memory=True)}

    def build(self):
        arrays=self.pool.arrays;n,a=self.pool.num_envs,self.pool.agents
        if self.backend=='jointloom':
            data=self.builder.from_rows(self.grid,arrays['rows'],arrays['terminated'])
            self.h2d_bytes=self.builder.h2d_bytes
            return data
        host=self.pinned
        np.copyto(host['x'].numpy(),arrays['x'].reshape(n*a,4,13,13))
        offset=0
        for e,count in enumerate(arrays['edge_count']):
            count=int(count);end=offset+count
            np.add(arrays['edge_index'][e,:,:count],e*a,out=host['edge_index'].numpy()[:,offset:end])
            np.copyto(host['edge_attr'].numpy()[offset:end],arrays['edge_attr'][e,:count])
            np.copyto(host['edge_weight'].numpy()[offset:end],arrays['edge_weight'][e,:count])
            offset=end
        values=dict(x=host['x'].to('cuda',non_blocking=True),
            edge_index=host['edge_index'][:,:offset].to('cuda',non_blocking=True),
            edge_attr=host['edge_attr'][:offset].to('cuda',non_blocking=True),
            edge_weight=host['edge_weight'][:offset].to('cuda',non_blocking=True))
        self.h2d_bytes+=sum(v.numel()*v.element_size() for v in values.values())
        return Batch(**values,batch=self.batch,ptr=self.ptr,
            y=torch.zeros(n*a,dtype=torch.long,device='cuda'),
            terminated=torch.as_tensor(arrays['terminated'].reshape(-1).astype(bool),device='cuda'))


def own_graph(data, env, agents):
    lo,hi=env*agents,(env+1)*agents
    edges=(data.edge_index[0]>=lo)&(data.edge_index[0]<hi)
    return Data(x=data.x[lo:hi].clone(),edge_index=(data.edge_index[:,edges]-lo).contiguous(),
        edge_attr=data.edge_attr[edges].clone(),edge_weight=data.edge_weight[edges].clone(),
        y=torch.zeros(agents,dtype=torch.long,device='cuda'),
        terminated=data.terminated[lo:hi].clone())
