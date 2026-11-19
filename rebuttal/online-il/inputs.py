"""Vectorized cached NumPy MAGAT inputs, including the published byte-energy convention."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


class CpuTopology:
    def __init__(self, grid):
        self.grid = grid
        self.h, self.w = grid.shape
        free = grid == 0
        edges = []
        for dx, dy in ((1, 0), (0, 1)):
            x, y = np.nonzero(free[:self.h-dx, :self.w-dy] & free[dx:, dy:])
            edges.append((x * self.w + y, (x+dx) * self.w + y+dy))
        src = np.concatenate([p[0] for p in edges]); dst = np.concatenate([p[1] for p in edges])
        self.graph = coo_matrix((np.ones(2*len(src)), (np.r_[src,dst], np.r_[dst,src])),
                                shape=(grid.size,grid.size)).tocsr()
        self.cache = {}
        embedded = np.zeros((128,128), dtype=np.float32)
        embedded[:self.h,:self.w] = grid
        if self.h < 128: embedded[self.h,:self.w+1] = 1
        if self.w < 128: embedded[:self.h+1,self.w] = 1
        self.obstacles = np.pad(embedded,6)
        self.obstacles[5,5:135] = 1; self.obstacles[134,5:135] = 1
        self.obstacles[5:135,5] = 1; self.obstacles[5:135,134] = 1

    def energies(self, goals):
        keys = goals[:,0].astype(np.int64)*self.w + goals[:,1]
        missing = sorted(set(map(int,keys)) - self.cache.keys())
        if missing:
            dist = shortest_path(self.graph, directed=False, unweighted=True, indices=missing)
            encoded = np.zeros(dist.shape, dtype=np.uint8)
            finite = dist <= 254
            encoded[finite] = (255-dist[finite]).astype(np.uint8)
            for k, flat in zip(missing,encoded):
                padded = np.zeros((128,128),dtype=np.uint8)
                padded[:self.h,:self.w] = flat.reshape(self.h,self.w)
                self.cache[k] = padded
        return [self.cache[int(k)] for k in keys]


class CpuMagatBuilder:
    def __init__(self, topology, goals):
        self.topology = topology
        self.goals = goals.copy()
        self.energy = np.stack(topology.energies(goals))
        self.offset = np.arange(-5,6)
        self.ids = np.arange(len(goals))[:,None,None]

    def build(self, positions, goals):
        changed = np.any(self.goals != goals,axis=1)
        if changed.any():
            self.energy[changed] = self.topology.energies(goals[changed])
            self.goals[:] = goals
        n = len(goals)
        x = np.zeros((n,4,13,13),dtype=np.float32)
        rr = positions[:,0,None,None]+self.offset[None,:,None]
        cc = positions[:,1,None,None]+self.offset[None,None,:]
        obs = self.topology.obstacles[rr+6,cc+6]
        x[:,0,1:12,1:12] = obs
        occupied = np.zeros((140,140),dtype=np.float32)
        occupied[positions[:,0]+6,positions[:,1]+6] = 1
        x[:,1,1:12,1:12] = occupied[rr+6,cc+6]
        base = self.energy[np.arange(n),positions[:,0],positions[:,1]].astype(np.float32)
        valid = (rr >= 0)&(rr < 128)&(cc >= 0)&(cc < 128)
        energy = self.energy[self.ids,np.clip(rr,0,127),np.clip(cc,0,127)]
        local = np.where(valid,energy,0).astype(np.float32)
        cost = np.clip((base[:,None,None]-local)/10.,-1,1)
        cost[(obs != 0)|(~valid)] = 1
        x[:,3,1:12,1:12] = cost
        delta = goals.astype(np.int64)-positions
        near = np.max(np.abs(delta),axis=1)<=5
        vertical = np.abs(delta[:,1])>=np.abs(delta[:,0])
        gx = np.where(vertical,6+np.rint(6*delta[:,0]/np.maximum(1,np.abs(delta[:,1]))),6*(np.sign(delta[:,0])+1)).astype(int)
        gy = np.where(vertical,6*(np.sign(delta[:,1])+1),6+np.rint(6*delta[:,1]/np.maximum(1,np.abs(delta[:,0])))).astype(int)
        gx[near]=6+delta[near,0]; gy[near]=6+delta[near,1]
        x[np.arange(n),2,gx,gy] = 1
        diff = positions[:,None,:].astype(np.int32)-positions[None,:,:]
        sq = np.sum(diff*diff,axis=-1)
        edges = np.array(np.nonzero((sq>0)&(sq<=49)),dtype=np.int64)
        rel = positions[edges[0]].astype(np.float32)-positions[edges[1]]
        attr = np.concatenate((rel,np.abs(rel).sum(axis=1,keepdims=True)),axis=1)
        return x,edges,attr


def combine_inputs(parts, agents):
    return (np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1]+i*agents for i,p in enumerate(parts)],axis=1),
            np.concatenate([p[2] for p in parts]))


class HostTransfer:
    def __init__(self, envs, agents, device):
        import torch
        from expert.fixed_magat_plus_runtime import RuntimePyGBatch
        self.torch = torch; self.device=device; self.n=envs*agents
        shapes = ((self.n,4,13,13),(2,envs*agents*agents),(envs*agents*agents,3))
        dtypes = (torch.float32,torch.int64,torch.float32)
        self.host = [torch.empty(s,dtype=d,pin_memory=True) for s,d in zip(shapes,dtypes)]
        self.gpu = [torch.empty(s,dtype=d,device=device) for s,d in zip(shapes,dtypes)]
        self.data = RuntimePyGBatch(x=self.gpu[0],batch=torch.arange(envs,device=device).repeat_interleave(agents),
            ptr=torch.arange(0,self.n+1,agents,device=device),y=torch.zeros(self.n,dtype=torch.long,device=device),
            arrived=torch.zeros(self.n,dtype=torch.bool,device=device),terminated=torch.zeros(self.n,dtype=torch.bool,device=device))
        self.bytes=0

    def send(self, arrays):
        x,edge,attr=arrays; e=edge.shape[1]
        views=[self.host[0],self.host[1][:,:e],self.host[2][:e]]
        dest=[self.gpu[0],self.gpu[1][:,:e],self.gpu[2][:e]]
        for h,g,a in zip(views,dest,arrays):
            np.copyto(h.numpy(),a);g.copy_(h,non_blocking=True)
        self.bytes+=sum(a.nbytes for a in arrays)
        self.data.edge_index=dest[1];self.data.edge_attr=dest[2]
        return self.data
