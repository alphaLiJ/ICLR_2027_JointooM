"""JointLoom CUDA inputs for the official variable-agent graph batches."""
import official_bootstrap as bootstrap
from collections import OrderedDict
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from data_store import arrays,ensure_occupancy_context
from native_extent_inputs import NativeExtentBatchBuilder
from mapf_cuda.simulation.grids import build_stateless_simulator


class CUDAGroup:
    def __init__(self, grid, agents, envs):
        self.grid=np.asarray(grid,dtype=np.uint8)
        self.agents,self.envs=agents,envs
        maps=np.repeat(self.grid[None],envs,axis=0)
        self.sim=build_stateless_simulator(maps,num_agents=agents,device='cuda:0')
        self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
        self.builder=NativeExtentBatchBuilder([self.grid.shape]*envs,agents,'cuda:0')
        self.host=torch.empty((envs*agents,8),dtype=torch.int16,pin_memory=True)
        self.raw=torch.empty_like(self.host,device='cuda')
        self.host_copy_done=torch.cuda.Event();self.uploaded=False
        self.free=torch.as_tensor(self.grid==0,device='cuda')
        self.offset=torch.arange(-5,6,device='cuda')

    def upload(self, records):
        if self.uploaded:self.host_copy_done.synchronize()
        rows=np.asarray(records,dtype=np.uint16).reshape(-1,8).copy()
        rows[:,0]=np.repeat(np.arange(self.envs,dtype=np.uint16),self.agents)
        np.copyto(self.host.numpy(),rows.view(np.int16))
        self.raw.copy_(self.host,non_blocking=True)
        self.host_copy_done.record(torch.cuda.current_stream());self.uploaded=True

    def unclipped_cost(self, data):
        # The published fine-tuning command leaves clamp_cost_to_go=None.
        # CUDA's byte distance cache is exact for the target map's finite
        # distances; recover the official unreachable/free-cell sentinel and
        # the +1 feature for obstacles/out-of-map positions before normalization.
        energy=self.sim.energy_maps.flatten(0,1)
        pos=self.raw[:,2:4].long()
        h,w=self.grid.shape
        rows=pos[:,0,None,None]+self.offset[None,:,None]
        cols=pos[:,1,None,None]+self.offset[None,None,:]
        rr=rows.clamp(0,h-1);cc=cols.clamp(0,w-1)
        valid=(rows>=0)&(rows<h)&(cols>=0)&(cols<w)&self.free[rr,cc]
        node=torch.arange(len(pos),device='cuda')
        centre_energy=energy[node,pos[:,0],pos[:,1]].to(torch.int32)
        centre=torch.where(centre_energy==0,h*w,255-centre_energy)
        local_energy=energy[node[:,None,None],rr,cc].to(torch.int32)
        local=torch.where(local_energy==0,h*w,255-local_energy)
        # Match NumPy's correctly rounded float32 division rather than the
        # CUDA scalar-division reciprocal shortcut at values such as 0.7.
        cost=((local-centre[:,None,None]).to(torch.float64)/10.).float()
        cost=torch.where(valid,cost,1.)
        data.x[:,3].zero_()
        data.x[:,3,1:12,1:12].copy_(cost)


class JointLoomInputs:
    def __init__(self):
        self.groups=OrderedDict();self.goal_work=OrderedDict();self.goals=OrderedDict()
        self.validated_maps=set();self.h2d_bytes=0;self.goal_rebuilds=0

    def validate_map(self,grid):
        key=grid.tobytes()
        if key in self.validated_maps:return
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        h,w=grid.shape
        cells=np.flatnonzero(grid.reshape(-1)==0)
        src=[];dst=[]
        for c in cells:
            x,y=divmod(int(c),w)
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                xx,yy=x+dx,y+dy
                if 0<=xx<h and 0<=yy<w and grid[xx,yy]==0:
                    src.append(c);dst.append(xx*w+yy)
        graph=csr_matrix((np.ones(len(src)),(src,dst)),shape=(h*w,h*w))
        distances=shortest_path(graph,directed=False,unweighted=True,indices=cells)
        max_distance=float(np.max(distances[np.isfinite(distances)]))
        if max_distance>=255:
            raise ValueError('Target map requires a wider goal-distance cache')
        self.validated_maps.add(key)

    def goal_energy(self,grid,rows):
        a=len(rows);key=(grid.tobytes(),rows[:,4:6].tobytes())
        if key in self.goals:
            value=self.goals.pop(key);self.goals[key]=value;return value
        work_key=(grid.tobytes(),a)
        if work_key not in self.goal_work:
            self.goal_work[work_key]=CUDAGroup(grid,a,1)
            if len(self.goal_work)>4:self.goal_work.popitem(last=False)
        work=self.goal_work[work_key]
        work.upload(rows[None])
        work.raw[:,7]=1
        work.sim.update_energy_maps(work.raw,a)
        value=work.sim.energy_maps[0].clone()
        self.goals[key]=value;self.goal_rebuilds+=1;self.h2d_bytes+=rows.nbytes
        if len(self.goals)>128:self.goals.popitem(last=False)
        return value

    def from_rows(self,grid,records,terminated=None):
        grid=np.asarray(grid,dtype=np.uint8)
        self.validate_map(grid)
        records=np.asarray(records,dtype=np.uint16).copy()
        occupancy=(records[:,:,7]&2)!=0
        records[:,:,7]&=1
        # Official Pogema1.3 MAPF observations express global coordinates in
        # the observation-padded grid. The native CUDA map has no such padding.
        records[:,:,2:6]-=5
        n,a,_=records.shape;key=(grid.tobytes(),a,n)
        if key not in self.groups:
            self.groups[key]=CUDAGroup(grid,a,n)
            if len(self.groups)>8:self.groups.popitem(last=False)
        work=self.groups.pop(key);self.groups[key]=work
        work.upload(records)
        energies=[self.goal_energy(grid,r) for r in records]
        torch.stack(energies,dim=0,out=work.sim.energy_maps)
        work.sim.refresh_compact_state_from_raw_batch(work.raw)
        # Retain the published Pogema observation occupancy exactly, including
        # its sequential bitmap-update behavior; graph positions remain full.
        invisible=torch.as_tensor(~occupancy.reshape(-1),device='cuda')
        xy=work.raw[:,2:4].long()
        flat=xy[:,0]*work.sim.energy_maps.shape[-1]+xy[:,1]
        words=work.raw[:,0].long()*work.sim.memory_tensors['grid_ocp'].shape[1]+flat//32
        mask=(torch.ones_like(flat)<<flat.remainder(32))*invisible
        removed=torch.zeros(work.sim.memory_tensors['grid_ocp'].numel(),dtype=torch.int64,device='cuda')
        removed.scatter_add_(0,words,mask)
        work.sim.memory_tensors['grid_ocp'].view(-1).bitwise_and_(~removed.to(torch.int32))
        data=work.builder.build(work.sim,work.raw,materialize_edges=True)
        work.unclipped_cost(data)
        self.h2d_bytes+=records.nbytes
        term=np.zeros((n,a),dtype=bool) if terminated is None else np.asarray(terminated,dtype=bool)
        return Batch(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr,
            edge_weight=torch.linalg.vector_norm(data.edge_attr[:,:2].double(),dim=-1),
            y=work.raw[:,6].long(),terminated=torch.as_tensor(term.reshape(-1),device='cuda'),
            batch=data.batch,ptr=data.ptr)

    def build(self,frames):
        for directory in {f.directory for f in frames}:
            ensure_occupancy_context(directory)
        counts=[arrays(f.directory)['compact'].shape[1] for f in frames]
        offsets=np.cumsum([0]+counts)
        grouped={}
        for i,(f,a) in enumerate(zip(frames,counts)):
            vals=arrays(f.directory);grid=np.asarray(vals['grid'])
            grouped.setdefault((a,grid.tobytes()),[]).append(i)
        pieces=[];edges=[];attrs=[];weights=[];indices=[]
        for (a,_),ids in grouped.items():
            vals=arrays(frames[ids[0]].directory);grid=np.asarray(vals['grid'])
            rows=np.stack([arrays(frames[i].directory)['compact'][frames[i].step] for i in ids])
            term=np.stack([arrays(frames[i].directory)['terminated'][frames[i].step] for i in ids])
            data=self.from_rows(grid,rows,term)
            # Own each group before its scratch storage can be used again.
            pieces.append(data.x.clone());attrs.append(data.edge_attr.clone());weights.append(data.edge_weight)
            original=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids])
            mapping=torch.as_tensor(original,device='cuda')
            edges.append(mapping[data.edge_index]);indices.extend(original.tolist())
        order=torch.as_tensor(np.argsort(indices),device='cuda')
        x=torch.cat(pieces).index_select(0,order)
        labels=np.concatenate([arrays(f.directory)['compact'][f.step,:,6] for f in frames]).astype(np.int64)
        terminated=np.concatenate([arrays(f.directory)['terminated'][f.step] for f in frames])
        result=Batch(x=x,edge_index=torch.cat(edges,dim=1),edge_attr=torch.cat(attrs),
            edge_weight=torch.cat(weights),y=torch.as_tensor(labels,device='cuda'),
            terminated=torch.as_tensor(terminated,device='cuda'),
            batch=torch.repeat_interleave(torch.arange(len(frames),device='cuda'),
                torch.as_tensor(counts,device='cuda')),
            ptr=torch.as_tensor(offsets,device='cuda'))
        result._num_graphs=len(frames)
        # Sort edges back to the published per-graph/source/target order.
        key=result.edge_index[0]*len(result.x)+result.edge_index[1]
        perm=key.argsort()
        result.edge_index=result.edge_index[:,perm].contiguous()
        result.edge_attr=result.edge_attr[perm]
        result.edge_weight=result.edge_weight[perm]
        return result


class RuntimeInputs:
    def __init__(self,grid_config,args,dataset_kwargs,builder):
        self.builder=builder

    def __call__(self,observations,env):
        a=len(observations);rows=np.zeros((1,a,8),dtype=np.uint16)
        rows[0,:,1]=np.arange(a)
        rows[0,:,2:4]=np.asarray([o['global_xy'] for o in observations])
        rows[0,:,4:6]=np.asarray([o['global_target_xy'] for o in observations])
        rows[0,:,7]=np.asarray([o['agents'][5,5]>0 for o in observations],dtype=np.uint16)*2
        grid=np.asarray(env.grid.get_obstacles(ignore_borders=True),dtype=np.uint8)
        return self.builder.from_rows(grid,rows)
