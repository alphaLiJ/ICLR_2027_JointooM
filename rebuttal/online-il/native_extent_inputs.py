"""Respect the native Pogema border when maps occupy a larger CUDA profile.

Pogema's one-cell artificial wall ends at the native corner. Profile-sized
virtual top/left walls otherwise incorrectly extend past that corner. This
small GPU adaptation is included in every new original-path comparison.
"""
import torch
from expert.fixed_magat_plus_runtime import PyGBatchBuilder


class NativeExtentBatchBuilder(PyGBatchBuilder):
    def __init__(self,extents,agents,device):
        super().__init__()
        extents=torch.as_tensor(extents,dtype=torch.int32,device=device).repeat_interleave(agents,dim=0)
        self.h=extents[:,0,None,None];self.w=extents[:,1,None,None]
        self.offset=torch.arange(-5,6,device=device)

    def adapt(self,data,x,y):
        rows=x.reshape(-1,1,1).to(torch.int32)+self.offset[None,:,None]
        cols=y.reshape(-1,1,1).to(torch.int32)+self.offset[None,None,:]
        beyond_wall=(rows < -1)|(rows>self.h)|(cols < -1)|(cols>self.w)
        outside_map=(rows<0)|(rows>=self.h)|(cols<0)|(cols>=self.w)
        data.x[:,0,1:12,1:12].masked_fill_(beyond_wall,0.)
        data.x[:,3,1:12,1:12].masked_fill_(outside_map,1.)
        return data

    def build_from_stateful(self,simulator,materialize_edges=True):
        data=super().build_from_stateful(simulator,materialize_edges)
        return self.adapt(data,simulator.cur_x,simulator.cur_y)

    def build(self,simulator,raw_batch,materialize_edges=True):
        data=super().build(simulator,raw_batch,materialize_edges)
        return self.adapt(data,raw_batch[:,2],raw_batch[:,3])
