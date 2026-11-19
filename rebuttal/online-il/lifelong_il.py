"""Common CE learner inputs: original expanded observations or compact restore."""
import numpy as np
import torch
from inputs import HostTransfer
from mapf_cuda.simulation.grids import build_stateless_simulator
from native_extent_inputs import NativeExtentBatchBuilder


class LifelongILConsumer:
    def __init__(self,path,tasks,device):
        self.path=path;self.device=device;self.n,self.a=tasks.positions.shape[:2]
        self.stream=torch.cuda.Stream(device=device)
        self.ready=[torch.cuda.Event() for _ in range(2)]
        self.finished=[torch.cuda.Event() for _ in range(2)];self.used=[False,False]
        self.bytes=0;self.refreshes=0
        if path=='pogema_original':
            self.transfers=[HostTransfer(self.n,self.a,device) for _ in range(2)]
            self.labels=[torch.empty(self.n*self.a,dtype=torch.long,pin_memory=True) for _ in range(2)]
        else:
            self.host=[torch.empty((self.n*self.a,8),dtype=torch.int16,pin_memory=True) for _ in range(2)]
            self.raw=[torch.empty_like(h,device=device) for h in self.host]
            self.sim=build_stateless_simulator([tasks.grid]*self.n,num_agents=self.a,device=device)
            self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
            self.builder=NativeExtentBatchBuilder([tasks.grid.shape]*self.n,self.a,device)
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def upload(self,packet,sequence):
        slot=sequence%2
        # Host staging may be overwritten only after its preceding H2D completed.
        if self.used[slot]:self.ready[slot].synchronize()
        with torch.cuda.stream(self.stream):
            if self.used[slot]:self.stream.wait_event(self.finished[slot])
            raw=packet['raw']
            self.refreshes+=int((raw[:,7]!=0).sum())
            if self.path=='pogema_original':
                data=self.transfers[slot].send(packet['arrays'])
                np.copyto(self.labels[slot].numpy(),raw[:,6],casting='unsafe')
                data.y.copy_(self.labels[slot],non_blocking=True)
                self.bytes+=sum(a.nbytes for a in packet['arrays'])+self.labels[slot].numel()*8
            else:
                np.copyto(self.host[slot].numpy(),raw)
                self.raw[slot].copy_(self.host[slot],non_blocking=True)
                self.bytes+=raw.nbytes
            self.ready[slot].record(self.stream)

    def get(self,sequence):
        slot=sequence%2;torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        if self.path=='pogema_original':return self.transfers[slot].data
        raw=self.raw[slot]
        changed=raw[raw[:,7]!=0]
        if len(changed):self.sim.update_energy_maps(changed,len(changed))
        self.sim.refresh_compact_state_from_raw_batch(raw)
        return self.builder.build(self.sim,raw,materialize_edges=True)

    def release(self,sequence):
        slot=sequence%2;self.finished[slot].record();self.used[slot]=True
