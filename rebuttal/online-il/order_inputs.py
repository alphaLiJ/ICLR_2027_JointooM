"""Full-configuration replay, original CPU prefetch and dependency-correct CUDA."""
from pathlib import Path
import numpy as np
import torch
from mapf_cuda.simulation.grids import build_stateless_simulator
from native_extent_inputs import NativeExtentBatchBuilder
from references.host_inputs_original import CachedHostBuilder, HostPrefetch


class ConfigurationCorpus:
    def __init__(self, directory, stages=None):
        self.directory = Path(directory)
        self.grids = np.load(self.directory/'grids.npy')
        self.envs = len(self.grids)
        if (self.directory/'stages.npy').exists():
            raw = np.load(self.directory/'stages.npy', mmap_mode='r')
            raw = np.asarray(raw[:stages]).copy()
            self.agents = raw.shape[1]//self.envs
            raw = raw.reshape(-1,self.envs,self.agents,8)
            self.arrays = [raw[:,e] for e in range(self.envs)]
        else:
            self.arrays = [np.load(self.directory/f'env-{e}.npy',mmap_mode='r')[:stages].copy()
                for e in range(self.envs)]
            self.agents = self.arrays[0].shape[1]
        self.stages = len(self.arrays[0])
        if any(len(a)!=self.stages for a in self.arrays):
            raise ValueError('different source lengths')

    def stage(self, indices):
        rows = np.concatenate([a[int(i)] for a,i in zip(self.arrays,indices)]).astype(np.uint16,copy=False)
        rows[:,0] = np.repeat(np.arange(self.envs,dtype=np.uint16),self.agents)
        # Published refresh flags describe the original predecessor, not this order.
        rows[:,7] = 0
        return rows


class CompactReplay:
    def __init__(self, corpus, device='cuda:0'):
        self.device=device;self.corpus=corpus
        self.sim=build_stateless_simulator(corpus.grids,num_agents=corpus.agents,device=device)
        self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
        self.builder=NativeExtentBatchBuilder([g.shape for g in corpus.grids],corpus.agents,device)
        shape=(corpus.envs*corpus.agents,8)
        self.host=[torch.empty(shape,dtype=torch.int16,pin_memory=True) for _ in range(2)]
        self.raw=[torch.empty(shape,dtype=torch.int16,device=device) for _ in range(2)]
        self.stream=torch.cuda.Stream(device=device)
        self.ready=[torch.cuda.Event() for _ in range(2)]
        self.finished=[torch.cuda.Event() for _ in range(2)]
        self.used=[False,False];self.indices=[None,None]
        self.previous_goals=None;self.bytes=0;self.refreshes=0
        self.stream.wait_stream(torch.cuda.current_stream(device))

    def upload(self, rows, sequence):
        slot=sequence%2
        if self.used[slot]:self.ready[slot].synchronize()
        changed=np.ones(len(rows),dtype=bool) if self.previous_goals is None else np.any(rows[:,4:6]!=self.previous_goals,axis=1)
        self.previous_goals=rows[:,4:6].copy()
        indices=np.flatnonzero(changed)
        self.refreshes+=len(indices);self.bytes+=rows.nbytes
        np.copyto(self.host[slot].numpy(),rows,casting='unsafe')
        with torch.cuda.stream(self.stream):
            if self.used[slot]:self.stream.wait_event(self.finished[slot])
            self.raw[slot].copy_(self.host[slot],non_blocking=True)
            self.indices[slot]=torch.as_tensor(indices,device=self.device)
            self.ready[slot].record(self.stream)

    def get(self, sequence):
        slot=sequence%2
        torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        raw=self.raw[slot];ids=self.indices[slot]
        if len(ids):
            changed=raw.index_select(0,ids)
            self.sim.update_energy_maps(changed,len(ids))
        self.sim.refresh_compact_state_from_raw_batch(raw)
        return self.builder.build(self.sim,raw,materialize_edges=False)

    def release(self, sequence):
        slot=sequence%2
        self.finished[slot].record();self.used[slot]=True


class OriginalHostReplay:
    def __init__(self, corpus, plan, workers=2, depth=4, device='cuda:0'):
        self.corpus,self.plan,self.depth,self.device=corpus,plan,depth,device
        self.pool=HostPrefetch('magat',corpus.grids,corpus.agents,workers=workers,depth=depth)
        self.stream=torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self.device_ready=None;self.bytes=0;self.rebuilds={}

    def _upload(self, prepared, stream=None):
        if stream is None:return prepared.to_device(self.device),None
        with torch.cuda.stream(stream):
            data=prepared.to_device(self.device)
            event=torch.cuda.Event();event.record(stream)
        return data,event

    def get(self, sequence):
        while self.pool.submitted<min(len(self.plan),sequence+self.depth):
            if hasattr(self.corpus,'try_stage'):
                rows=self.corpus.try_stage(self.plan[self.pool.submitted])
                if rows is None:break
            else:rows=self.corpus.stage(self.plan[self.pool.submitted])
            self.pool.submit(rows)
        # Live expert supply must not wait for an entire lookahead queue before
        # accepting the current available batch. Offline ready data still fills Q.
        if self.pool.submitted<=sequence:
            self.pool.submit(self.corpus.stage(self.plan[self.pool.submitted]))
        if self.device_ready is None:
            prepared=self.pool.get();data,_=self._upload(prepared)
        else:
            prepared,data,event=self.device_ready
            torch.cuda.current_stream(self.device).wait_event(event)
            for name in ('x','edge_index','edge_attr','batch','ptr','y','arrived','terminated'):
                value=getattr(data,name,None)
                if value is not None:value.record_stream(torch.cuda.current_stream(self.device))
            self.device_ready=None
        if prepared.stage_sequence!=sequence:raise RuntimeError('source order mismatch')
        self.bytes+=prepared.nbytes
        self.rebuilds[sequence%len(self.pool.processes)]=prepared.cache_rebuilds
        upcoming=self.pool.try_get()
        if upcoming is not None:
            future,event=self._upload(upcoming,self.stream)
            self.device_ready=(upcoming,future,event)
        return data

    def close(self):self.pool.close()

    def poll(self, sequence):
        """Return an available batch without waiting for future expert labels."""
        while self.pool.submitted<min(len(self.plan),sequence+self.depth):
            if hasattr(self.corpus,'try_stage'):
                rows=self.corpus.try_stage(self.plan[self.pool.submitted])
                if rows is None:break
            else:rows=self.corpus.stage(self.plan[self.pool.submitted])
            self.pool.submit(rows)
        if self.device_ready is None:
            prepared=self.pool.try_get()
            if prepared is None:return None
            data,_=self._upload(prepared)
        else:
            prepared,data,event=self.device_ready
            torch.cuda.current_stream(self.device).wait_event(event)
            for name in ('x','edge_index','edge_attr','batch','ptr','y','arrived','terminated'):
                value=getattr(data,name,None)
                if value is not None:value.record_stream(torch.cuda.current_stream(self.device))
            self.device_ready=None
        if prepared.stage_sequence!=sequence:raise RuntimeError('source order mismatch')
        self.bytes+=prepared.nbytes
        self.rebuilds[sequence%len(self.pool.processes)]=prepared.cache_rebuilds
        upcoming=self.pool.try_get()
        if upcoming is not None:
            future,event=self._upload(upcoming,self.stream)
            self.device_ready=(upcoming,future,event)
        return data
