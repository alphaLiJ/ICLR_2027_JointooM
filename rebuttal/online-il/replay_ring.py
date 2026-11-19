"""Existing complete-stage ring/DMA, with offline per-map configuration orders."""
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from expert.expert_running import RingBuffer,ExtremeMAPFPipeline
from mapf_cuda.simulation.grids import build_stateless_simulator
from native_extent_inputs import NativeExtentBatchBuilder


def _producer(directory,env_id,agents,indices,ring,ready,start,stop,errors):
    try:
        from expert.expert_running import configure_cpu_thread_limits
        configure_cpu_thread_limits()
        directory=Path(directory)
        if (directory/'stages.npy').exists():
            blocks=np.load(directory/'stages.npy',mmap_mode='r')[:,env_id*agents:(env_id+1)*agents].copy()
        else:blocks=np.load(directory/f'env-{env_id}.npy',mmap_mode='r').copy()
        ready.set();start.wait();previous=None
        for index in indices:
            if stop.is_set():break
            block=blocks[int(index)].copy();block[:,0]=env_id
            changed=np.ones(agents,dtype=bool) if previous is None else np.any(block[:,4:6]!=previous,axis=1)
            block[:,7]=changed
            previous=block[:,4:6].copy()
            ring.reserve_and_write(block)
        stop.wait()
    except BaseException:
        errors.put(traceback.format_exc());ready.set()


class ReplayRingSource:
    def __init__(self,ring,corpus,plan):
        self.ring=ring;self.sequence=0;self.count=len(plan)
        self.start_event=mp.Event();self.stop_event=mp.Event();self.errors=mp.Queue()
        self.ready=[mp.Event() for _ in range(corpus.envs)]
        self.processes=[]
        for e,ready in enumerate(self.ready):
            p=mp.Process(target=_producer,args=(str(corpus.directory),e,corpus.agents,plan[:,e],
                ring,ready,self.start_event,self.stop_event,self.errors),daemon=True)
            p.start();self.processes.append(p)
        began=time.monotonic()
        while not all(r.is_set() for r in self.ready):
            self.check()
            if time.monotonic()-began>180:raise TimeoutError('replay source initialization')
            time.sleep(.01)
        self.check()

    def check(self):
        if not self.errors.empty():raise RuntimeError(self.errors.get())
        if any(p.exitcode is not None for p in self.processes):raise RuntimeError('replay producer exited')

    def start(self):self.start_event.set()

    def next_rows(self):
        began=time.monotonic()
        while not self.ring.is_stage_ready(self.sequence):
            self.check()
            if time.monotonic()-began>180:raise TimeoutError('replay source stalled')
            time.sleep(.0005)
        lo,hi=self.ring._stage_row_bounds(self.sequence)
        rows=self.ring.cpu_buffer[lo:hi].numpy().astype(np.uint16,copy=True)
        self.ring.release_stage(self.sequence)
        self.sequence+=1;self.ring.shared_compute_ptr.value=self.sequence*self.ring.stage_rows
        return rows

    def try_next_rows(self):
        return self.next_rows() if self.ring.is_stage_ready(self.sequence) else None

    def close(self):
        self.stop_event.set();self.start_event.set()
        for p in self.processes:
            p.join(1)
            if p.is_alive():p.terminate();p.join()
        self.errors.cancel_join_thread();self.errors.close()


class HostRingCorpus:
    def __init__(self,corpus,source):
        self.grids,self.agents,self.source=corpus.grids,corpus.agents,source

    def stage(self,_):
        rows=self.source.next_rows()
        # CPU builders have independent predecessor caches (round-robin workers).
        # They already compare their own goals, so the producer's bit is redundant.
        rows[:,7]=0
        return rows

    def try_stage(self,_):
        rows=self.source.try_next_rows()
        if rows is not None:rows[:,7]=0
        return rows


class PaperRingReplay:
    def __init__(self,corpus,plan,path,workers=2,depth=4,ring_depth=128):
        self.path=path;self.corpus=corpus;self.bytes=0;self.refreshes=0
        self.pipeline=None;self.host=None;self.thread=None
        capacity=corpus.envs*corpus.agents*ring_depth
        if path=='jointloom':
            self.sim=build_stateless_simulator(corpus.grids,num_agents=corpus.agents,device='cuda:0')
            self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
            self.builder=NativeExtentBatchBuilder([g.shape for g in corpus.grids],corpus.agents,'cuda:0')
            self.pipeline=ExtremeMAPFPipeline(self.sim,capacity=capacity,
                batch_threshold=corpus.envs*corpus.agents,device='cuda:0')
            self.pipeline.initialize();ring=self.pipeline.ring_buffer
        else:ring=RingBuffer(capacity,8,num_envs=corpus.envs,agents_per_env=corpus.agents)
        self.source=ReplayRingSource(ring,corpus,plan)
        if path!='jointloom':
            from order_inputs import OriginalHostReplay
            self.host=OriginalHostReplay(HostRingCorpus(corpus,self.source),plan,workers,depth)

    def start(self):
        self.source.start()
        if self.pipeline:self.thread=self.pipeline.dma_worker.start()

    def get(self,sequence):
        if self.host:
            data=self.host.get(sequence);self.bytes=self.host.bytes
            self.refreshes=sum(self.host.rebuilds.values());return data
        pipeline=self.pipeline;size=self.corpus.envs*self.corpus.agents
        began=time.monotonic()
        while pipeline.dma_worker.dma_read_ptr-pipeline.compute_ptr<size:
            self.source.check()
            if time.monotonic()-began>180:raise TimeoutError('paper DMA source stalled')
            time.sleep(.0005)
        pipeline.dma_event.synchronize()
        raw=pipeline.extract_env_aligned_batch(size,self.corpus.agents,self.corpus.envs)
        if raw is None:raise RuntimeError('complete source stage disappeared')
        changed=raw[raw[:,7]!=0];self.refreshes+=len(changed);self.bytes+=raw.numel()*raw.element_size()
        if len(changed):self.sim.update_energy_maps(changed,len(changed))
        self.sim.refresh_compact_state_from_raw_batch(raw)
        return self.builder.build(self.sim,raw,materialize_edges=False)

    def release(self,_):pass

    def poll(self,sequence):
        """Nonblocking source acquisition for fixed-wall-time live runs."""
        if self.host:
            data=self.host.poll(sequence)
            self.bytes=self.host.bytes;self.refreshes=sum(self.host.rebuilds.values())
            return data
        self.source.check()
        size=self.corpus.envs*self.corpus.agents
        if self.pipeline.dma_worker.dma_read_ptr-self.pipeline.compute_ptr<size:return None
        return self.get(sequence)

    def close(self):
        if self.pipeline:
            self.pipeline.shutdown()
            if self.thread:self.thread.join(2)
        if self.host:self.host.close()
        self.source.close()
