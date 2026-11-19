"""Original E8 expert/prefetch/ring layout, with lifelong or window shuffle."""
import bootstrap
import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from expert.expert_running import RingBuffer,ExtremeMAPFPipeline
from live_source import LiveRingSource
from order_inputs import ConfigurationCorpus,CompactReplay,OriginalHostReplay
from replay_ring import PaperRingReplay,HostRingCorpus
from native_extent_inputs import NativeExtentBatchBuilder
from mapf_cuda.simulation.grids import build_stateless_simulator
from run_il_order import MAP_NAMES,load_maps,configure,make_learner


class UnboundedPlan:
    def __len__(self):return 1000000000
    def __getitem__(self,_):return 0


class LiveInputs(PaperRingReplay):
    def __init__(self,corpus,names,seed,mode,order,path,ring_depth=512,limit=1000000000):
        self.corpus=corpus;self.path=path;self.pipeline=None;self.host=None;self.thread=None
        self.bytes=0;self.refreshes=0
        capacity=corpus.envs*corpus.agents*ring_depth
        if path=='jointloom':
            self.sim=build_stateless_simulator(corpus.grids,num_agents=corpus.agents,device='cuda:0')
            self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
            self.builder=NativeExtentBatchBuilder([g.shape for g in corpus.grids],corpus.agents,'cuda:0')
            self.pipeline=ExtremeMAPFPipeline(self.sim,capacity=capacity,batch_threshold=corpus.envs*corpus.agents,device='cuda:0')
            self.pipeline.initialize();ring=self.pipeline.ring_buffer
        else:ring=RingBuffer(capacity,8,num_envs=corpus.envs,agents_per_env=corpus.agents)
        self.source=LiveRingSource(ring,names,corpus.agents,seed,mode,1 if order=='ordered' else 128,limit)
        if path!='jointloom':self.host=OriginalHostReplay(HostRingCorpus(corpus,self.source),UnboundedPlan(),8,16)


def capture(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'stages.npy').exists():raise FileExistsError(args.out)
    configure();names=MAP_NAMES*(args.envs//4);grids=load_maps(names)
    ring=RingBuffer(args.envs*args.agents*128,8,num_envs=args.envs,agents_per_env=args.agents)
    source=LiveRingSource(ring,names,args.agents,args.seed,args.mode,1,args.updates)
    rows=np.empty((args.updates,args.envs*args.agents,8),dtype=np.uint16)
    try:
        started=time.perf_counter();source.start()
        for step in range(args.updates):
            rows[step]=source.next_rows()
            if (step+1)%64==0:print(json.dumps(dict(stage=step+1,seconds=time.perf_counter()-started)),flush=True)
        while min(source.stats()['published_per_env'])<args.updates:
            source.check();time.sleep(.001)
        seconds=time.perf_counter()-started;stats=source.stats()
        np.save(args.out/'stages.npy',rows);np.save(args.out/'grids.npy',grids)
        result=dict(mode=args.mode,seed=args.seed,agents=args.agents,envs=args.envs,stages=args.updates,
            generation_wall_seconds=seconds,source=stats,map_names=names,
            scope='real lifelong LaCAM labels; original topological initial tasks; natural goals; no learner')
        (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    finally:source.close()


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'result.json').exists():raise FileExistsError(args.out)
    if args.mode=='lifelong' and args.order!='ordered':raise ValueError('no lifelong-by-shuffle matrix planned')
    configure();setup=time.perf_counter();names=MAP_NAMES*(args.envs//4);grids=load_maps(names)
    # Exact model/batch warmup; the source is still stopped and has no labels.
    base=ConfigurationCorpus(bootstrap.input_path('JOINTLOOM_RAW_CORPUS', bootstrap.ROOT/'artifacts/rebuttal/magat-records'),stages=1)
    base.arrays=base.arrays*(args.envs//4);base.envs=args.envs;base.grids=grids
    warm=CompactReplay(base);warm.upload(base.stage([0]*args.envs),0);data=warm.get(0)
    learner=make_learner(args.seed)
    for _ in range(20):learner.train_step_from_batch(data)
    torch.cuda.synchronize();del learner;gc.collect();learner=make_learner(args.seed)
    learner.model.eval();initial_eval=float(learner.loss_from_batch(data));learner.model.train()
    warm.release(0);del warm,data;gc.collect()
    corpus=SimpleNamespace(grids=grids,envs=args.envs,agents=args.agents)
    inputs=LiveInputs(corpus,names,args.seed,args.mode,args.order,args.path)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup
    torch.manual_seed(args.seed+1000);torch.cuda.reset_peak_memory_stats();losses=[];wait_s=0.
    started=time.perf_counter();inputs.start()
    try:
        while time.perf_counter()-started<args.seconds:
            t=time.perf_counter();batch=inputs.poll(len(losses))
            if batch is None:
                time.sleep(.0005);wait_s+=time.perf_counter()-t
                continue
            wait_s+=time.perf_counter()-t
            losses.append(learner.train_step_from_batch(batch))
            if len(losses)%128==0:
                torch.cuda.synchronize()
                print(json.dumps(dict(updates=len(losses),elapsed_seconds=time.perf_counter()-started)),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started;stats=inputs.source.stats()
        values=torch.stack(losses).cpu().numpy() if losses else np.empty(0,dtype=np.float32)
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite live training loss')
        np.save(args.out/'losses.npy',values)
        torch.save(dict(model=learner.model.state_dict(),optimizer=learner.optimizer.state_dict(),
            scheduler=learner.scheduler.state_dict(),optimizer_updates=len(losses)),args.out/'checkpoint.pt')
        result=dict(path=args.path,mode=args.mode,order=args.order,seed=args.seed,agents=args.agents,envs=args.envs,
            batch=args.agents*args.envs,ring_depth=512,host_workers=8,host_depth=16,
            target_seconds=args.seconds,wall_seconds=seconds,optimizer_updates=len(losses),
            samples_per_second=len(losses)*args.agents*args.envs/seconds,setup_seconds=setup_seconds,
            initial_common_eval_loss=initial_eval,first_loss=float(values[0]) if len(values) else None,
            last_loss=float(values[-1]) if len(values) else None,
            consumer_input_seconds=wait_s,h2d_bytes=inputs.bytes,cache_refreshes=inputs.refreshes,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),source=stats,
            map_names=names,cpu_affinity=list(range(16)),precision='float32; TF32 disabled',
            scope='source released empty inside timer; actual expert/prefetch/DMA/update overlap; fixed time not fixed record sequence',
            shuffle_placement='producer owns a complete128-configuration window before ordered ring publication' if args.order!='ordered' else 'none',
            cpu_schedule='original H8/Q16 components; submit available lookahead without blocking on a full Q')
        result['timing_protocol']='deadline-aware source polling; stop waiting for new labels at target time; finish current update; teardown excluded'
        (args.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['path','mode','order','wall_seconds','optimizer_updates','samples_per_second']}),flush=True)
    finally:inputs.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','capture'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--envs',type=int,default=8);p.add_argument('--agents',type=int,default=512)
    p.add_argument('--mode',choices=['standard_mapf','lifelong'],default='standard_mapf')
    p.add_argument('--order',choices=['ordered','window_shuffle_128'],default='ordered')
    p.add_argument('--path',choices=['jointloom','cpu_cached_prefetch'],default='jointloom')
    p.add_argument('--seconds',type=float,default=300.);p.add_argument('--updates',type=int,default=512)
    args=p.parse_args();mp.set_start_method('spawn',force=True)
    if args.envs%4:raise ValueError('repeat the original four maps')
    if args.agents!=512:raise ValueError('this runner aligns the original A512 workload')
    capture(args) if args.command=='capture' else run(args)
