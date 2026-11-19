"""Small causal control: fixed LaCAM queries, canonical labels, isolated cores.

Actual LaCAM computations are retained. Returned plans do not determine the next
query; both learners receive the same recorded expert labels. This is a workload
attribution control, not an online policy-learning or task-quality experiment.
"""
import bootstrap
import argparse
from datetime import datetime,timezone
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import psutil
import torch
from expert.expert_running import RingBuffer,ExtremeMAPFPipeline
from mapf_cuda.simulation.grids import build_stateless_simulator
from native_extent_inputs import NativeExtentBatchBuilder
from order_inputs import ConfigurationCorpus,CompactReplay,OriginalHostReplay
from replay_ring import ReplayRingSource,PaperRingReplay,HostRingCorpus
from run_il_order import configure,make_learner,LIBRARY


def prepare(source,out,queries):
    if (out/'selection.json').exists():return
    corpus=ConfigurationCorpus(source)
    selected=[];blocks=[]
    for e in range(8):
        base=e%4;replica=e//4
        choices=np.flatnonzero(np.any(corpus.arrays[base][:,:,7]!=0,axis=1))
        indices=choices[replica*queries:(replica+1)*queries]
        if len(indices)!=queries:raise ValueError('not enough distinct original replanning states')
        rows=corpus.arrays[base][indices].copy();rows[:,:,0]=e
        blocks.append(rows);selected.append(indices.tolist())
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'stages.npy',np.concatenate(blocks,axis=1))
    np.save(out/'grids.npy',np.concatenate((corpus.grids,corpus.grids)))
    (out/'selection.json').write_text(json.dumps(dict(source=str(source),queries_per_env=queries,
        source_stage_ids=selected,selection='first natural goal-change states; replica1 uses the next disjoint query block',
        labels='canonical recorded labels shared by both consumers; newly computed plans are diagnostic only'),indent=2))


def _query_worker(directory,e,agents,count,ring,ready,start,stop,errors,timings):
    try:
        os.sched_setaffinity(0,{e})
        from expert.expert_running import configure_cpu_thread_limits,LacamExpertPolicy
        from pogema_path import OriginalPogemaLifelong
        configure_cpu_thread_limits()
        directory=Path(directory);grids=np.load(directory/'grids.npy')
        all_rows=np.load(directory/'stages.npy',mmap_mode='r')
        rows=all_rows[:,e*agents:(e+1)*agents].copy()
        tasks=SimpleNamespace(grid=grids[e],grids=grids[e:e+1],positions=rows[0,None,:,2:4].astype(np.int16),
            goals=rows[0,None,:,4:6].astype(np.int16))
        env=OriginalPogemaLifelong(tasks,prepare=False,with_inputs=False)
        policy=LacamExpertPolicy(lib_path=str(LIBRARY.resolve()),timeouts=[1.,5.,10.,30.])
        ready.set();start.wait();began=time.perf_counter();previous=None
        for i,raw in enumerate(rows):
            env.load_rows(raw);policy.reset_states(env.envs[0])
            t=time.perf_counter();fresh=policy.act(env.observations[0]);seconds=time.perf_counter()-t
            record=(e*count+i)*4
            timings[record:record+4]=[seconds,time.perf_counter()-began,
                float(np.mean(fresh==raw[:,6])),float(np.count_nonzero(fresh))]
            block=raw.copy()
            changed=np.ones(agents,dtype=bool) if previous is None else np.any(block[:,4:6]!=previous,axis=1)
            previous=block[:,4:6].copy();block[:,7]=changed
            ring.reserve_and_write(block)
        stop.wait()
    except BaseException:
        errors.put(traceback.format_exc());ready.set()


class QuerySource(ReplayRingSource):
    def __init__(self,ring,corpus):
        self.ring=ring;self.sequence=0;self.count=corpus.stages
        self.start_event=mp.Event();self.stop_event=mp.Event();self.errors=mp.Queue()
        self.ready=[mp.Event() for _ in range(corpus.envs)]
        self.timings=mp.Array('d',corpus.envs*corpus.stages*4,lock=False)
        self.processes=[]
        for e,ready in enumerate(self.ready):
            p=mp.Process(target=_query_worker,args=(str(corpus.directory),e,corpus.agents,corpus.stages,
                ring,ready,self.start_event,self.stop_event,self.errors,self.timings),daemon=True)
            p.start();self.processes.append(p)
        began=time.monotonic()
        while not all(r.is_set() for r in self.ready):
            self.check()
            if time.monotonic()-began>180:raise TimeoutError('query-source initialization')
            time.sleep(.01)
        self.check()


class QueryInputs(PaperRingReplay):
    def __init__(self,corpus,path):
        self.path=path;self.corpus=corpus;self.bytes=0;self.refreshes=0
        self.pipeline=None;self.host=None;self.thread=None
        capacity=corpus.envs*corpus.agents*512
        if path=='jointloom':
            self.sim=build_stateless_simulator(corpus.grids,num_agents=corpus.agents,device='cuda:0')
            self.sim.pyg_builder_mode='local_gather';self.sim.pyg_local_gather_impl='auto'
            self.builder=NativeExtentBatchBuilder([g.shape for g in corpus.grids],corpus.agents,'cuda:0')
            self.pipeline=ExtremeMAPFPipeline(self.sim,capacity=capacity,batch_threshold=corpus.envs*corpus.agents,device='cuda:0')
            self.pipeline.initialize();ring=self.pipeline.ring_buffer
        else:ring=RingBuffer(capacity,8,num_envs=corpus.envs,agents_per_env=corpus.agents)
        self.source=QuerySource(ring,corpus)
        if path!='jointloom':
            self.host=OriginalHostReplay(HostRingCorpus(corpus,self.source),range(corpus.stages),8,16)


def child(corpus_dir,path,out):
    if (out/'result.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);configure()
    # The same eight expert cores in both conditions; input workers use the
    # other eight cores, rather than sharing all16 with those experts.
    os.sched_setaffinity(0,set(range(8,16)))
    corpus=ConfigurationCorpus(corpus_dir);setup=time.perf_counter()
    common=corpus.stage([0]*corpus.envs)
    warm=CompactReplay(corpus);warm.upload(common,0);batch=warm.get(0)
    learner=make_learner(42)
    for _ in range(20):learner.train_step_from_batch(batch)
    torch.cuda.synchronize();del learner;gc.collect();learner=make_learner(42)
    learner.model.eval();initial=float(learner.loss_from_batch(batch));learner.model.train()
    warm.release(0);del warm,batch;gc.collect()
    inputs=QueryInputs(corpus,path)
    os.sched_setaffinity(0,{15})
    torch.manual_seed(1042);losses=torch.empty(corpus.stages,device='cuda:0')
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup
    began=time.perf_counter();inputs.start();input_seconds=0.
    try:
        for step in range(corpus.stages):
            t=time.perf_counter();batch=inputs.get(step);input_seconds+=time.perf_counter()-t
            losses[step].copy_(learner.train_step_from_batch(batch))
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        times=np.array(inputs.source.timings).reshape(corpus.envs,corpus.stages,4)
        values=losses.cpu().numpy();np.save(out/'losses.npy',values);np.save(out/'per-query.npy',times)
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite control loss')
        result=dict(path=path,corpus=str(corpus_dir),wall_seconds=seconds,optimizer_updates=corpus.stages,
            setup_seconds=setup_seconds,input_acquisition_seconds=input_seconds,initial_common_eval_loss=initial,
            first_loss=float(values[0]),last_loss=float(values[-1]),h2d_bytes=inputs.bytes,
            expert_seconds_per_worker=times[:,:,0].sum(1).tolist(),query_finish_seconds_per_worker=times[:,-1,1].tolist(),
            expert_core_by_env=list(range(8)),input_worker_cores=list(range(8,16)),consumer_core=15,
            fresh_vs_canonical_action_agreement=times[:,:,2].tolist(),
            scope='same actual LaCAM query sequence and canonical training labels; no query feedback; fixed work; resource-isolated attribution control; not a new online-quality experiment')
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:inputs.close()


def compare(args):
    root=args.out;root.mkdir(parents=True,exist_ok=True)
    state=dict(status='waiting',wait_pid=args.wait_pid,resume_queue=args.resume_queue,runs=[],pairs=[])
    def save():(root/'control-state.json').write_text(json.dumps(state,indent=2))
    save()
    try:
        if args.wait_pid:
            while psutil.pid_exists(args.wait_pid):
                try:
                    if psutil.Process(args.wait_pid).status()==psutil.STATUS_ZOMBIE:break
                except psutil.NoSuchProcess:break
                time.sleep(5)
        prepare(args.source,root/'queries',args.queries)
        state['status']='running';save()
        for rep in range(args.repetitions):
            paths=['cpu_cached_prefetch','jointloom'] if rep%2==0 else ['jointloom','cpu_cached_prefetch']
            for path in paths:
                out=root/f'repeat-{rep}'/path
                if not (out/'result.json').exists():
                    with (root/f'repeat-{rep}-{path}.log').open('a') as log:
                        proc=subprocess.Popen([sys.executable,__file__,'child','--source',str(root/'queries'),
                            '--path',path,'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=Path(__file__).parent.parent)
                        state['active']=dict(pid=proc.pid,repetition=rep,path=path);save()
                        code=proc.wait()
                    if code:raise RuntimeError(f'query control failed: {path} repeat{rep}, see its log')
                state['runs'].append(json.loads((out/'result.json').read_text()));save()
            a=json.loads((root/f'repeat-{rep}/cpu_cached_prefetch/result.json').read_text())
            b=json.loads((root/f'repeat-{rep}/jointloom/result.json').read_text())
            state['pairs'].append(dict(repetition=rep,speedup=a['wall_seconds']/b['wall_seconds'],
                cpu_seconds=a['wall_seconds'],jointloom_seconds=b['wall_seconds']));save()
        state['status']='complete';state.pop('active',None);save()
    except BaseException:
        state['status']='failed';state['error']=traceback.format_exc();save();raise
    finally:
        if args.resume_queue:
            try:
                owner=psutil.Process(args.resume_queue)
                if any(x.endswith('run_rebuttal_queue.py') for x in owner.cmdline()):
                    os.kill(args.resume_queue,signal.SIGCONT)
            except (ProcessLookupError,psutil.NoSuchProcess,psutil.ZombieProcess):pass


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['child','compare'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--path',choices=['cpu_cached_prefetch','jointloom']);p.add_argument('--queries',type=int,default=16)
    p.add_argument('--repetitions',type=int,default=3);p.add_argument('--wait-pid',type=int);p.add_argument('--resume-queue',type=int)
    args=p.parse_args();mp.set_start_method('spawn',force=True)
    child(args.source,args.path,args.out) if args.command=='child' else compare(args)
