"""Same-corpus MAGAT order learning and original cached-CPU replay controls."""
import bootstrap
import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import yaml
from bootstrap import SOURCE
from config_sampler import ORDERS,configuration_order
from order_inputs import ConfigurationCorpus,CompactReplay,OriginalHostReplay
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter,PyGBatchBuilder,RuntimePyGBatch


MAP_NAMES=['mazes-s0_wc8_od55','mazes-s1_wc3_od70','mazes-s2_wc8_od25','mazes-s3_wc6_od35']
LIBRARY=bootstrap.input_path('JOINTLOOM_LACAM_LIBRARY', bootstrap.SOURCE/'src/real_expert_alg/lacam/liblacam.so')
VALIDATION=bootstrap.input_path('JOINTLOOM_VALIDATION', bootstrap.SOURCE/'artifacts/rebuttal/validation')


def configure():
    os.sched_setaffinity(0,set(range(16)));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def load_maps(names):
    catalog=yaml.safe_load((SOURCE/'maps/maps.yaml').read_text())
    return np.stack([np.array([[c in '#@T' for c in row] for row in catalog[name].strip().splitlines()],dtype=np.uint8) for name in names])


def _capture_worker(directory,env_id,map_name,stages,agents,seed):
    out=Path(directory)
    try:
        configure()
        from expert.expert_running import LacamExpertPolicy,put_maps_into_registry,initial_refresh_flags,no_refresh_flags
        from mapf_cuda.training.topology_async import _build_pogema_topology_map_env,_build_step_data
        put_maps_into_registry(str(SOURCE/'maps/maps.yaml'))
        env=_build_pogema_topology_map_env(num_agents=agents,map_name=map_name,seed=seed+env_id,max_episode_steps=256,collision_system='soft')
        policy=LacamExpertPolicy(lib_path=str(LIBRARY.resolve()),timeouts=[1.,5.,10.,30.]);policy.reset_states(env)
        observations=env.env.unwrapped._obs();flags=initial_refresh_flags(agents)
        target=np.lib.format.open_memmap(out/f'env-{env_id}.npy',mode='w+',dtype=np.uint16,shape=(stages,agents,8))
        episode_ids=np.empty(stages,dtype=np.int32);timesteps=np.empty(stages,dtype=np.int32)
        episode=0;timestep=0;expert_s=0.;env_s=0.;new_calls=0
        started=time.perf_counter()
        for step in range(stages):
            t=time.perf_counter();actions=policy.act(observations);expert_s+=time.perf_counter()-t
            new_calls+=int(np.any(flags))
            target[step]=_build_step_data(observations,actions,env_id=env_id,reset_flag=flags)
            episode_ids[step]=episode;timesteps[step]=timestep
            t=time.perf_counter();_,_,terminated,truncated,_=env.step(actions.astype(np.int64,copy=False))
            if all(terminated) or all(truncated):
                env.reset();observations=env.env.unwrapped._obs();policy.reset_states(env)
                flags=initial_refresh_flags(agents);episode+=1;timestep=0
            else:
                observations=env.env.unwrapped._obs();flags=no_refresh_flags(agents);timestep+=1
            env_s+=time.perf_counter()-t
            if (step+1)%4096==0:
                target.flush()
                (out/f'progress-env-{env_id}.json').write_text(json.dumps(dict(stages=step+1,total=stages,seconds=time.perf_counter()-started)))
        target.flush();np.savez_compressed(out/f'index-env-{env_id}.npz',episode_id=episode_ids,timestep=timesteps)
        (out/f'report-env-{env_id}.json').write_text(json.dumps(dict(stages=stages,map_name=map_name,seed=seed+env_id,
            episodes_seen=int(episode_ids.max())+1,expert_seconds=expert_s,environment_seconds=env_s,
            goal_version_calls=new_calls,wall_seconds=time.perf_counter()-started)))
    except BaseException:
        (out/f'error-env-{env_id}.txt').write_text(traceback.format_exc());raise


def capture(out,stages,seed,agents):
    out.mkdir(parents=True,exist_ok=True)
    if any(out.glob('env-*.npy')):raise FileExistsError(out)
    configure();np.save(out/'grids.npy',load_maps(MAP_NAMES))
    ctx=mp.get_context('spawn');processes=[];started=time.perf_counter()
    for e,name in enumerate(MAP_NAMES):
        p=ctx.Process(target=_capture_worker,args=(str(out),e,name,stages,agents,seed));p.start();processes.append(p)
    try:
        for p in processes:
            p.join()
            if p.exitcode!=0:raise RuntimeError('expert capture worker failed; see error-env files')
        reports=[json.loads((out/f'report-env-{e}.json').read_text()) for e in range(4)]
        summary=dict(stages=stages,agents=agents,envs=4,map_names=MAP_NAMES,seed=seed,
            generation_wall_seconds=time.perf_counter()-started,reports=reports,
            protocol='original fixed-goal LaCAM; horizon256; original env reset; [1,5,10,30] timeouts; no learner')
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:
        for p in processes:
            if p.is_alive():p.terminate();p.join()


def validation_batches():
    arrays=[];names=[]
    for i in range(1,5):
        with np.load(VALIDATION/f'val_{i}.npz') as d:
            arrays.append(d['raw_batches'].copy());names.append(json.loads(str(d['metadata_json']))['map_name'])
    corpus=ConfigurationCorpus.__new__(ConfigurationCorpus)
    corpus.grids=load_maps(names);corpus.envs=4;corpus.agents=arrays[0].shape[1]
    corpus.stages=min(map(len,arrays));corpus.arrays=arrays
    compact=CompactReplay(corpus);batches=[]
    for step in range(corpus.stages):
        raw=corpus.stage([step]*4);compact.upload(raw,step);data=compact.get(step)
        edge,attr=PyGBatchBuilder._materialize_valid_edges(data.edge_index_storage,data.edge_attr_storage,data.num_edges)
        batches.append(RuntimePyGBatch(x=data.x.clone(),edge_index=edge.clone(),edge_attr=attr.clone(),
            y=data.y.clone(),arrived=data.arrived.clone(),batch=data.batch.clone(),ptr=data.ptr.clone()))
        compact.release(step)
    torch.cuda.synchronize();return batches


@torch.no_grad()
def validate(learner,batches,step):
    learner.model.eval();correct=torch.zeros((),dtype=torch.int64,device='cuda:0')
    nonstay_correct=correct.clone();count=0;nonstay_count=correct.clone()
    for data in batches:
        prediction=learner.model(data.x,data).argmax(-1);matched=prediction==data.y;nonstay=data.y!=0
        correct+=matched.sum();nonstay_correct+=(matched&nonstay).sum();nonstay_count+=nonstay.sum();count+=len(data.y)
    result=dict(optimizer_step=step,overall_accuracy=int(correct)/count,
        nonstay_accuracy=int(nonstay_correct)/int(nonstay_count),agent_labels=count)
    learner.model.train();return result


def make_learner(seed):
    torch.manual_seed(seed)
    return MAGATRuntimeAdapter(device='cuda:0',lr_scheduler='cosine-annealing',scheduler_total_steps=100000,train_on_arrived_agents=True)


def run(args):
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    configure();setup=time.perf_counter()
    corpus=ConfigurationCorpus(args.corpus,stages=args.updates)
    if corpus.stages!=args.updates:raise ValueError('collect distinct stages; short corpus must not be cycled')
    plan=configuration_order(corpus.stages,corpus.envs,args.order,args.seed+10000)
    np.save(out/'configuration-indices.npy',plan)
    common=corpus.stage([0]*corpus.envs)
    warm=CompactReplay(corpus);warm.upload(common,0);data=warm.get(0)
    learner=make_learner(args.seed)
    for _ in range(20):learner.train_step_from_batch(data)
    torch.cuda.synchronize();del learner;gc.collect()
    learner=make_learner(args.seed);learner.model.eval()
    initial_eval_loss=float(learner.loss_from_batch(data));learner.model.train()
    warm.release(0);del data,warm;gc.collect()
    validation=validation_batches() if args.phase=='learn' else None
    validation_rows=[validate(learner,validation,0)] if validation else []
    if validation_rows:validation_rows[0]['training_elapsed_seconds']=0.
    (out/'initial-validation.json').write_text(json.dumps(validation_rows))
    paper=None
    if args.transport=='paper_ring':
        from replay_ring import PaperRingReplay
        paper=PaperRingReplay(corpus,plan,args.path,args.workers,args.depth,128)
    compact=CompactReplay(corpus) if not paper and args.path=='jointloom' else None
    host=OriginalHostReplay(corpus,plan,args.workers,args.depth) if not paper and args.path=='cpu_cached_prefetch' else None
    losses=torch.empty(args.updates,device='cuda:0');wait_s=0.;checkpoint_s=0.;validation_s=0.
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup
    torch.manual_seed(args.seed+1000);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    try:
        if paper:paper.start()
        if compact:compact.upload(corpus.stage(plan[0]),0)
        for step in range(args.updates):
            t=time.perf_counter()
            data=paper.get(step) if paper else compact.get(step) if compact else host.get(step)
            wait_s+=time.perf_counter()-t
            if compact and step+1<args.updates:compact.upload(corpus.stage(plan[step+1]),step+1)
            losses[step].copy_(learner.train_step_from_batch(data))
            if compact:compact.release(step)
            completed=step+1
            if validation and (completed in [1000,10000,25000,50000,100000] or completed==args.updates):
                torch.cuda.synchronize();t=time.perf_counter();validation_rows.append(validate(learner,validation,completed));torch.cuda.synchronize();validation_s+=time.perf_counter()-t
                validation_rows[-1]['training_elapsed_seconds']=time.perf_counter()-started
                (out/'validation.json').write_text(json.dumps(validation_rows,indent=2))
                t=time.perf_counter()
                torch.save(dict(model=learner.model.state_dict(),optimizer=learner.optimizer.state_dict(),
                    scheduler=learner.scheduler.state_dict(),optimizer_updates=completed),out/f'checkpoint-{completed:06d}.pt')
                checkpoint_s+=time.perf_counter()-t
            if completed%1000==0:
                torch.cuda.synchronize()
                progress=dict(path=args.path,order=args.order,seed=args.seed,updates=completed,
                    elapsed_seconds=time.perf_counter()-started,last_loss=float(losses[step]))
                (out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        values=losses.cpu().numpy()
        if not np.isfinite(values).all():raise FloatingPointError('nonfinite training loss')
        np.save(out/'losses.npy',values)
        t=time.perf_counter()
        torch.save(dict(model=learner.model.state_dict(),optimizer=learner.optimizer.state_dict(),
            scheduler=learner.scheduler.state_dict(),optimizer_updates=args.updates),out/'checkpoint.pt')
        save_outside=time.perf_counter()-t
        result=dict(phase=args.phase,path=args.path,order=args.order,seed=args.seed,corpus=str(args.corpus),
            envs=corpus.envs,agents=corpus.agents,optimizer_updates=args.updates,wall_seconds=seconds,
            samples_per_second=args.updates*corpus.envs*corpus.agents/seconds,setup_seconds=setup_seconds,
            initial_common_eval_loss=initial_eval_loss,first_training_loss=float(values[0]),last_training_loss=float(values[-1]),
            mean_loss=float(values.mean()),h2d_bytes=paper.bytes if paper else compact.bytes if compact else host.bytes,
            cache_refreshes=paper.refreshes if paper else compact.refreshes if compact else sum(host.rebuilds.values()),
            consumer_input_seconds=wait_s,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            validation=validation_rows,validation_seconds_in_wall=validation_s,checkpoint_seconds_in_wall=checkpoint_s,
            final_checkpoint_seconds_outside_wall=save_outside,optimizer_config=learner.training_config(),
            optimizer_defaults=learner.optimizer.defaults,
            host_workers=args.workers,host_depth=args.depth,cpu_affinity=list(range(16)),
            compact_transport=('original complete-stage ring128/DMA; shuffled full configurations; no live experts' if paper else
                'two asynchronous copy slots; full-configuration replay; no live expert ring'),
            precision='float32; TF32 disabled',
            scope='same complete-configuration corpus; no expert cost; no cycling; timing/learning cohort separate from live online')
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ['path','order','seed','optimizer_updates','wall_seconds','first_training_loss','last_training_loss','cache_refreshes']}),flush=True)
    finally:
        if paper:paper.close()
        if host:host.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['capture','run'])
    p.add_argument('--corpus',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=100000)
    p.add_argument('--agents',type=int,default=512)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--order',choices=ORDERS,default='ordered')
    p.add_argument('--path',choices=['jointloom','cpu_cached_prefetch'],default='jointloom')
    p.add_argument('--phase',choices=['replay','learn'],default='replay')
    p.add_argument('--transport',choices=['paper_ring','two_slot'],default='paper_ring')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--depth',type=int,default=4)
    args=p.parse_args();mp.set_start_method('spawn',force=True)
    if args.command=='capture':capture(args.out,args.updates,args.seed,args.agents)
    else:run(args)


if __name__=='__main__':main()
