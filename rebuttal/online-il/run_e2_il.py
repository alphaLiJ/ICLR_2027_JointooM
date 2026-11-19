"""Lifelong original-Pogema versus JointLoom CE input/online training paths."""
import bootstrap
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from tasks import make_tasks
from expert_stream import ExpertPool,save_worker_reports
from lifelong_il import LifelongILConsumer
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter


def configure(c):
    os.sched_setaffinity(0,set(c['cpu_affinity']));torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def make_learner(c):
    torch.manual_seed(c['model_seed'])
    return MAGATRuntimeAdapter(device=c['device'],lr_scheduler='cosine-annealing',
        scheduler_total_steps=100000,train_on_arrived_agents=True)


def get_tasks(c):return make_tasks(c['map_name'],c['agents'],c['envs'],c['task_seed'])


def capture(c,out):
    if (out/'summary.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);configure(c);tasks=get_tasks(c)
    (out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    records=np.empty((c['updates'],c['envs']*c['agents'],8),dtype=np.int16)
    pool=ExpertPool(tasks,c,False,c['updates'])
    try:
        began=time.perf_counter();pool.start()
        for step in range(c['updates']):
            records[step]=pool.get(step)['raw']
            if (step+1)%64==0:print(json.dumps(dict(event='capture',steps=step+1,seconds=time.perf_counter()-began)),flush=True)
        reports=pool.finish();seconds=time.perf_counter()-began
        np.save(out/'records.npy',records);np.save(out/'grid.npy',tasks.grid)
        np.savez_compressed(out/'initial-tasks.npz',positions=tasks.positions,goals=tasks.goals,candidates=tasks.candidates,cells=tasks.cells)
        stats=save_worker_reports(out,reports)
        summary=dict(config=c,generation_wall_seconds=seconds,stages=len(records),
            configs=len(records)*c['envs'],agent_records=len(records)*c['envs']*c['agents'],
            rows_bytes=records.nbytes,**stats,
            scope='real LaCAM commands and pre-transition lifelong states; natural goals; finite source; no model training')
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    finally:pool.close()


def run_child(c,phase,path,trace_dir,out):
    if (out/'result.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);configure(c);setup_start=time.perf_counter();tasks=get_tasks(c)
    records=np.load(trace_dir/'records.npy')
    if records.shape[1:]!=(c['envs']*c['agents'],8):raise ValueError('trace/config shape mismatch')
    if phase=='consume' and c['updates']>len(records):raise ValueError('capture enough distinct stages first')
    warm=LifelongILConsumer('jointloom',tasks,c['device']);warm.upload(dict(raw=records[0]),0);data=warm.get(0)
    learner=make_learner(c)
    for _ in range(c['warmup_updates']):learner.train_step_from_batch(data)
    warm.release(0);torch.cuda.synchronize();del learner;gc.collect()
    pool=None
    if phase=='online':pool=ExpertPool(tasks,c,path=='pogema_original',c['updates'])
    elif path=='pogema_original':pool=ExpertPool(tasks,c,True,c['updates'],str((trace_dir/'records.npy').resolve()))
    learner=make_learner(c);learner.model.eval()
    with torch.no_grad():initial_loss=float(learner.loss_from_batch(data).item())
    learner.model.train();torch.cuda.synchronize();del data,warm;gc.collect()
    consumer=LifelongILConsumer(path,tasks,c['device'])
    losses=torch.empty(c['updates'],device=c['device'])
    consumed=np.empty((c['updates'],c['envs']*c['agents'],8),dtype=np.int16)
    torch.cuda.synchronize();setup_seconds=time.perf_counter()-setup_start
    torch.manual_seed(c['model_seed']+1000);torch.cuda.reset_peak_memory_stats()
    wait_seconds=0.;upload_seconds=0.;model_host_seconds=0.
    try:
        began=time.perf_counter()
        if pool:pool.start()
        for step in range(c['updates']):
            t=time.perf_counter();packet=pool.get(step) if pool else dict(raw=records[step])
            wait_seconds+=time.perf_counter()-t;consumed[step]=packet['raw']
            t=time.perf_counter();consumer.upload(packet,step);upload_seconds+=time.perf_counter()-t
            t=time.perf_counter();batch=consumer.get(step)
            losses[step].copy_(learner.train_step_from_batch(batch));consumer.release(step)
            model_host_seconds+=time.perf_counter()-t
            progress_every=16 if phase=='online' else 128
            if (step+1)%progress_every==0:print(json.dumps(dict(event='training',phase=phase,path=path,updates=step+1,elapsed_host_seconds=time.perf_counter()-began)),flush=True)
        reports=pool.finish() if pool else None
        torch.cuda.synchronize();seconds=time.perf_counter()-began
        peak_gpu=torch.cuda.max_memory_allocated()
        source=save_worker_reports(out,reports) if reports else {}
        if phase=='consume':source.pop('completed_goals',None)
        np.save(out/'consumed-records.npy',consumed);np.save(out/'loss.npy',losses.cpu().numpy())
        save_start=time.perf_counter()
        torch.save(dict(model=learner.model.state_dict(),optimizer=learner.optimizer.state_dict(),
            scheduler=learner.scheduler.state_dict(),optimizer_updates=c['updates'],config=c),out/'checkpoint.pt')
        checkpoint_seconds=time.perf_counter()-save_start
        result=dict(phase=phase,path=path,config=c,trace=str(trace_dir),setup_seconds=setup_seconds,
            wall_seconds=seconds,optimizer_updates=c['updates'],updates_per_second=c['updates']/seconds,
            agent_samples_per_second=c['updates']*c['envs']*c['agents']/seconds,
            h2d_bytes=consumer.bytes,goal_rows_refreshed_including_initial=consumer.refreshes,
            observed_goal_changes_in_records=int(np.count_nonzero(consumed[1:,:,7])),
            peak_gpu_allocated_bytes=peak_gpu,initial_common_eval_loss=initial_loss,
            final_checkpoint_seconds_outside_timer=checkpoint_seconds,
            first_training_loss=float(losses[0].item()),last_training_loss=float(losses[-1].item()),
            host_ranges=dict(source_wait=wait_seconds,upload_enqueue=upload_seconds,builder_model_update_dispatch=model_host_seconds),
            source=source,expert_in_main_timer=phase=='online',
            timing='continuous source release/input/complete CE updates/final source successor; boundary GPU sync',
            interpretation='source workers and GPU updates overlap; component durations are not additive wall time',
            policy='MAGAT original train mode, BN/Dropout enabled; full 2048-node batch at default config; CE/Adam/cosine100000',
            precision='float32; TF32 disabled',gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
        (out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False));(out/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ['phase','path','wall_seconds','updates_per_second','first_training_loss','last_training_loss']}),flush=True)
    finally:
        if pool:pool.close()


def compare(c,phase,trace,out,repetitions):
    out.mkdir(parents=True,exist_ok=True);config_file=out/'run-config.yaml'
    config_file.write_text(yaml.safe_dump(c,sort_keys=False));runs=[];pairs=[]
    for repeat in range(repetitions):
        paths=['pogema_original','jointloom'] if repeat%2==0 else ['jointloom','pogema_original'];current={}
        for path in paths:
            directory=out/f'repeat-{repeat}'/path
            subprocess.run([sys.executable,__file__,'child','--config',str(config_file.resolve()),
                '--phase',phase,'--path',path,'--trace',str(trace.resolve()),'--out',str(directory.resolve())],check=True)
            r=json.loads((directory/'result.json').read_text());current[path]=r;runs.append(r)
        a=np.load(out/f'repeat-{repeat}/pogema_original/consumed-records.npy',mmap_mode='r')
        b=np.load(out/f'repeat-{repeat}/jointloom/consumed-records.npy',mmap_mode='r')
        pairs.append(dict(repetition=repeat,speedup=current['pogema_original']['wall_seconds']/current['jointloom']['wall_seconds'],
            consumed_records_equal=bool(np.array_equal(a,b)),
            completed_goals_original=current['pogema_original']['source'].get('completed_goals'),
            completed_goals_jointloom=current['jointloom']['source'].get('completed_goals')))
    summary=dict(phase=phase,config=c,runs=runs,pairs=pairs,
        median_paired_speedup=float(np.median([r['speedup'] for r in pairs])),
        scope='timing repetitions with common model initialization, not independent learning seeds; no new quality claim')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(phase=phase,median_paired_speedup=summary['median_paired_speedup'],pairs=pairs)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['capture','compare','child'])
    p.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/e2_il.yaml')
    p.add_argument('--phase',choices=['consume','online'],default='consume')
    p.add_argument('--path',choices=['pogema_original','jointloom']);p.add_argument('--trace',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--repetitions',type=int,default=3)
    for name in ['updates','envs','workers','agents']:p.add_argument('--'+name,type=int)
    p.add_argument('--map-name')
    args=p.parse_args();c=yaml.safe_load(args.config.read_text())
    c['lacam_library']=os.environ.get('JOINTLOOM_LACAM_LIBRARY',str(bootstrap.ROOT/'src/real_expert_alg/lacam/liblacam.so'))
    for name in ['updates','envs','workers','agents','map_name']:
        if getattr(args,name) is not None:c[name]=getattr(args,name)
    if args.command=='capture':capture(c,args.out)
    elif args.command=='child':run_child(c,args.phase,args.path,args.trace,args.out)
    else:compare(c,args.phase,args.trace,args.out,args.repetitions)


if __name__=='__main__':main()
