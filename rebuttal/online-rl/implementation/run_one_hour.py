"""Durable sequential, equal-wall-time native SF / JointLoom training pair."""
import bootstrap
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import resource

from sf_tasks import SF_ROOT, prepare


def atomic_json(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def last_row(path):
    if not path.exists():return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16384));lines=f.read().splitlines()
    for line in reversed(lines):
        try:return json.loads(line)
        except (ValueError,UnicodeDecodeError):continue
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--desktop',type=Path,required=True);p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--bank-episodes',type=int,default=2048);p.add_argument('--probe-seconds',type=float,default=20)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=False)
    impl=Path(__file__).resolve().parent;python=bootstrap.PYTHON
    training=root/'training';training.mkdir();args.desktop.mkdir(parents=True,exist_ok=True)
    shutil.copytree(impl,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for dest in (root,args.desktop):shutil.copy2(impl.parent/'ONE_HOUR_PLAN.txt',dest/'EXPERIMENT_PLAN.txt')
    state=dict(status='preparing_tasks',started=time.time(),pid=os.getpid(),seconds_per_profile=args.seconds,
        bank_episodes=args.bank_episodes,seed=0,num_envs=16,agents_per_env=128,
        cpu_affinity=sorted(os.sched_getaffinity(0)),root=str(root),desktop=str(args.desktop),jobs=[])
    env=dict(os.environ,WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    for key in ('JOINTLOOM_NEURAL_PROFILE_DIR','JOINTLOOM_MODEL_STAGE_PROFILE'):env.pop(key,None)
    stopfile=root/'STOP_REQUESTED'

    def status():
        state['updated']=time.time();atomic_json(root/'state.json',state)
        atomic_json(training/'status.json',state['jobs'])
        atomic_json(args.desktop/'training-status.json',state)
        lines=['CostTracer: one-hour CPU/SF and JointLoom training','Status: '+state['status'],
            '16env x128agents; shared seed0 initial weights; sequential runs on the same GPU.','Output directory: '+str(root)]
        if state.get('task_preparation'):
            lines.append('Task-bank preparation: '+str(round(state['task_preparation']['fraction']*100,1))+'%')
        for job in state['jobs']:
            duration=job.get('finished',time.time())-job['started']
            lines.append(f"{job['run']}: {job['status']}, elapsed {duration/60:.1f} minutes, budget {job['budget_seconds']/60:.1f} minutes; latest fully updated samples {job.get('live_processed_samples',0):,}")
        (args.desktop/'training-progress.txt').write_text('\n'.join(lines)+'\n')

    def execute(profile,phase,seconds):
        if stopfile.exists():raise InterruptedError('STOP_REQUESTED')
        name=f'{phase}-{profile}-s0';out=training/name
        begin_ns=time.perf_counter_ns();deadline=begin_ns+int(seconds*1e9)
        if profile=='jointloom':
            cmd=[python,impl/'train_pipeline.py','--profile','jointloom_optimized','--sf2-compatible',
                '--num-envs','16','--seed','0','--iterations','1000000','--minibatch-order','sf_env_major',
                '--output',out,'--task-bank-episodes',str(args.bank_episodes),
                '--checkpoint-seconds','300','--stop-perf-ns',str(deadline),'--stop-file',stopfile]
        else:
            out.mkdir()
            cmd=[python,impl/'sf_native.py','--algo=APPO','--env=jointloom_costtracer_pogema',
                '--experiment='+root.name+'-'+name,'--train_dir='+str(SF_ROOT/'runs'),
                '--seed=0','--async_rl=True','--num_workers=8','--num_envs_per_worker=2',
                '--shuffle_minibatches=False','--costtracer_reward_log','--costtracer_progress_log',
                '--costtracer_iterations=1000000','--train_for_env_steps=1000000000000000',
                '--train_for_seconds=7200','--save_every_sec=300','--keep_checkpoints=24',
                '--costtracer_initial_weights='+str(SF_ROOT/'initial-seed-0.pt'),
                '--costtracer_observer_dir='+str(out),'--costtracer_task_bank_episodes='+str(args.bank_episodes),
                '--costtracer_stop_perf_ns='+str(deadline),'--costtracer_stop_file='+str(stopfile)]
        job=dict(run=name,profile=profile,phase=phase,seed=0,status='running',started=time.time(),
            start_perf_counter_ns=begin_ns,deadline_perf_counter_ns=deadline,budget_seconds=seconds,
            command=[str(x) for x in cmd])
        state['jobs'].append(job);state['status']='training_'+name;status()
        print('starting',name,flush=True)
        logpath=training/(name+'.log')
        with logpath.open('w') as log:
            child=subprocess.Popen(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env)
            job['pid']=child.pid;status();last_update=0
            while child.poll() is None:
                if time.monotonic()-last_update>=30:
                    row=last_row(out/('iterations.jsonl' if profile=='jointloom' else 'native-iterations.jsonl'))
                    if row:job.update(live_processed_samples=row['agent_samples'],latest_training_row=row)
                    status();last_update=time.monotonic()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        job.update(status='complete' if child.returncode==0 else 'failed',finished=time.time(),exit_code=child.returncode)
        status()
        if child.returncode:raise RuntimeError('training failed: '+str(logpath))
        import torch
        if profile=='jointloom':
            checkpoint_path=out/'final.pt';rows=[json.loads(s) for s in (out/'iterations.jsonl').read_text().splitlines()]
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            processed=int(checkpoint['agent_samples']);events=[r for r in rows if r['end_perf_counter_ns']<=deadline]
            observed=rows[-1]['env_joint_steps']
            if not rows:raise RuntimeError('missing GPU training records')
            job['peak_gpu_memory_bytes']=rows[-1]['peak_gpu_memory_bytes']
        else:
            checkpoints=SF_ROOT/'runs'/(root.name+'-'+name)/'checkpoint_p0'
            checkpoint_path=max(checkpoints.glob('checkpoint_*.pth'),key=lambda x:x.stat().st_mtime)
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            processed=int(checkpoint['env_steps']);rows=[json.loads(s) for s in (out/'native-iterations.jsonl').read_text().splitlines()]
            events=[r for r in rows if r['timestamp_perf_counter']*1e9<=deadline]
            rewards=[json.loads(x.read_text()) for x in out.glob('reward-env-*-pid-*.json')]
            active=[r for r in rewards if r['records']]
            if {r['env_id'] for r in active}!=set(range(16)) or any(r['agents']!=128 for r in active):
                raise RuntimeError('missing native SF reward records or incorrect agent count')
            observed=max(r['total_joint_steps'] for r in active)
        steps=sorted(set(float(v['step']) for v in checkpoint['optimizer']['state'].values()))
        if steps!=[processed/2048] or not all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()):
            raise RuntimeError('invalid saved model or optimizer counters')
        within=int(events[-1]['agent_samples']) if events else 0
        job.update(checkpoint=str(checkpoint_path),checkpoint_processed_samples=processed,
            checkpoint_adam_steps=steps,processed_samples_within_deadline=within,
            processed_samples_after_deadline=processed-within,live_processed_samples=processed,
            max_env_joint_steps=observed,task_bank_cycles=observed//(args.bank_episodes*512),
            full_job_seconds=job['finished']-job['started'],completed_iterations=len(rows))
        if observed>=args.bank_episodes*512:raise RuntimeError('task bank exhausted')
        if phase=='probe' and len(rows)<2:raise RuntimeError('probe too short to exercise timed stopping')
        status();print('complete',name,processed,'samples',flush=True)
        del checkpoint
        if stopfile.exists():raise InterruptedError('STOP_REQUESTED')

    try:
        def preparation_progress(record):
            state['task_preparation']=record
            state['preparation_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            status()
        status();prepare(0,num_envs=16,bank_episodes=args.bank_episodes,progress=preparation_progress)
        state['status']='checking_parameters';status()
        # Run the already-established matching check in a separate process so it
        # releases its CUDA context before either timed training job starts.
        audit_code='from pathlib import Path;from run_reward_trials import audit;audit(Path('+repr(str(root))+'),bank_episodes='+str(args.bank_episodes)+',seeds=(0,))'
        with (root/'parameter-check.log').open('w') as log:
            subprocess.run([str(python),'-c',audit_code],cwd=impl,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for profile in ('sf-async','jointloom'):execute(profile,'probe',args.probe_seconds)
        for profile in ('sf-async','jointloom'):execute(profile,'formal',args.seconds)
        for stage,script in (('evaluation','evaluate_paper_budget.py'),('report','one_hour_report.py')):
            state['status']=stage;status()
            command=[str(python),str(impl/script),'--root',str(root)]
            if stage=='report':command+=['--desktop',str(args.desktop)]
            with (root/(stage+'.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        state.update(status='complete',finished=time.time());status();print('complete: both one-hour runs, evaluation and desktop plots',flush=True)
    except BaseException as error:
        state.update(status='stopped_by_user' if isinstance(error,InterruptedError) else 'failed',
            error=repr(error),finished=time.time());status();raise


if __name__=='__main__':main()
