"""Same native SF/JointLoom configurations with opt-in neural timing only."""
import bootstrap
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from sf_tasks import SF_ROOT
from neural_timing import analyze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--envs',type=int,nargs='+',default=[16,32])
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--iterations',type=int,default=16)
    parser.add_argument('--full-model-stages',action='store_true')
    args=parser.parse_args()
    root=args.output_root
    root.mkdir(parents=True,exist_ok=False)
    impl=Path(__file__).resolve().parent
    shutil.copytree(impl,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    python=bootstrap.PYTHON
    metadata=dict(envs=args.envs,seeds=args.seeds,iterations=args.iterations,
        cpu_affinity=sorted(os.sched_getaffinity(0)),agents=128,rollout=32,minibatch=2048,
        epochs=1,SF='official native asynchronous sampler/learner, 8 workers, 2 splits',
        excluded='neural forward + entire loss.backward interval UNION on shared device clock',
        retained='environment, observations, device transfers, GAE, shuffling/gather, loss forward, gradient clipping, Adam, scheduling')
    if args.full_model_stages:
        metadata.update(excluded='complete model evaluation/action sampling, loss/backward/clipping/Adam and model invocation/update-loop control; union of intervals',
            retained='environment, observations, device transfers, GAE, trajectory buffers, shuffle/gather, data handoff/queue waiting',
            stage_boundaries='learner._train excluding _get_minibatches/_get_minibatch; inference forward/update_model; JointLoom learner.update excluding shuffle/gather; policy and action selection excluding input copy')
    (root/'experiment.json').write_text(json.dumps(metadata,indent=2))
    results=[]
    for seed in args.seeds:
        for ne in args.envs:
            samples=ne*128*32
            order=('sf-async','jointloom') if seed%2==0 else ('jointloom','sf-async')
            for profile in order:
                name=f'{profile}-e{ne}-s{seed}'
                out=root/name
                env=dict(os.environ,JOINTLOOM_NEURAL_PROFILE_DIR=str(out),WANDB_MODE='disabled',
                    TRITON_CACHE_DIR=str(bootstrap.ARTIFACTS/'pipeline-v2-cache'/'triton'))
                if args.full_model_stages:
                    env['JOINTLOOM_MODEL_STAGE_PROFILE']='1'
                if profile=='jointloom':
                    command=[python,impl/'train_pipeline.py','--profile','jointloom_optimized',
                        '--sf2-compatible','--num-envs',str(ne),'--seed',str(seed),
                        '--iterations',str(args.iterations),'--output',out]
                else:
                    out.mkdir()
                    command=[python,impl/'sf_native.py','--algo=APPO','--env=jointloom_costtracer_pogema',
                        '--experiment='+root.name+'-'+name,'--train_dir='+str(SF_ROOT/'runs'),
                        '--seed='+str(seed),'--async_rl=True','--num_workers=8',
                        '--num_envs_per_worker='+str(ne//8),
                        '--num_batches_per_epoch='+str(samples//2048),
                        '--train_for_env_steps='+str(args.iterations*samples-1),
                        '--costtracer_iterations='+str(args.iterations),
                        '--costtracer_initial_weights='+str(SF_ROOT/f'initial-seed-{seed}.pt'),
                        '--costtracer_observer_dir='+str(out)]
                print('starting',name,flush=True)
                with (root/(name+'.log')).open('w') as log:
                    run=subprocess.run([str(c) for c in command],env=env,stdout=log,stderr=subprocess.STDOUT)
                if run.returncode:
                    raise RuntimeError(f'{name} failed: '+str(root/(name+'.log')))
                result=analyze(out,native=profile=='sf-async')
                entry=dict(profile=profile,seed=seed,num_envs=ne,**result['mean'],
                    calibration_brackets_ns=result['min_calibration_brackets_ns'])
                results.append(entry)
                (root/'results.json').write_text(json.dumps(results,indent=2))
                print(json.dumps(entry),flush=True)
    summary={}
    for ne in args.envs:
        summary[ne]={}
        for profile in ('sf-async','jointloom'):
            rows=[r for r in results if r['profile']==profile and r['num_envs']==ne]
            summary[ne][profile]={k:sum(r[k] for r in rows)/len(rows) for k in
                ('wall_seconds','neural_union_seconds','forward_union_seconds',
                 'backward_union_seconds','residual_seconds')}
            if args.full_model_stages:
                summary[ne][profile].update({k:sum(r[k] for r in rows)/len(rows) for k in
                    ('model_stage_union_seconds','system_residual_seconds','shuffle_gather_hole_seconds')})
        summary[ne]['residual_speedup']=summary[ne]['sf-async']['residual_seconds']/summary[ne]['jointloom']['residual_seconds']
        summary[ne]['instrumented_wall_speedup']=summary[ne]['sf-async']['wall_seconds']/summary[ne]['jointloom']['wall_seconds']
        if args.full_model_stages:
            summary[ne]['model_stage_residual_speedup']=summary[ne]['sf-async']['system_residual_seconds']/summary[ne]['jointloom']['system_residual_seconds']
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
