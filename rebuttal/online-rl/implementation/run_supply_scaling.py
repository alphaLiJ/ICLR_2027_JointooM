"""Fixed physical-core budget, real SF async, and proportional PPO workloads."""
import bootstrap
import argparse
import json
import shutil
import time
from pathlib import Path
from sf_tasks import SF_ROOT
from scaling_monitor import physical_core_cpus,run_monitored


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--envs',type=int,nargs='+',default=[16,32])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--iterations',type=int,default=16)
    args=p.parse_args();root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    python=bootstrap.PYTHON;impl=Path(__file__).resolve().parent
    cpus=physical_core_cpus(8);status=[]
    (root/'experiment.json').write_text(json.dumps(dict(envs=args.envs,seeds=args.seeds,
        iterations=args.iterations,agents=128,rollout=32,minibatch=2048,epochs=1,cpu_affinity=cpus,
        cpu_budget='8 distinct physical cores, one hardware thread each',
        SF='official2.1.1 native async, 8 workers, 2-way double buffering',
        samples_per_iteration={e:e*128*32 for e in args.envs},
        updates_per_iteration={e:e*128*32//2048 for e in args.envs}),indent=2))
    shutil.copytree(impl,root/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for seed in args.seeds:
        for ne in (args.envs if seed%2==0 else args.envs[::-1]):
            profiles=('sf-async','jointloom') if seed%2==0 else ('jointloom','sf-async')
            samples=ne*128*32
            for profile in profiles:
                name=f'{profile}-e{ne}-s{seed}';out=root/name
                entry=dict(run=name,status='running',started=time.time());status.append(entry)
                (root/'status.json').write_text(json.dumps(status,indent=2));print('starting',name,flush=True)
                if profile=='jointloom':
                    cmd=[python,impl/'train_pipeline.py','--profile','jointloom_optimized','--sf2-compatible',
                         '--num-envs',str(ne),'--seed',str(seed),'--iterations',str(args.iterations),'--output',out]
                else:
                    out.mkdir()
                    cmd=[python,impl/'sf_native.py','--algo=APPO','--env=jointloom_costtracer_pogema',
                        '--experiment='+root.name+'-'+name,'--train_dir='+str(SF_ROOT/'runs'),
                        '--seed='+str(seed),'--async_rl=True','--num_workers=8',
                        '--num_envs_per_worker='+str(ne//8),'--num_batches_per_epoch='+str(samples//2048),
                        '--train_for_env_steps='+str(args.iterations*samples-1),
                        '--costtracer_iterations='+str(args.iterations),
                        '--costtracer_initial_weights='+str(SF_ROOT/f'initial-seed-{seed}.pt'),
                        '--costtracer_observer_dir='+str(out)]
                code=run_monitored(cmd,root/(name+'.log'),out,cpus,args.iterations)
                entry.update(status='complete' if code==0 else 'failed',finished=time.time(),exit_code=code)
                (root/'status.json').write_text(json.dumps(status,indent=2))
                if code:raise RuntimeError('scaling run failed: '+name)
                print('complete',name,flush=True)
    print('scaling runs complete',flush=True)


if __name__=='__main__':main()
