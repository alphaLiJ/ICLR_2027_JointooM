"""Actual SF synchronous/default-asynchronous runs and matched resident runs."""
import bootstrap
import json
import os
import subprocess
import time
from pathlib import Path
from sf_tasks import SF_ROOT


def main():
    root=SF_ROOT/'paired-v1';root.mkdir(parents=True,exist_ok=False)
    python=bootstrap.PYTHON
    implementation=Path(__file__).resolve().parent
    status=[]
    env=dict(os.environ,WANDB_MODE='disabled',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    for seed in (0,1,2):
        profiles=('sf-sync','jointloom-sf2','sf-async') if seed%2==0 else ('sf-async','jointloom-sf2','sf-sync')
        for profile in profiles:
            name=f'{profile}-s{seed}';out=root/name
            entry=dict(run=name,status='running',started=time.time());status.append(entry)
            (root/'status.json').write_text(json.dumps(status,indent=2));print('starting',name,flush=True)
            if profile=='jointloom-sf2':
                cmd=[str(python),str(implementation/'train_pipeline.py'),
                    '--profile','jointloom_optimized','--sf2-compatible','--seed',str(seed),
                    '--iterations','16','--output',str(out)]
            else:
                out.mkdir()
                cmd=[str(python),str(implementation/'sf_native.py'),'--algo=APPO',
                    '--env=jointloom_costtracer_pogema',f'--experiment={name}',
                    '--train_dir='+str(SF_ROOT/'runs'),f'--seed={seed}',
                    '--async_rl='+str(profile=='sf-async'),'--costtracer_iterations=16',
                    '--train_for_env_steps=1048575','--costtracer_initial_weights='+str(SF_ROOT/f'initial-seed-{seed}.pt'),
                    '--costtracer_observer_dir='+str(out)]
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            entry.update(status='complete' if result.returncode==0 else 'failed',finished=time.time(),exit_code=result.returncode)
            (root/'status.json').write_text(json.dumps(status,indent=2))
            if result.returncode:raise RuntimeError('run failed; inspect '+str(root/(name+'.log')))
            print('complete',name,flush=True)
    print('nine native/matched runs complete',flush=True)


if __name__=='__main__':main()
