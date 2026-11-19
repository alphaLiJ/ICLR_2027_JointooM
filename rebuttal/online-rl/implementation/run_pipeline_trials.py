"""Bounded six runs with fixed selected execution profiles; old evidence stays intact."""
import bootstrap
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(bootstrap.WORK/'implementation',root/'source/implementation',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(bootstrap.WORK/'configs',root/'source/configs')
    shutil.copy2(bootstrap.WORK/'PIPELINE_OPTIMIZATION_PLAN.txt',root/'source/PIPELINE_OPTIMIZATION_PLAN.txt')
    states=[]
    for seed in (0,1,2):
        order=('jointloom_optimized','reference_cpu_buffers') if seed%2==0 else ('reference_cpu_buffers','jointloom_optimized')
        for profile in order:
            name=f'{profile}-s{seed}';entry=dict(run=name,status='running',started=time.time());states.append(entry)
            (root/'status.json').write_text(json.dumps(states,indent=2));print('starting',name,flush=True)
            command=[sys.executable,str(Path(__file__).with_name('train_pipeline.py')),
                '--profile',profile,'--seed',str(seed),'--iterations','16','--output',str(root/name)]
            if profile=='jointloom_optimized':command+=['--nchw','--compile-model','--compile-loss']
            with (root/(name+'.log')).open('w') as log:
                r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            entry.update(status='complete' if r.returncode==0 else 'failed',finished=time.time(),exit_code=r.returncode)
            (root/'status.json').write_text(json.dumps(states,indent=2))
            if r.returncode:raise RuntimeError('run failed: '+name)
            print('complete',name,flush=True)
    print('six trials complete',flush=True)


if __name__=='__main__':main()
