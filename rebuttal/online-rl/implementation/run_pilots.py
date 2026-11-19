"""Bounded, sequential three-seed pilot; no automatic expansion of RL budget."""
import bootstrap
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    states=[]
    for seed in (0,1,2):
        # Alternate order, with no simultaneous GPU experiments.
        for backend in (('cuda','cpu') if seed%2==0 else ('cpu','cuda')):
            name=f'{backend}-s{seed}';out=root/name
            state=dict(run=name,status='running',started=time.time());states.append(state)
            (root/'status.json').write_text(json.dumps(states,indent=2))
            print('starting',name,flush=True)
            with (root/(name+'.log')).open('w') as log:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),
                    '--backend',backend,'--seed',str(seed),'--iterations','16','--output',str(out)],
                    stdout=log,stderr=subprocess.STDOUT)
            state.update(status='complete' if result.returncode==0 else 'failed',finished=time.time(),
                         exit_code=result.returncode)
            (root/'status.json').write_text(json.dumps(states,indent=2))
            if result.returncode:raise RuntimeError(f'{name} failed; see {root/(name+".log")}')
            print('complete',name,flush=True)
    print('bounded pilot finished',flush=True)


if __name__=='__main__':main()
