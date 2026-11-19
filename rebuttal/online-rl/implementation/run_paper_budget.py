"""First full-budget pair, grounded in CostTracer Table1 (75M training steps)."""
import bootstrap
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from sf_tasks import SF_ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=False)
    impl=Path(__file__).resolve().parent;python=bootstrap.PYTHON
    shutil.copy2(impl.parent/'PAPER_BUDGET_PLAN.txt',root/'PAPER_BUDGET_PLAN.txt')
    state=dict(status='preparing',seed=0,agent_steps_reference=75000000,
        iterations=math.ceil(75000000/65536),bank_episodes=96,agents_per_env=128,num_envs=16,
        cpu_affinity=sorted(os.sched_getaffinity(0)),started=time.time(),stages=[])
    def run_stage(name,command):
        state['status']=name;entry=dict(name=name,status='running',started=time.time(),command=[str(v) for v in command])
        state['stages'].append(entry);(root/'state.json').write_text(json.dumps(state,indent=2))
        print('starting',name,flush=True)
        with (root/(name+'.log')).open('w') as log:
            code=subprocess.run(entry['command'],stdout=log,stderr=subprocess.STDOUT).returncode
        entry.update(status='complete' if code==0 else 'failed',finished=time.time(),exit_code=code)
        (root/'state.json').write_text(json.dumps(state,indent=2))
        if code:raise RuntimeError(name+' failed; inspect '+str(root/(name+'.log')))
    run_stage('training',[python,impl/'run_reward_trials.py','--root',root/'training',
        '--iterations',str(state['iterations']),'--seeds','0','--profiles','sf-async','jointloom',
        '--task-bank-episodes','96','--checkpoint-every','128',
        '--plan-file',impl.parent/'PAPER_BUDGET_PLAN.txt','--experiment-prefix',root.name])
    run_stage('evaluation',[python,impl/'evaluate_paper_budget.py','--root',root])
    run_stage('report',[python,impl/'paper_budget_report.py','--root',root])
    state.update(status='complete',finished=time.time());(root/'state.json').write_text(json.dumps(state,indent=2))
    print('paper-budget pair and fixed held-out evaluations complete',flush=True)


if __name__=='__main__':main()
