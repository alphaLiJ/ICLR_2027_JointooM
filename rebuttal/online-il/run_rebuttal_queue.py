"""Serial execution of the authorized aligned and order experiments."""
import bootstrap
from repository_paths import ARTIFACTS, input_path
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil


HERE=Path(__file__).resolve().parent
ROOT=input_path('JOINTLOOM_ONLINE_ARTIFACTS', ARTIFACTS/'online-il')
LEGACY=input_path('JOINTLOOM_LEGACY_ARTIFACTS', ARTIFACTS/'legacy')/'jointloom-training-formal-20260920-v2/inputs'
ORDERS=['ordered','window_shuffle_128','global_shuffle']


def now():return datetime.now(timezone.utc).isoformat()


def jobs(root):
    rows=[]
    def add(name,script,arguments,out,marker='result.json'):
        rows.append(dict(name=name,command=[sys.executable,str(HERE/script),*map(str,arguments),'--out',str(out)],
            output=str(out),marker=marker,status='pending'))
    for path in ['cpu_cached_prefetch','jointloom']:
        add('paper-ring-pilot-'+path,'run_il_order.py',['run','--corpus',LEGACY/'magat-s42','--updates',16,
            '--seed',42,'--order','global_shuffle','--path',path,'--transport','paper_ring'],root/'order/pilot-paper-ring'/path)
    add('validation-and-learning-pilot','run_il_order.py',['run','--corpus',LEGACY/'magat-s42','--updates',16,
        '--seed',42,'--phase','learn','--transport','paper_ring'],root/'order/pilot-validation')
    add('original-expert-source-pilot','run_il_order.py',['capture','--updates',16,'--seed',0],root/'order/pilot-expert-source','summary.json')
    for mode,order in [('standard_mapf','ordered'),('lifelong','ordered'),('standard_mapf','window_shuffle_128')]:
        for path in ['cpu_cached_prefetch','jointloom']:
            add('live-pilot-'+mode+'-'+order+'-'+path,'run_live_il.py',['run','--mode',mode,'--order',order,
                '--path',path,'--seconds',12],root/'live-pilot'/mode/order/path)
    # Paper fixed-work shapes/host layout, archived exact per-pair records.
    for path in ['cpu_cached_prefetch','jointloom']:
        add('deadline-pilot-lifelong-'+path,'run_live_il.py',['run','--mode','lifelong','--path',path,
            '--seconds',30],root/'live-deadline-pilot'/path)
    for seed in [42,43,44]:
        for path in ['cpu_cached_prefetch','jointloom']:
            add(f'L2-fixed-{seed}-{path}','run_il_order.py',['run','--corpus',LEGACY/f'magat-s{seed}',
                '--updates',512,'--seed',seed,'--path',path],root/'l2-fixed'/f'seed-{seed}'/path)
    add('old-checkpoints-native256','run_e4.py',['evaluate'],root/'old-checkpoints-native256','summary.json')
    # Same E8 ordered controls are reused by the window-shuffle comparison.
    for seed in [0,1,2]:
        for mode,order in [('standard_mapf','ordered'),('lifelong','ordered'),('standard_mapf','window_shuffle_128')]:
            paths=['cpu_cached_prefetch','jointloom'] if seed%2==0 else ['jointloom','cpu_cached_prefetch']
            for path in paths:
                add(f'live-{mode}-{order}-{seed}-{path}','run_live_il.py',['run','--seed',seed,
                    '--mode',mode,'--order',order,'--path',path,'--seconds',300],root/'live-deadline'/mode/order/f'seed-{seed}'/path)
    for seed in [42,43,44]:
        corpus=root/'l2-lifelong-inputs'/f'seed-{seed}'
        add(f'L2-lifelong-capture-{seed}','run_live_il.py',['capture','--mode','lifelong','--envs',4,
            '--updates',512,'--seed',seed],corpus,'summary.json')
        for path in ['cpu_cached_prefetch','jointloom']:
            add(f'L2-lifelong-{seed}-{path}','run_il_order.py',['run','--corpus',corpus,'--updates',512,
                '--seed',seed,'--path',path],root/'l2-lifelong'/f'seed-{seed}'/path)
    for seed in [0,1,2]:
        corpus=root/'order/corpus'/f'seed-{seed}'
        add(f'order-corpus-{seed}','run_il_order.py',['capture','--updates',100000,'--seed',seed],corpus,'summary.json')
        # Prioritize the reviewer-requested learning controls once each corpus exists.
        order_sequence=ORDERS[seed:]+ORDERS[:seed]
        for order in order_sequence:
            add(f'order-learn-{seed}-{order}','run_il_order.py',['run','--corpus',corpus,'--updates',100000,
                '--seed',seed,'--order',order,'--phase','learn'],root/'order/learning'/f'seed-{seed}'/order)
        for order in ORDERS:
            paths=['cpu_cached_prefetch','jointloom'] if seed%2==0 else ['jointloom','cpu_cached_prefetch']
            for path in paths:
                add(f'order-replay-{seed}-{order}-{path}','run_il_order.py',['run','--corpus',corpus,
                    '--updates',4096,'--seed',seed,'--order',order,'--path',path],root/'order/replay'/f'seed-{seed}'/order/path)
    add('order-finite-task-quality','evaluate_order.py',['--models',root/'order/learning'],root/'order/task-quality','summary.json')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--wait-pid',type=int);p.add_argument('--max-jobs',type=int)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    logs=root/'queue-logs';logs.mkdir(exist_ok=True)
    snapshot=root/'source-at-queue';snapshot.mkdir(exist_ok=True)
    for source in HERE.rglob('*.py'):
        if 'native' in source.relative_to(HERE).parts:continue
        target=snapshot/source.relative_to(HERE);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
    state=dict(status='waiting' if args.wait_pid else 'running',started_at=now(),jobs=jobs(root),wait_pid=args.wait_pid)
    for row in state['jobs']:
        if (Path(row['output'])/row['marker']).exists():row['status']='already_complete'
    def save():(root/'execution-state.json').write_text(json.dumps(state,indent=2))
    save()
    if args.wait_pid:
        while psutil.pid_exists(args.wait_pid):
            try:
                if psutil.Process(args.wait_pid).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(5)
        state['status']='running';save()
    completed=0
    for row in state['jobs']:
        if (Path(row['output'])/row['marker']).exists():
            row['status']='already_complete';save();continue
        if args.max_jobs is not None and completed>=args.max_jobs:
            state['status']='bounded_batch_finished';save();return
        row['status']='running';row['started_at']=now();save()
        with (logs/(row['name']+'.log')).open('a',buffering=1) as log:
            process=subprocess.Popen(row['command'],cwd=HERE.parent,stdout=log,stderr=subprocess.STDOUT)
            row['pid']=process.pid;save();code=process.wait()
        row['exit_code']=code;row['ended_at']=now()
        if code or not (Path(row['output'])/row['marker']).exists():
            row['status']='failed';state['status']='failed';state['failed_job']=row['name'];save()
            print(json.dumps(dict(status='failed',job=row['name'],log=str(logs/(row['name']+'.log')))),flush=True);return
        row['status']='complete';completed+=1;save()
        print(json.dumps(dict(status='complete',job=row['name'],time=now())),flush=True)
    state['status']='complete';state['ended_at']=now();save()


if __name__=='__main__':main()
