"""Wait at the current job boundary, run ready4 checks/pairs, resume the old queue."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import psutil


HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--resume-queue', type=int)
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    jobs = []
    def add(name, script, arguments, output):
        jobs.append(dict(name=name, status='pending', output=str(output),
            command=[sys.executable, str(HERE/script), *map(str, arguments), '--out', str(output)]))
    add('dynamic-input-and-fifo-check', 'check_ready.py', ['--source', args.source], args.root/'check')
    for path in ['cpu_cached_prefetch', 'jointloom']:
        add('pilot-'+path, 'run_live_ready_il.py', ['--path',path,'--seconds',30], args.root/'pilot'/path)
    for seed in [0,1,2]:
        paths = ['cpu_cached_prefetch','jointloom'] if seed%2==0 else ['jointloom','cpu_cached_prefetch']
        for path in paths:
            add(f'formal-{seed}-{path}', 'run_live_ready_il.py', ['--seed',seed,'--path',path,'--seconds',300],
                args.root/'formal'/f'seed-{seed}'/path)
    state = dict(status='waiting', started_at=now(), wait_pid=args.wait_pid,
        resume_queue=args.resume_queue, jobs=jobs, pairs=[])
    def save():
        (args.root/'execution-state.json').write_text(json.dumps(state, indent=2))
    save()
    try:
        if args.wait_pid:
            while psutil.pid_exists(args.wait_pid):
                try:
                    if psutil.Process(args.wait_pid).status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(2)
        state['status'] = 'running'
        save()
        for job in jobs:
            output = Path(job['output'])
            if (output/'result.json').exists():
                job['status'] = 'already_complete'
                save()
                continue
            job['status'], job['started_at'] = 'running', now()
            with (args.root/(job['name']+'.log')).open('a') as log:
                proc = subprocess.Popen(job['command'], cwd=HERE.parent, stdout=log, stderr=subprocess.STDOUT)
                job['pid'] = proc.pid
                save()
                code = proc.wait()
            job['exit_code'], job['ended_at'] = code, now()
            if code or not (output/'result.json').exists():
                job['status'] = 'failed'
                raise RuntimeError('ready4 job failed: '+job['name'])
            result = json.loads((output/'result.json').read_text())
            if job['name'].startswith('pilot-') and result['optimizer_updates'] == 0:
                job['status'] = 'failed'
                raise RuntimeError('pilot produced no actual optimizer update')
            job['status'] = 'complete'
            if job['name'].startswith('formal-'):
                pair_dir = output.parent
                pair = [pair_dir/path/'result.json' for path in ['cpu_cached_prefetch','jointloom']]
                if all(f.exists() for f in pair):
                    cpu, gpu = [json.loads(f.read_text()) for f in pair]
                    state['pairs'].append(dict(seed=cpu['seed'], cpu_updates=cpu['optimizer_updates'],
                        jointloom_updates=gpu['optimizer_updates'], cpu_rate=cpu['samples_per_second'],
                        jointloom_rate=gpu['samples_per_second'],
                        speedup=gpu['samples_per_second']/cpu['samples_per_second'] if cpu['samples_per_second'] else None))
            save()
            print(json.dumps(dict(job=job['name'], status='complete', time=now())), flush=True)
        state['status'], state['ended_at'] = 'complete', now()
        save()
    except BaseException:
        state['status'], state['error'] = 'failed', traceback.format_exc()
        save()
        raise
    finally:
        if args.resume_queue:
            try:
                owner = psutil.Process(args.resume_queue)
                if any(x.endswith('run_rebuttal_queue.py') for x in owner.cmdline()):
                    os.kill(args.resume_queue, signal.SIGCONT)
                    state['old_queue_resumed_at'] = now()
                    save()
            except (psutil.NoSuchProcess, psutil.ZombieProcess, ProcessLookupError):
                pass


if __name__ == '__main__':
    main()
