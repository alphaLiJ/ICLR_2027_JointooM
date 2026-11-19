"""Narrow resumed work: six-model quality, static random-ready16, third replay seed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
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
    p.add_argument('--previous-root', type=Path, required=True)
    p.add_argument('--quality-pid', type=int, required=True)
    args = p.parse_args()
    root, previous = args.root, args.previous_root
    root.mkdir(parents=True, exist_ok=True)
    quality = root/'quality-six-models'
    state = dict(status='waiting_quality', started_at=now(), quality_pid=args.quality_pid,
        scope='six existing models; source-indexed random-ready16 fixed-goal IL; third fixed-record replay seed; old queues stay stopped',
        jobs=[], online_pairs=[], shuffle_pairs=[])
    def save():
        (root/'execution-state.json').write_text(json.dumps(state, indent=2))
    def job(name, script, arguments, output, marker='result.json'):
        item = dict(name=name, status='pending', output=str(output), marker=marker,
            command=[sys.executable, str(HERE/script), *map(str, arguments), '--out', str(output)])
        state['jobs'].append(item)
        return item
    quality_job = dict(name='six-existing-models-quality', status='running', pid=args.quality_pid,
        output=str(quality), marker='summary.json')
    state['jobs'].append(quality_job)
    check = job('random-ready16-input-check', 'check_ready.py',
        ['--source',previous/'l2-lifelong-inputs/seed-42','--producer-envs',16], root/'random-ready16/check')
    pilots = []
    common = ['--producer-envs',16,'--mode','standard_mapf','--selection','random_ready']
    for path in ['cpu_cached_prefetch','jointloom']:
        pilots.append(job('random-ready16-pilot-'+path, 'run_live_ready_il.py',
            [*common,'--path',path,'--seconds',60], root/'random-ready16/pilot'/path))
    formals = []
    for seed in [0,1,2]:
        paths = ['cpu_cached_prefetch','jointloom'] if seed%2==0 else ['jointloom','cpu_cached_prefetch']
        for path in paths:
            item = job(f'random-ready16-formal-{seed}-{path}', 'run_live_ready_il.py',
                [*common,'--seed',seed,'--path',path,'--seconds',300],
                root/'random-ready16/formal'/f'seed-{seed}'/path)
            item['status'] = 'conditional_on_input_supply'
            formals.append(item)
    corpus = root/'shuffle-replay/inputs-seed-2-4096'
    capture = job('replay-seed2-capture4096', 'run_il_order.py',
        ['capture','--updates',4096,'--seed',2], corpus, 'summary.json')
    replays = []
    for order in ['ordered','window_shuffle_128','global_shuffle']:
        for path in ['cpu_cached_prefetch','jointloom']:
            replays.append(job(f'replay-seed2-{order}-{path}', 'run_il_order.py',
                ['run','--corpus',corpus,'--updates',4096,'--seed',2,'--order',order,
                 '--path',path,'--workers',2,'--depth',4],root/'shuffle-replay/seed-2'/order/path))
    save()

    def execute(item):
        output = Path(item['output'])
        marker = output/item['marker']
        if marker.exists():
            item['status'] = 'already_complete'
            save()
            return json.loads(marker.read_text())
        item['status'], item['started_at'] = 'running', now()
        with (root/(item['name']+'.log')).open('a') as log:
            proc = subprocess.Popen(item['command'], cwd=HERE.parent, stdout=log, stderr=subprocess.STDOUT)
            item['pid'] = proc.pid
            save()
            code = proc.wait()
        item['exit_code'], item['ended_at'] = code, now()
        if code or not marker.exists():
            item['status'] = 'failed'
            save()
            raise RuntimeError('focused job failed: '+item['name'])
        item['status'] = 'complete'
        save()
        print(json.dumps(dict(job=item['name'], status='complete', time=now())), flush=True)
        return json.loads(marker.read_text())

    try:
        while psutil.pid_exists(args.quality_pid):
            try:
                if psutil.Process(args.quality_pid).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(2)
        if not (quality/'summary.json').exists():
            quality_job['status'] = 'failed'
            raise RuntimeError('six-model evaluation did not finish; see quality-six-models.log')
        quality_job['status'], quality_job['ended_at'] = 'complete', now()
        state['status'] = 'running'
        save()
        execute(check)
        pilot_results = [execute(item) for item in pilots]
        evidence = []
        for result in pilot_results:
            source = result['source']
            evidence.append(dict(path=result['path'], updates=result['optimizer_updates'],
                output_ready_batches=source['output_ready_batches'], output_full_polls=source['output_full_polls'],
                ready_set_shortage_polls=source['ready_set_shortage_polls'],
                samples_per_second=result['samples_per_second']))
        # This chooses the intended workload, not a positive speedup: formal
        # runs require a ready output backlog on both paths, regardless of ratio.
        supplied = all(e['updates']>0 and (e['output_ready_batches']>=4 or e['output_full_polls']>0) for e in evidence)
        state['online_supply_assessment'] = dict(evidence=evidence, run_formal=supplied,
            criterion='both paths have at least four ready output batches or observed output-ring backpressure; no speedup threshold',
            scope='short-run supply diagnostic; pilot timings retained regardless of result')
        save()
        if supplied:
            for item in formals:
                result = execute(item)
                pair_dir = Path(item['output']).parent
                pair = [pair_dir/path/'result.json' for path in ['cpu_cached_prefetch','jointloom']]
                if all(p.exists() for p in pair):
                    cpu, gpu = [json.loads(p.read_text()) for p in pair]
                    seed = cpu['seed']
                    state['online_pairs'] = [x for x in state['online_pairs'] if x['seed']!=seed]
                    state['online_pairs'].append(dict(seed=seed,cpu_rate=cpu['samples_per_second'],
                        jointloom_rate=gpu['samples_per_second'],speedup=gpu['samples_per_second']/cpu['samples_per_second']))
                    save()
        else:
            for item in formals:
                item['status'] = 'not_scheduled_source_limited'
            save()
        execute(capture)
        for item in replays:
            execute(item)
        for seed in [0,1,2]:
            base = root/'shuffle-replay'/f'seed-{seed}' if seed==2 else previous/'order/replay'/f'seed-{seed}'
            for order in ['ordered','window_shuffle_128','global_shuffle']:
                cpu, gpu = [json.loads((base/order/path/'result.json').read_text()) for path in ['cpu_cached_prefetch','jointloom']]
                state['shuffle_pairs'].append(dict(seed=seed,order=order,cpu_seconds=cpu['wall_seconds'],
                    jointloom_seconds=gpu['wall_seconds'],speedup=cpu['wall_seconds']/gpu['wall_seconds']))
        state['shuffle_median_speedups'] = {order:statistics.median(x['speedup'] for x in state['shuffle_pairs'] if x['order']==order)
            for order in ['ordered','window_shuffle_128','global_shuffle']}
        if len(state['online_pairs'])==3:
            state['online_median_speedup'] = statistics.median(x['speedup'] for x in state['online_pairs'])
        state['status'], state['ended_at'] = 'complete', now()
        save()
    except BaseException:
        state['status'], state['error'] = 'failed', traceback.format_exc()
        save()
        raise


if __name__ == '__main__':
    main()
