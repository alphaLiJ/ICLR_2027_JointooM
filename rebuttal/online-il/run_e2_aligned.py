"""Original A256 tasks: fixed-goal/lifelong first and sustained decisions."""
import bootstrap
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from aligned_tasks import load_paper_tasks, AlignedResident, PAPER_INPUT, PAPER_CHECKPOINT
from envs import CpuPool
from inputs import HostTransfer
from native_extent_inputs import NativeExtentBatchBuilder
from policies import MagatPolicy
from run_e2 import cpu_batched_window


def configure():
    os.sched_setaffinity(0, set(range(16)))
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def gpu_window(env, policy, steps):
    for _ in range(steps):
        env.sim.update_derived_state()
        data = policy.builder.build_from_stateful(env.sim, materialize_edges=False)
        actions = policy.act(data).reshape(env.tasks.positions.shape[:2])
        env.sim.update_actions(actions)
        env.sim.step_sim_only()


def child(config, path, mode, out):
    if (out/'result.json').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    configure()
    setup = time.perf_counter()
    tasks = load_paper_tasks(config['envs'])
    np.savez_compressed(out/'initial-tasks.npz', positions=tasks.positions, goals=tasks.goals,
        initial_arrived=tasks.initial_arrived)
    policy = MagatPolicy(str(PAPER_CHECKPOINT), 'cuda:0')
    policy.builder = NativeExtentBatchBuilder([g.shape for g in tasks.grids], 256, 'cuda:0')
    # Warm model and exact batch shape, then invalidate every target-task cache.
    env = AlignedResident(tasks, mode)
    gpu_window(env, policy, 2)
    torch.cuda.synchronize()
    pool = None
    try:
        if path == 'jointloom':
            env.reset()
            window = lambda steps: gpu_window(env, policy, steps)
            snapshot = env.result
        else:
            del env
            gc.collect()
            pool = CpuPool(tasks, 8, list(range(16)), backend='original', mode=mode, prepare=False)
            transfer = HostTransfer(config['envs'], 256, 'cuda:0')
            window = lambda steps: cpu_batched_window(pool, transfer, policy, steps, 256)
            def snapshot():
                for c in pool.conns:
                    c.send(('result', None))
                reports = [pool.receive(c) for c in pool.conns]
                state = {k:np.concatenate([r[k] for r in reports]) for k in
                    ('positions', 'goals', 'cursors', 'arrived')}
                return state, reports
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - setup
        torch.cuda.reset_peak_memory_stats()
        metrics = {}
        for phase, steps in [('first',1), ('after_one',1), ('sustained',config['steps'])]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            ranges = window(steps)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            metrics[phase] = dict(steps=steps, wall_seconds=seconds,
                batch_decision_ms=seconds*1000/steps,
                nominal_joint_transitions_per_second=steps*config['envs']/seconds)
            if isinstance(ranges, dict):
                metrics[phase]['diagnostics'] = ranges
            print(json.dumps(dict(path=path, mode=mode, envs=config['envs'], phase=phase, seconds=seconds)), flush=True)
        if pool:
            state, reports = snapshot()
            payload = transfer.bytes
            diagnostics = dict(worker_seconds=[dict(build=r['build_seconds'], transition=r['step_seconds'],
                refreshed_goals=r['refreshed_goals']) for r in reports],
                note='worker durations overlap; not an additive wall-clock decomposition')
        else:
            state = snapshot()
            payload, diagnostics = 0, {}
        initial_done = int(tasks.initial_arrived.sum())
        final_done = int(state['arrived'].sum())
        result = dict(config=config, path=path, mode=mode, source=str(PAPER_INPUT),
            checkpoint=str(PAPER_CHECKPOINT), map_shapes=list(tasks.grids.shape),
            initial_task_protocol=tasks.source_stats,
            initial_arrived_agents=initial_done, final_arrived_agents=final_done,
            final_active_agent_fraction=1. if mode=='lifelong' else 1-final_done/state['arrived'].size,
            completed_goals=int(state['cursors'].sum()), metrics=metrics, setup_seconds=setup_seconds,
            h2d_bytes=payload, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            diagnostics=diagnostics, precision='float32; TF32 disabled',
            cpu_affinity=list(range(16)), workers=8, policy_env_batch=config['envs'],
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            timer='first cache inside first; second decision after one; sustained after two decisions; boundary sync',
            scope='same original tasks and policy; first/after-one separate from sustained; no task-quality claim')
        np.savez_compressed(out/'final-state.npz', **state)
        (out/'result.json').write_text(json.dumps(result, indent=2))
    finally:
        if pool:
            pool.close()


def compare(config, out, repetitions, modes):
    out.mkdir(parents=True, exist_ok=True)
    runs, pairs = [], []
    for mode in modes:
        for repeat in range(repetitions):
            paths = ['pogema_original','jointloom'] if repeat%2==0 else ['jointloom','pogema_original']
            current = {}
            for path in paths:
                directory = out/mode/f'repeat-{repeat}'/path
                if not (directory/'result.json').exists():
                    subprocess.run([sys.executable,__file__,'child','--path',path,'--mode',mode,
                        '--envs',str(config['envs']),'--steps',str(config['steps']),'--out',str(directory)],check=True)
                current[path] = json.loads((directory/'result.json').read_text())
                runs.append(current[path])
            a = np.load(out/mode/f'repeat-{repeat}/pogema_original/final-state.npz')
            b = np.load(out/mode/f'repeat-{repeat}/jointloom/final-state.npz')
            ratios = {phase:current['pogema_original']['metrics'][phase]['wall_seconds']/
                current['jointloom']['metrics'][phase]['wall_seconds'] for phase in ('first','after_one','sustained')}
            pair = dict(mode=mode, repetition=repeat, speedup=ratios,
                final_state_equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files},
                completed_goals={p:r['completed_goals'] for p,r in current.items()})
            pairs.append(pair)
            (out/'summary.json').write_text(json.dumps(dict(config=config,runs=runs,pairs=pairs),indent=2))
            print(json.dumps(pair),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['compare','child'])
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--mode',choices=['standard_mapf','lifelong'])
    p.add_argument('--path',choices=['pogema_original','jointloom'])
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    c=dict(envs=args.envs,agents=256,steps=args.steps)
    if args.command=='child':child(c,args.path,args.mode,args.out.resolve())
    else:compare(c,args.out.resolve(),args.repetitions,[args.mode] if args.mode else ['standard_mapf','lifelong'])


if __name__=='__main__':main()
