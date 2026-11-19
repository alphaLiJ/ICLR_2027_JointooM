"""Matched-action component timing, NOT a network-free full SF training run.

CPU: the same CostTracerPogema + unmodified SF vector wrappers as sf_native.py,
8 workers, shared CPU observations, GPU actions -> next observation on GPU.
CUDA: the existing CudaEnvironment, GPU actions -> next observation on GPU.
No inference, learning, GAE, SF scheduler/batcher or rollout storage is timed.
"""
import bootstrap
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from environments import CudaEnvironment
from model import CostTracer, configure_torch
from sf_tasks import SF_ROOT, load


def cpu_worker(conn, seed, ne, worker, actions, observations, trajectory=None, policy_outputs=None):
    try:
        from sample_factory.algo.utils.make_env import BatchedVecEnv, SequentialVectorizeWrapper
        from sample_factory.utils.attr_dict import AttrDict
        from sf_native import CostTracerPogema
        torch.set_num_threads(1)
        per_worker = ne // 8
        per_split = per_worker // 2
        cfg = AttrDict(seed=seed, num_workers=8, num_envs_per_worker=per_worker)
        inners = []
        splits = []
        for split in range(2):
            envs = []
            for j in range(per_split):
                e = worker * per_worker + split * per_split + j
                inner = CostTracerPogema(cfg, dict(env_id=e))
                inners.append(inner)
                envs.append(BatchedVecEnv(inner))
            vector = envs[0] if len(envs) == 1 else SequentialVectorizeWrapper(envs)
            lo = worker * per_worker + split * per_split
            hi = lo + per_split
            obs, _ = vector.reset()
            observations[lo:hi].copy_(obs['obs'].reshape(per_split,128,2,11,11))
            splits.append((lo, hi, vector))
        conn.send('ready')
        while True:
            command = conn.recv()
            rollout_step = None
            if isinstance(command,tuple):
                command,rollout_step = command
            if command == 'close':
                break
            if command == 'state':
                states = [env.inner.state() for env in inners]
                conn.send({k: np.concatenate([s[k] for s in states]) for k in states[0]})
                continue
            if command != 'step':
                raise ValueError(command)
            for lo, hi, vector in splits:
                if trajectory is not None:
                    at=(slice(lo*128,hi*128),rollout_step)
                    trajectory['obs']['obs'][at].copy_(observations[lo:hi].reshape(-1,2,11,11))
                    output=policy_outputs[lo:hi].reshape(-1,8)
                    trajectory['actions'][at].copy_(output[:,:1])
                    trajectory['log_prob_actions'][at].copy_(output[:,1])
                    trajectory['values'][at].copy_(output[:,2])
                    trajectory['action_logits'][at].copy_(output[:,3:])
                    trajectory['policy_id'][at].fill_(0)
                    trajectory['policy_version'][at].fill_(0)
                obs, reward, terminated, truncated, _ = vector.step(actions[lo:hi].reshape(-1).numpy())
                observations[lo:hi].copy_(obs['obs'].reshape(hi-lo,128,2,11,11))
                if trajectory is not None:
                    trajectory['rewards'][at].copy_(reward)
                    trajectory['dones'][at].copy_(terminated|truncated)
                    trajectory['time_outs'][at].copy_(truncated)
                    if rollout_step==31:
                        trajectory['obs']['obs'][lo*128:hi*128,32].copy_(obs['obs'])
            conn.send('done')
    except BaseException:
        conn.send(dict(error=traceback.format_exc()))
    finally:
        conn.close()


def receive(conn):
    if not conn.poll(120):
        raise TimeoutError('environment worker')
    value = conn.recv()
    if isinstance(value, dict) and 'error' in value:
        raise RuntimeError(value['error'])
    return value


def record_actions(seed, ne, steps, output):
    """A frozen, actual short-run checkpoint defines identical trajectories."""
    configure_torch(seed)
    torch.backends.cudnn.benchmark = True
    env = CudaEnvironment(load(seed, num_envs=ne))
    checkpoint = SF_ROOT/'supply-scaling-8cores-v1'/f'jointloom-e{ne}-s{seed}'/'final.pt'
    model = CostTracer().cuda()
    model.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
    rng = torch.Generator(device='cuda').manual_seed(seed+47000)
    actions = torch.empty((steps,ne,128),dtype=torch.int64,device='cuda')
    goals = torch.zeros((),device='cuda')
    with torch.no_grad():
        for i in range(steps):
            logits, _ = model(env.obs.reshape(ne*128,2,11,11))
            act = torch.multinomial(logits.softmax(-1),1,generator=rng).reshape(ne,128)
            actions[i].copy_(act)
            _,_,hit,_ = env.step(act)
            goals.add_(hit.sum())
            if (i+1)%512 == 0:
                env.reset_due()
    torch.cuda.synchronize()
    np.save(output/'actions.npy', actions.cpu().numpy())
    info = dict(checkpoint=str(checkpoint), goals_during_trace=float(goals),
                steps=steps, seed=seed, num_envs=ne)
    del env, model
    torch.cuda.empty_cache()
    return actions, info


def cpu_measure(actions, seed, ne, steps):
    shared_actions = torch.empty((ne,128),dtype=torch.int32).share_memory_()
    shared_obs = torch.empty((ne,128,2,11,11)).share_memory_()
    gpu_obs = torch.empty_like(shared_obs,device='cuda')
    ctx = mp.get_context('spawn')
    conns, processes = [], []
    try:
        for worker in range(8):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=cpu_worker,args=(child,seed,ne,worker,shared_actions,shared_obs))
            process.start()
            child.close()
            processes.append(process)
            conns.append(parent)
        for conn in conns:
            assert receive(conn) == 'ready'
        rows, checks = [], []
        torch.cuda.synchronize()
        for block in range(steps//32):
            start = time.perf_counter()
            for i in range(block*32,(block+1)*32):
                # Same direction of device crossings as native SF's CPU environment.
                shared_actions.copy_(actions[i].to(torch.int32).cpu())
                for conn in conns:
                    conn.send('step')
                for conn in conns:
                    assert receive(conn) == 'done'
                gpu_obs.copy_(shared_obs,non_blocking=True)
            torch.cuda.synchronize()
            rows.append(time.perf_counter()-start)
            if block in (7,14):
                for conn in conns:
                    conn.send('state')
                states = [receive(conn) for conn in conns]
                checks.append(dict(step=(block+1)*32,
                    state={k:np.concatenate([s[k] for s in states]) for k in states[0]},
                    obs=gpu_obs.cpu().numpy()))
        for conn in conns:
            conn.send('state')
        states = [receive(conn) for conn in conns]
        state = {k:np.concatenate([s[k] for s in states]) for k in states[0]}
        return rows, state, gpu_obs.cpu().numpy(), checks
    finally:
        for conn in conns:
            try:
                conn.send('close')
            except (BrokenPipeError,EOFError):
                pass
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join()


def cuda_measure(actions, seed, ne, steps):
    env = CudaEnvironment(load(seed,num_envs=ne))
    rows, checks = [], []
    torch.cuda.synchronize()
    for block in range(steps//32):
        start = time.perf_counter()
        for i in range(block*32,(block+1)*32):
            env.step(actions[i])
            if (i+1)%512 == 0:
                env.reset_due()
        torch.cuda.synchronize()
        rows.append(time.perf_counter()-start)
        if block in (7,14):
            checks.append(dict(step=(block+1)*32,state=env.state(),obs=env.obs.cpu().numpy()))
    return rows, env.state(), env.obs.cpu().numpy(), checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--envs',type=int,nargs='+',default=[16,32])
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--steps',type=int,default=512)
    args = parser.parse_args()
    if args.steps%32:
        raise ValueError('32-step timing blocks required')
    args.output_root.mkdir(parents=True,exist_ok=False)
    metadata = dict(metric='matched-action synchronous environment-to-GPU-observation delivery',
        cpu_affinity=sorted(os.sched_getaffinity(0)),workers=8,worker_splits=2,
        excludes=['network forward/backward','GAE','PPO loss','optimizer','rollout storage',
                  'native SF scheduler/inference/batcher/learner processes'],
        cpu_backend='actual CostTracerPogema with unmodified SF2.1.1 BatchedVecEnv/SequentialVectorizeWrapper',
        standalone_driver='8 workers, shared CPU tensors, per-step completion barrier; not full native SF',
        includes=['GPU-to-CPU action transfer on CPU path','CPU-worker step synchronization',
                  'soft collision movement','lifelong rewards/goals/distances','11x11 input construction',
                  'next observation upload to GPU on CPU path','512-step time-limit reset'],
        timing='synchronized wall per 32 joint timesteps; exclude first 2 blocks',
        gpu=torch.cuda.get_device_name(),sample_factory=__import__('importlib.metadata').metadata.version('sample-factory'))
    (args.output_root/'metadata.json').write_text(json.dumps(metadata,indent=2))
    results = []
    for seed in args.seeds:
        for ne in args.envs:
            output = args.output_root/f'e{ne}-s{seed}'
            output.mkdir()
            print(f'recording e{ne} s{seed}',flush=True)
            actions, trace_info = record_actions(seed,ne,args.steps,output)
            measured = {}
            order = ('cpu','cuda') if seed%2 == 0 else ('cuda','cpu')
            for profile in order:
                print(f'measuring {profile} e{ne} s{seed}',flush=True)
                fn = cpu_measure if profile == 'cpu' else cuda_measure
                measured[profile] = fn(actions,seed,ne,args.steps)
            cpu_rows, cpu_state, cpu_obs, cpu_checks = measured['cpu']
            gpu_rows, gpu_state, gpu_obs, gpu_checks = measured['cuda']
            equal = {k:bool(np.array_equal(cpu_state[k],gpu_state[k])) for k in cpu_state}
            obs_difference = float(np.max(np.abs(cpu_obs-gpu_obs)))
            if not all(equal.values()) or obs_difference>1e-6:
                raise AssertionError(dict(state_equal=equal,obs_difference=obs_difference))
            check_results = []
            for c,g in zip(cpu_checks,gpu_checks):
                state_equal = {k:bool(np.array_equal(c['state'][k],g['state'][k])) for k in c['state']}
                error = float(np.max(np.abs(c['obs']-g['obs'])))
                if not all(state_equal.values()) or error>1e-6:
                    raise AssertionError(dict(step=c['step'],state_equal=state_equal,obs_error=error))
                check_results.append(dict(step=c['step'],state_equal=state_equal,observation_max_difference=error))
            row = dict(trace_info,cpu_block_seconds=cpu_rows,cuda_block_seconds=gpu_rows,
                cpu_mean_seconds=float(np.mean(cpu_rows[2:])),cuda_mean_seconds=float(np.mean(gpu_rows[2:])),
                state_equal=equal,final_observation_max_difference=obs_difference,
                before_reset_checks=check_results)
            row['component_speedup'] = row['cpu_mean_seconds']/row['cuda_mean_seconds']
            (output/'result.json').write_text(json.dumps(row,indent=2))
            results.append(row)
            (args.output_root/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:row[k] for k in ('seed','num_envs','cpu_mean_seconds','cuda_mean_seconds','component_speedup')}),flush=True)
    summary = {}
    for ne in args.envs:
        rows = [r for r in results if r['num_envs']==ne]
        cpu = float(np.mean([r['cpu_mean_seconds'] for r in rows]))
        gpu = float(np.mean([r['cuda_mean_seconds'] for r in rows]))
        summary[ne] = dict(cpu_seconds=cpu,cuda_seconds=gpu,component_speedup=cpu/gpu,
            individual_speedups=[r['component_speedup'] for r in rows])
    (args.output_root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
