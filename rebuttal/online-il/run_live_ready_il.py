"""Actual online IL: independent experts, four ready joint states per update."""
import bootstrap
import argparse
import gc
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from order_inputs import ConfigurationCorpus, CompactReplay
from ready_inputs import ReadyInputs
from run_il_order import MAP_NAMES, load_maps, configure, make_learner


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'result.json').exists():
        raise FileExistsError(args.out)
    configure()
    setup = time.perf_counter()
    if args.producer_envs < 4 or args.producer_envs%4:
        raise ValueError('repeat the original four maps with at least four producers')
    names = MAP_NAMES * (args.producer_envs//4)
    grids = load_maps(names)
    corpus = ConfigurationCorpus(bootstrap.input_path('JOINTLOOM_RAW_CORPUS', bootstrap.ROOT/'artifacts/rebuttal/magat-records'), stages=1)
    warm = CompactReplay(corpus)
    warm.upload(corpus.stage([0]*4), 0)
    data = warm.get(0)
    learner = make_learner(args.seed)
    for _ in range(20):
        learner.train_step_from_batch(data)
    torch.cuda.synchronize()
    del learner
    gc.collect()
    learner = make_learner(args.seed)
    learner.model.eval()
    initial = float(learner.loss_from_batch(data))
    learner.model.train()
    warm.release(0)
    del warm, data
    gc.collect()
    inputs = ReadyInputs(grids, names, 512, args.seed, args.path, args.mode, selection=args.selection)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-setup
    torch.manual_seed(args.seed+1000)
    torch.cuda.reset_peak_memory_stats()
    losses, acquisition = [], 0.
    started = time.perf_counter()
    inputs.start()
    try:
        while time.perf_counter()-started < args.seconds:
            t = time.perf_counter()
            batch = inputs.poll(len(losses))
            if batch is None:
                time.sleep(.0005)
                acquisition += time.perf_counter()-t
                continue
            acquisition += time.perf_counter()-t
            sequence = len(losses)
            losses.append(learner.train_step_from_batch(batch))
            inputs.release(sequence)
            if len(losses) % 32 == 0:
                torch.cuda.synchronize()
                progress = dict(updates=len(losses), elapsed_seconds=time.perf_counter()-started,
                    trained_per_env=inputs.trained.tolist())
                (args.out/'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        stats = inputs.stats()
        values = torch.stack(losses).cpu().numpy() if losses else np.empty(0, np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError('nonfinite ready4 loss')
        worker_ids = np.asarray(inputs.history_ids, dtype=np.int64).reshape(-1, 4)
        worker_sequences = np.asarray(inputs.history_sequences, dtype=np.int64).reshape(-1, 4)
        for worker in range(args.producer_envs):
            actual = worker_sequences[worker_ids == worker]
            if not np.array_equal(actual, np.arange(len(actual))):
                raise RuntimeError('per-worker consumption must be FIFO without duplicates or gaps')
        if any(len(set(row)) != 4 for row in worker_ids):
            raise RuntimeError('a batch must use four distinct workers')
        np.save(args.out/'losses.npy', values)
        np.savez_compressed(args.out/'consumption.npz', worker_ids=worker_ids, worker_sequences=worker_sequences)
        torch.save(dict(model=learner.model.state_dict(), optimizer=learner.optimizer.state_dict(),
            scheduler=learner.scheduler.state_dict(), optimizer_updates=len(losses)), args.out/'checkpoint.pt')
        result = dict(path=args.path, mode=args.mode, seed=args.seed, agents=512,
            producer_envs=args.producer_envs, batch_envs=4, batch=2048, per_worker_depth=512, output_depth=512,
            host_workers=8 if args.path!='jointloom' else 0, host_depth=16 if args.path!='jointloom' else 0,
            target_seconds=args.seconds, wall_seconds=seconds, optimizer_updates=len(losses),
            samples_per_second=len(losses)*2048/seconds, joint_configurations_per_second=len(losses)*4/seconds,
            setup_seconds=setup_seconds, initial_common_eval_loss=initial,
            first_loss=float(values[0]) if len(values) else None,
            last_loss=float(values[-1]) if len(values) else None,
            consumer_input_seconds=acquisition, h2d_bytes=inputs.bytes,
            cache_refreshes=inputs.refreshes, cache_selection_d2d_bytes=inputs.d2d_bytes,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), source=stats,
            map_names=names, cpu_affinity=list(range(16)), precision='float32; TF32 disabled',
            expert_timeouts_seconds=[1,5,10,30],
            selection=args.selection,
            scheduling='four distinct ready worker heads; '+args.selection+'; one complete configuration each; per-worker FIFO',
            timing_protocol='empty real expert source released inside timer; deadline-aware polling; finish current update; teardown excluded',
            scope='actual expert/Pogema production, ready4 packing, input construction, transport, and MAGAT CE/Adam; independent online traces; not a convergence experiment')
        (args.out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({k:result[k] for k in ['path','wall_seconds','optimizer_updates','samples_per_second','joint_configurations_per_second']}), flush=True)
    finally:
        inputs.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--path', choices=['jointloom','cpu_cached_prefetch'], required=True)
    p.add_argument('--seconds', type=float, default=300.)
    p.add_argument('--producer-envs',type=int,default=8)
    p.add_argument('--mode',choices=['standard_mapf','lifelong'],default='lifelong')
    p.add_argument('--selection',choices=['oldest','random_ready'],default='oldest')
    args = p.parse_args()
    mp.set_start_method('spawn', force=True)
    run(args)
