"""Small checks for delayed workers, FIFO wrap and source-indexed caches."""
import bootstrap
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from expert.expert_running import RingBuffer
from order_inputs import ConfigurationCorpus
from ready_source import WorkerSink, ReadySource, choose_heads
from ready_inputs import SelectedHostBuilder, SelectedGpuBuilder
from references.host_inputs_original import CachedHostBuilder
from run_il_order import configure, make_learner


def check_fifo(corpus):
    source = ReadySource.__new__(ReadySource)
    source.envs, source.batch_envs, source.agents = 8, 4, corpus.agents
    source.ring = RingBuffer(4*corpus.agents*2, 8, num_envs=4, agents_per_env=corpus.agents)
    source.fifos = [RingBuffer(corpus.agents*2, 8, num_envs=1, agents_per_env=corpus.agents) for _ in range(8)]
    source.timestamps = [mp.Array('d', 2, lock=False) for _ in range(8)]
    source.read_seq = np.zeros(8, np.int64)
    source.metadata, source.processes = {}, []
    source.selection = 'oldest'
    source.rng = np.random.default_rng(42)
    source.empty_ready_polls = source.full_output_polls = 0
    source.errors = mp.Queue()
    source.packed = source.sequence = 0
    sinks = [WorkerSink(ring, times) for ring, times in zip(source.fifos, source.timestamps)]
    def publish(worker, index):
        sinks[worker].reserve_and_write(corpus.arrays[worker%4][index])
    for e in [1,2,3]:
        publish(e, 0)
    assert not source._pack_once(), 'three available workers cannot make a four-configuration batch'
    for e in [4,5]:
        publish(e, 0)
    assert source._pack_once(), 'worker0 must not block four other ready workers'
    results = [source.poll_rows()[1]]
    for e in [0,6,7]:
        publish(e, 0)
    assert source._pack_once()
    results.append(source.poll_rows()[1])
    for group in [[1,2,3,4],[5,0,6,7]]:
        for e in group:
            publish(e, 1)
        assert source._pack_once()
        results.append(source.poll_rows()[1])
    ids = np.stack([r['worker_ids'] for r in results])
    sequences = np.stack([r['worker_sequences'] for r in results])
    assert ids[0].tolist() == [1,2,3,4]
    assert 0 in ids[1], 'the delayed worker must be retained for a later batch'
    for e in range(8):
        assert sequences[ids==e].tolist() == [0,1]
    source.errors.close()
    return dict(worker_ids=ids.tolist(), worker_sequences=sequences.tolist(),
        slow_worker_does_not_block=True, delayed_worker_retained=True, ring_wrap_fifo=True)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    configure()
    began = time.perf_counter()
    corpus = ConfigurationCorpus(args.source)
    fifo = check_fifo(corpus)
    grids = np.concatenate([corpus.grids]*(args.producer_envs//4))
    heads=[(e,float(e)) for e in range(args.producer_envs)]
    a,b=np.random.default_rng(42),np.random.default_rng(42)
    draws=[]
    for _ in range(16):
        first,second=choose_heads(heads,4,a),choose_heads(heads,4,b)
        assert first==second
        assert len({e for e,_ in first})==4
        draws.append([e for e,_ in first])
    assert len({tuple(x) for x in draws})>1
    assert choose_heads(heads[:3],4,a) is None
    cpu = SelectedHostBuilder(grids, corpus.agents)
    gpu = SelectedGpuBuilder(grids, corpus.agents)
    learner = make_learner(42)
    previous, results = {}, []
    cases=([([0,6,9,15],0),([1,7,8,14],0),([15,6,0,9],64)] if args.producer_envs==16 else
        [([0,2,5,7],0),([1,3,4,6],0),([7,2,0,5],64)])
    for ids, index in cases:
        blocks = []
        for slot, worker in enumerate(ids):
            block = corpus.arrays[worker%4][index].copy()
            block[:,0] = slot
            block[:,7] = (np.ones(corpus.agents, bool) if worker not in previous else
                np.any(block[:,4:6] != previous[worker], axis=1))
            previous[worker] = block[:,4:6].copy()
            blocks.append(block)
        rows = np.concatenate(blocks)
        prepared = cpu.build(rows, ids)
        fresh_rows = rows.copy()
        fresh_rows[:,7] = 0
        fresh = CachedHostBuilder('magat', grids[ids], corpus.agents).build(fresh_rows)
        for name in prepared.arrays:
            np.testing.assert_array_equal(prepared.arrays[name], fresh.arrays[name])
        raw = torch.as_tensor(rows.astype(np.int16), device='cuda:0')
        data = gpu.build(raw, np.asarray(ids), materialize_edges=True)
        err = float(np.max(np.abs(data.x.cpu().numpy()-prepared.arrays['x'])))
        assert err < 1e-6, err
        for name in ['edge_index','edge_attr','y','batch','ptr','arrived']:
            np.testing.assert_array_equal(getattr(data,name).cpu().numpy(), prepared.arrays[name])
        reference = prepared.to_device('cuda:0')
        learner.model.eval()
        with torch.no_grad():
            a, b = float(learner.loss_from_batch(reference)), float(learner.loss_from_batch(data))
        assert abs(a-b) < 1e-5, (a,b)
        learner.model.train()
        loss = float(learner.train_step_from_batch(data))
        assert np.isfinite(loss)
        results.append(dict(worker_ids=ids, source_index=index, max_input_error=err,
            eval_ce_difference=abs(a-b), training_loss=loss, cache_refreshes=gpu.refreshes))
    torch.cuda.synchronize()
    result = dict(fifo=fifo, dynamic_inputs=results, producer_envs=args.producer_envs,
        random_ready_draws=draws, actual_optimizer_updates=len(results),
        wall_seconds=time.perf_counter()-began, cache_selection_d2d_bytes=gpu.d2d_bytes)
    (args.out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--producer-envs',type=int,choices=[8,16],default=8)
    args = p.parse_args()
    mp.set_start_method('spawn', force=True)
    run(args)
