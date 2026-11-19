"""Dynamic source identities with historical CPU builders and existing GPU kernels."""
import multiprocessing as mp
import time
import traceback
import numpy as np
import torch
from expert.expert_running import RingBuffer, ExtremeMAPFPipeline
from mapf_cuda.simulation.grids import build_stateless_simulator
from native_extent_inputs import NativeExtentBatchBuilder
from ready_source import ReadySource
from references.host_inputs_original import CachedHostBuilder, HostInputs, HostPrefetch


class SelectedHostBuilder:
    def __init__(self, grids, agents):
        self.grids, self.agents, self.builders = grids, agents, {}

    def build(self, rows, worker_ids):
        refs = []
        for slot, worker in enumerate(worker_ids):
            worker = int(worker)
            if worker not in self.builders:
                self.builders[worker] = CachedHostBuilder('magat', self.grids[worker:worker+1], self.agents)
            block = rows[slot*self.agents:(slot+1)*self.agents].copy()
            block[:, 0] = 0
            # Each CPU process compares against its own actual predecessor.
            block[:, 7] = 0
            refs.append(self.builders[worker].build(block).arrays)
        n = len(worker_ids)
        arrays = dict(x=np.concatenate([r['x'] for r in refs]),
            edge_index=np.concatenate([r['edge_index'] + slot*self.agents for slot, r in enumerate(refs)], axis=1),
            edge_attr=np.concatenate([r['edge_attr'] for r in refs]),
            y=np.concatenate([r['y'] for r in refs]),
            batch=np.repeat(np.arange(n, dtype=np.int64), self.agents),
            ptr=np.arange(n+1, dtype=np.int64)*self.agents,
            arrived=np.concatenate([r['arrived'] for r in refs]))
        return HostInputs('magat', arrays, sum(b.cache_rebuilds for b in self.builders.values()))


def _prepare(grids, agents, tasks, results, ready):
    try:
        from expert.expert_running import configure_cpu_thread_limits
        configure_cpu_thread_limits()
        builder = SelectedHostBuilder(grids, agents)
        ready.set()
        while True:
            task = tasks.get()
            if task is None:
                return
            seq, worker_ids, rows = task
            prepared = builder.build(rows, worker_ids)
            prepared.stage_sequence = seq
            results.put((seq, prepared, None))
    except BaseException:
        results.put((-1, None, traceback.format_exc()))
        ready.set()


class SelectedHostPrefetch(HostPrefetch):
    def __init__(self, grids, agents, workers=8, depth=16):
        ctx = mp.get_context('spawn')
        self.tasks = [ctx.Queue(maxsize=depth) for _ in range(workers)]
        self.results = ctx.Queue(maxsize=depth)
        self.ready = [ctx.Event() for _ in self.tasks]
        self.processes = [ctx.Process(target=_prepare, args=(grids, agents, task,
            self.results, ready), daemon=True) for task, ready in zip(self.tasks, self.ready)]
        self.depth, self.timeout = depth, 120
        self.submitted = self.consumed = 0
        self.pending = {}
        try:
            for proc in self.processes:
                proc.start()
            began = time.monotonic()
            while not all(event.is_set() for event in self.ready):
                if time.monotonic()-began > self.timeout or any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError('selected CPU prefetch initialization')
                time.sleep(.01)
        except BaseException:
            self.close()
            raise

    def submit(self, rows, worker_ids):
        if self.submitted-self.consumed >= self.depth:
            raise RuntimeError('prefetch admission bound exceeded')
        seq = self.submitted
        self.tasks[seq % len(self.tasks)].put((seq, np.array(worker_ids, copy=True),
            np.array(rows, copy=True)), timeout=self.timeout)
        self.submitted += 1


class SelectedGpuBuilder:
    def __init__(self, grids, agents, batch_envs=4):
        self.bank = build_stateless_simulator(grids, num_agents=agents, device='cuda:0')
        self.work = build_stateless_simulator(grids[:batch_envs], num_agents=agents, device='cuda:0')
        self.work.pyg_builder_mode = 'local_gather'
        self.work.pyg_local_gather_impl = 'auto'
        self.builder = NativeExtentBatchBuilder([g.shape for g in grids[:batch_envs]], agents, 'cuda:0')
        self.bank_maps = self.bank.memory_tensors['grid_compressed']
        self.work_maps = self.work.memory_tensors['grid_compressed']
        self.seen = np.zeros(len(grids), dtype=bool)
        self.agents = agents
        self.refreshes = self.d2d_bytes = 0

    def build(self, raw, worker_ids, materialize_edges=False):
        ids = torch.as_tensor(worker_ids, device='cuda:0', dtype=torch.int64)
        for slot, worker in enumerate(worker_ids):
            if not self.seen[worker]:
                raw[slot*self.agents:(slot+1)*self.agents, 7] = 1
                self.seen[worker] = True
        changed = raw[raw[:, 7] != 0]
        if len(changed):
            changed[:, 0] = ids[changed[:, 0].long()].to(torch.int16)
            self.bank.update_energy_maps(changed, len(changed))
            self.refreshes += len(changed)
        torch.index_select(self.bank_maps, 0, ids, out=self.work_maps)
        torch.index_select(self.bank.energy_maps, 0, ids, out=self.work.energy_maps)
        self.d2d_bytes += self.work_maps.numel()*self.work_maps.element_size() + self.work.energy_maps.numel()
        self.work.refresh_compact_state_from_raw_batch(raw)
        return self.builder.build(self.work, raw, materialize_edges=materialize_edges)


class ReadyInputs:
    def __init__(self, grids, names, agents, seed, path, mode='lifelong', depth=512, selection='oldest'):
        self.path, self.agents = path, agents
        self.bytes = self.refreshes = self.d2d_bytes = 0
        self.pipeline = self.dma_thread = self.host = None
        self.pending_metadata, self.rebuilds = {}, {}
        self.history_ids, self.history_sequences = [], []
        self.trained = np.zeros(len(grids), dtype=np.int64)
        self.device_ready = None
        if path == 'jointloom':
            self.gpu = SelectedGpuBuilder(grids, agents)
            self.pipeline = ExtremeMAPFPipeline(self.gpu.work, capacity=4*agents*depth,
                batch_threshold=4*agents, device='cuda:0')
            self.pipeline.initialize()
            ring = self.pipeline.ring_buffer
        else:
            ring = RingBuffer(4*agents*depth, 8, num_envs=4, agents_per_env=agents)
        self.source = ReadySource(ring, names, agents, seed, mode, depth, selection)
        try:
            if path != 'jointloom':
                self.host = SelectedHostPrefetch(grids, agents, 8, 16)
                self.stream = torch.cuda.Stream(device='cuda:0')
                self.stream.wait_stream(torch.cuda.current_stream())
        except BaseException:
            self.source.close()
            raise

    def start(self):
        if self.pipeline:
            self.dma_thread = self.pipeline.dma_worker.start()
        self.source.start()

    def _upload(self, prepared):
        with torch.cuda.stream(self.stream):
            data = prepared.to_device('cuda:0')
            event = torch.cuda.Event()
            event.record(self.stream)
        return prepared, data, event

    def poll(self, sequence):
        self.source.check()
        if self.pipeline:
            size = 4*self.agents
            if self.pipeline.dma_worker.dma_read_ptr-self.pipeline.compute_ptr < size:
                return None
            meta = self.source.take_metadata(sequence)
            self.pipeline.dma_event.synchronize()
            raw = self.pipeline.extract_env_aligned_batch(size, self.agents, 4)
            if raw is None:
                raise RuntimeError('ready DMA stage disappeared')
            self.pending_metadata[sequence] = meta
            self.bytes += raw.numel()*raw.element_size() + 4*8
            data = self.gpu.build(raw, meta['worker_ids'])
            self.refreshes, self.d2d_bytes = self.gpu.refreshes, self.gpu.d2d_bytes
            return data
        while self.host.submitted-self.host.consumed < self.host.depth:
            current = self.source.poll_rows()
            if current is None:
                break
            rows, meta = current
            self.pending_metadata[self.host.submitted] = meta
            self.host.submit(rows, meta['worker_ids'])
        if self.device_ready is None:
            prepared = self.host.try_get()
            if prepared is None:
                return None
            self.device_ready = self._upload(prepared)
        prepared, data, event = self.device_ready
        torch.cuda.current_stream().wait_event(event)
        for name in ('x', 'edge_index', 'edge_attr', 'batch', 'ptr', 'y', 'arrived', 'terminated'):
            value = getattr(data, name, None)
            if value is not None:
                value.record_stream(torch.cuda.current_stream())
        self.device_ready = None
        if prepared.stage_sequence != sequence:
            raise RuntimeError('ready input sequence mismatch')
        self.bytes += prepared.nbytes
        self.rebuilds[sequence % len(self.host.processes)] = prepared.cache_rebuilds
        self.refreshes = sum(self.rebuilds.values())
        future = self.host.try_get()
        if future is not None:
            self.device_ready = self._upload(future)
        return data

    def release(self, sequence):
        meta = self.pending_metadata.pop(sequence)
        self.trained[meta['worker_ids']] += 1
        self.history_ids.append(meta['worker_ids'].copy())
        self.history_sequences.append(meta['worker_sequences'].copy())

    def stats(self):
        return dict(**self.source.stats(), trained_per_env=self.trained.tolist(),
            selected_not_trained_per_env=(self.source.read_seq-self.trained).tolist())

    def close(self):
        if self.pipeline:
            self.pipeline.shutdown()
            if self.dma_thread:
                self.dma_thread.join(2)
        self.source.close()
        if self.host:
            self.host.close()
