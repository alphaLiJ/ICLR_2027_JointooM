"""Independent expert FIFOs packed into four ready complete configurations."""
import multiprocessing as mp
from pathlib import Path
import threading
import time
import traceback
import numpy as np
from expert.expert_running import RingBuffer
from live_source import _expert, FIELDS


class WorkerSink:
    def __init__(self, ring, timestamps):
        self.ring, self.timestamps = ring, timestamps

    def reserve_and_write(self, raw):
        # Retain the original reserve/write/publish protocol for a single FIFO.
        local = raw.copy()
        local[:, 0] = 0
        _, tensor = self.ring._validate_stage_block(local)
        seq, lo, hi = self.ring._reserve_stage_block(0)
        self.ring.cpu_buffer[lo:hi].copy_(tensor)
        self.ring.ready_flags[lo:hi] = [1] * len(local)
        self.timestamps[seq % self.ring.num_stage_slots] = time.perf_counter()
        self.ring._publish_stage_block(0, seq)


def choose_heads(heads, count, rng=None):
    """One head per ready worker; optionally sample the ready set uniformly."""
    if len(heads) < count:
        return None
    if rng is not None:
        return [heads[int(i)] for i in rng.choice(len(heads), count, replace=False)]
    return sorted(heads, key=lambda item: (item[1], item[0]))[:count]


class ReadySource:
    def __init__(self, output_ring, names, agents, seed, mode='lifelong', depth=512, selection='oldest'):
        self.ring, self.agents = output_ring, agents
        self.envs, self.batch_envs = len(names), output_ring.num_envs
        if selection not in ('oldest','random_ready'):
            raise ValueError(selection)
        self.selection = selection
        self.rng = np.random.default_rng(np.random.SeedSequence([seed,20261109]))
        self.empty_ready_polls = self.full_output_polls = 0
        self.start_event, self.stop_event = mp.Event(), mp.Event()
        self.errors = mp.Queue()
        self.shared = mp.Array('d', self.envs * len(FIELDS), lock=False)
        self.fifos = [RingBuffer(depth * agents, 8, num_envs=1, agents_per_env=agents)
                      for _ in names]
        self.timestamps = [mp.Array('d', depth, lock=False) for _ in names]
        self.read_seq = np.zeros(self.envs, dtype=np.int64)
        self.metadata = {}
        self.packed = self.sequence = 0
        self.thread = None
        self.ready = [mp.Event() for _ in names]
        self.processes = []
        directory = str(Path(__file__).resolve().parent)
        try:
            for e, name in enumerate(names):
                sink = WorkerSink(self.fifos[e], self.timestamps[e])
                proc = mp.Process(target=_expert, args=(directory, e, name, agents, seed,
                    mode, 1, 1000000000, sink, self.ready[e], self.start_event,
                    self.stop_event, self.errors, self.shared), daemon=True)
                proc.start()
                self.processes.append(proc)
            began = time.monotonic()
            while not all(event.is_set() for event in self.ready):
                self.check()
                if time.monotonic() - began > 180:
                    raise TimeoutError('ready source initialization')
                time.sleep(.01)
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if not self.errors.empty():
            raise RuntimeError(self.errors.get())
        if any(proc.exitcode is not None for proc in self.processes):
            raise RuntimeError('ready expert exited')

    def _pack_once(self):
        # A single packer owns the output ring; this check prevents blocking in
        # reserve_and_write at shutdown when the bounded output ring is full.
        with self.ring.reserve_lock:
            consumed = self.ring.shared_compute_ptr.value // self.ring.stage_rows
            if self.packed - consumed >= self.ring.num_stage_slots:
                self.full_output_polls += 1
                return False
        heads = []
        for e, fifo in enumerate(self.fifos):
            seq = int(self.read_seq[e])
            if fifo.is_stage_ready(seq):
                heads.append((e, float(self.timestamps[e][seq % fifo.num_stage_slots])))
        chosen = choose_heads(heads, self.batch_envs, self.rng if self.selection=='random_ready' else None)
        if chosen is None:
            self.empty_ready_polls += 1
            return False
        ids = np.array([e for e, _ in chosen], dtype=np.int64)
        sequences = self.read_seq[ids].copy()
        blocks = []
        for slot, (e, _) in enumerate(chosen):
            fifo = self.fifos[e]
            seq = int(self.read_seq[e])
            lo, hi = fifo._stage_row_bounds(seq)
            rows = fifo.cpu_buffer[lo:hi].numpy().astype(np.uint16, copy=True)
            rows[:, 0] = slot
            blocks.append(rows)
            fifo.release_stage(seq)
            fifo.shared_compute_ptr.value = (seq + 1) * fifo.stage_rows
            self.read_seq[e] += 1
        # Metadata/counters precede visibility of the output stage, including
        # when the learner immediately consumes its last published block.
        self.metadata[self.packed] = dict(worker_ids=ids, worker_sequences=sequences,
            published_seconds=np.array([t for _, t in chosen]))
        self.packed += 1
        for rows in blocks:
            self.ring.reserve_and_write(rows)
        return True

    def _loop(self):
        try:
            while not self.stop_event.is_set():
                self.check()
                if not self._pack_once():
                    time.sleep(.0005)
        except BaseException:
            self.errors.put(traceback.format_exc())

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.start_event.set()

    def take_metadata(self, sequence):
        return self.metadata.pop(sequence)

    def poll_rows(self):
        self.check()
        seq = self.sequence
        if not self.ring.is_stage_ready(seq):
            return None
        lo, hi = self.ring._stage_row_bounds(seq)
        rows = self.ring.cpu_buffer[lo:hi].numpy().astype(np.uint16, copy=True)
        metadata = self.take_metadata(seq)
        self.ring.release_stage(seq)
        self.ring.shared_compute_ptr.value = (seq + 1) * self.ring.stage_rows
        self.sequence += 1
        return rows, metadata

    def stats(self):
        values = np.array(self.shared).reshape(self.envs, len(FIELDS))
        published = [int(fifo.env_write_seq[0]) for fifo in self.fifos]
        return dict(totals=dict(zip(FIELDS, map(float, values.sum(0)))),
            workers=[dict(zip(FIELDS, map(float, row))) for row in values],
            published_per_env=published, packed_per_env=self.read_seq.tolist(),
            pending_per_env=(np.array(published) - self.read_seq).tolist(),
            packed_batches=self.packed, per_worker_depth=self.fifos[0].num_stage_slots,
            output_depth=self.ring.num_stage_slots,
            selection=self.selection, ready_set_shortage_polls=self.empty_ready_polls,
            output_full_polls=self.full_output_polls,
            output_ready_batches=max(0,min(map(int,self.ring.env_write_seq))-
                int(self.ring.shared_compute_ptr.value//self.ring.stage_rows)),
            scope='independent real expert streams; one configuration per selected worker; packed is not trained')

    def close(self):
        self.stop_event.set()
        self.start_event.set()
        if self.thread:
            self.thread.join(2)
        for proc in self.processes:
            proc.join(1)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self.errors.cancel_join_thread()
        self.errors.close()
