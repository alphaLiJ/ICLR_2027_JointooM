"""Cached CPU inputs and a bounded, ordered process prefetcher.

Workers own only CPU state. Device copies and their lifetimes remain with the
consumer. These adapters preserve the existing input/optimizer contracts.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import dataclass

import numpy as np


def validate_stage(rows, grids, agents, features):
    rows = np.asarray(rows)
    if rows.dtype != np.uint16 or rows.shape != (len(grids) * agents, features):
        raise ValueError("stage must contain complete uint16 environment-major records")
    blocks = rows.reshape(len(grids), agents, features)
    if not np.array_equal(blocks[:, :, 0], np.broadcast_to(np.arange(len(grids))[:, None], blocks.shape[:2])):
        raise ValueError("stage environment order mismatch")
    if not np.array_equal(blocks[:, :, 1], np.broadcast_to(np.arange(agents), blocks.shape[:2])):
        raise ValueError("stage agent order mismatch")
    if np.any(rows[:, 6] > 4) or np.any(rows[:, 7] > 1):
        raise ValueError("invalid action or refresh flag")
    for grid, block in zip(grids, blocks):
        for coords in (block[:, 2:4], block[:, 4:6]):
            if np.any(coords[:, 0] >= grid.shape[0]) or np.any(coords[:, 1] >= grid.shape[1]):
                raise ValueError("coordinates outside map")
            if np.any(grid[coords[:, 0], coords[:, 1]]):
                raise ValueError("coordinates on obstacles")
            if len(np.unique(coords, axis=0)) != agents:
                raise ValueError("duplicate joint positions or goals")
    return blocks


@dataclass
class HostInputs:
    backend: str
    arrays: dict[str, np.ndarray]
    cache_rebuilds: int
    stage_sequence: int | None = None
    stage_sha256: str | None = None

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self.arrays.values())

    def to_device(self, device):
        import torch
        # Keep pinned owners alive on the returned object through backward.
        pinned = {k: torch.from_numpy(np.ascontiguousarray(v)).pin_memory()
                  for k, v in self.arrays.items()}
        values = {k: v.to(device, non_blocking=True) for k, v in pinned.items()}
        if self.backend == "magat":
            from expert.fixed_magat_plus_runtime import RuntimePyGBatch
            obj = RuntimePyGBatch(**values, terminated=torch.zeros_like(values["arrived"]))
        else:
            obj = PreparedTokens(values)
        obj._pinned_owners = pinned
        return obj


class PreparedTokens:
    """Builder protocol adapter; training/shuffle code stays unchanged."""
    def __init__(self, values):
        import torch
        self.tokens = values["tokens"]
        self.labels = values["labels"]
        self.active_mask = values["active_mask"]
        self.diagnostics = torch.zeros(4, dtype=torch.int64, device=self.tokens.device)

    def build_tokens(self, raw_stage):
        # The caller supplies an already prepared, ordered complete stage.
        if raw_stage is not None:
            raise ValueError("prepared tokens must not rebuild from another stage")


class CachedHostBuilder:
    def __init__(self, backend, grids, agents):
        if backend not in ("magat", "mapf_gpt") or agents <= 0:
            raise ValueError("unsupported builder or agent count")
        self.backend, self.agents = backend, agents
        self.cache_rebuilds = 0
        self.replace_maps(grids)

    def replace_maps(self, grids):
        grids = np.asarray(grids, dtype=np.uint8)
        if grids.ndim != 3 or not len(grids) or np.any(grids > 1):
            raise ValueError("expected nonempty binary [environment,height,width] maps")
        self.grids = grids.copy()
        self.goals = [None] * len(grids)
        self.envs = [None] * len(grids)
        self.calculators = [None] * len(grids)
        self.costs = [[None] * self.agents for _ in grids]

    def build(self, rows):
        features = 8 if self.backend == "magat" else 13
        blocks = validate_stage(rows, self.grids, self.agents, features)
        if self.backend == "mapf_gpt":
            from expert.mapf_gpt_schema import _bfs_cost_to_go, tokenize_stage_reference
            for e, block in enumerate(blocks):
                changed = np.ones(self.agents, dtype=bool) if self.goals[e] is None else np.any(self.goals[e] != block[:, 4:6], axis=1)
                changed |= block[:, 7] != 0
                for a in np.flatnonzero(changed):
                    self.costs[e][a] = _bfs_cost_to_go(self.grids[e], tuple(map(int, block[a, 4:6])))
                    self.cache_rebuilds += 1
                self.goals[e] = block[:, 4:6].copy()
            tokens, labels = tokenize_stage_reference(self.grids, rows, num_agents=self.agents, cost_maps_by_env=self.costs)
            arrays = dict(tokens=tokens, labels=labels, active_mask=np.any(rows[:, 2:4] != rows[:, 4:6], axis=1))
        else:
            from expert.fixed_magat_plus_runtime import OBS_RADIUS
            from expert.magat_reference_oracle import build_magat_reference_batch_from_env
            from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator, DistTable
            from pogema.envs import PogemaCoopFinish
            from pogema.grid_config import GridConfig
            refs = []
            for e, block in enumerate(blocks):
                env = self.envs[e]
                if env is None:
                    env = PogemaCoopFinish(grid_config=GridConfig(
                        map=self.grids[e].tolist(), agents_xy=block[:, 2:4].tolist(),
                        targets_xy=block[:, 4:6].tolist(), num_agents=self.agents,
                        obs_radius=OBS_RADIUS, on_target="nothing", collision_system="soft",
                        observation_type="MAPF", max_episode_steps=256, empty_outside=True))
                    env._initialize_grid()
                    env.num_agents = self.agents
                    self.envs[e] = env
                # Restore a full expert configuration, not an action replay.
                env.grid.positions.fill(0)
                xy = block[:, 2:4].astype(np.int64) + OBS_RADIUS
                env.grid.positions_xy = [tuple(p) for p in xy]
                env.grid.positions[xy[:, 0], xy[:, 1]] = 1
                env.grid.finishes_xy = [tuple(p) for p in block[:, 4:6].astype(np.int64) + OBS_RADIUS]
                env.grid.is_active = {a: True for a in range(self.agents)}
                env.update_was_on_goal()
                if self.calculators[e] is None:
                    self.calculators[e] = CostToGoCalculator(env, OBS_RADIUS, dtype="float32", pad_cost_to_go=True, clamp_value=1.0)
                    self.cache_rebuilds += self.agents
                else:
                    changed = np.any(self.goals[e] != block[:, 4:6], axis=1) | (block[:, 7] != 0)
                    for a in np.flatnonzero(changed):
                        self.calculators[e].dist_tables[a] = DistTable(self.grids[e] == 0, tuple(map(int, block[a, 4:6])))
                        self.cache_rebuilds += 1
                self.goals[e] = block[:, 4:6].copy()
                refs.append(build_magat_reference_batch_from_env(env, block, cost_to_go_calculator=self.calculators[e]))
            count = len(rows)
            arrays = dict(x=np.concatenate([r.x for r in refs]),
                          edge_index=np.concatenate([r.edge_index + e*self.agents for e,r in enumerate(refs)], axis=1),
                          edge_attr=np.concatenate([r.edge_attr for r in refs]),
                          y=rows[:, 6].astype(np.int64),
                          batch=np.repeat(np.arange(len(blocks), dtype=np.int64), self.agents),
                          ptr=np.arange(0, count+1, self.agents, dtype=np.int64),
                          arrived=np.all(rows[:, 2:4] == rows[:, 4:6], axis=1))
        return HostInputs(self.backend, arrays, self.cache_rebuilds)


def _prepare_worker(backend, grids, agents, tasks, results, ready, capture_hashes):
    try:
        from expert.expert_running import configure_cpu_thread_limits
        configure_cpu_thread_limits()
        builder = CachedHostBuilder(backend, grids, agents)
        ready.set()
        while True:
            task = tasks.get()
            if task is None:
                return
            seq, rows = task
            prepared = builder.build(rows)
            prepared.stage_sequence = seq
            if capture_hashes:
                import hashlib
                prepared.stage_sha256 = hashlib.sha256(rows.tobytes()).hexdigest()
            results.put((seq, prepared, None))
    except BaseException:
        results.put((-1, None, traceback.format_exc()))


class HostPrefetch:
    """At most depth submitted stages, ordered regardless of completion order."""
    def __init__(self, backend, grids, agents, *, workers=1, depth=2, timeout=120, capture_hashes=False):
        if workers < 1 or depth < workers or timeout <= 0:
            raise ValueError("require 1 <= workers <= depth and positive timeout")
        ctx = mp.get_context("spawn")
        self.tasks = [ctx.Queue(maxsize=depth) for _ in range(workers)]
        self.results = ctx.Queue(maxsize=depth)
        self.ready = [ctx.Event() for _ in self.tasks]
        self.processes = [ctx.Process(target=_prepare_worker, args=(backend, grids, agents, q, self.results, ready, capture_hashes), daemon=True) for q, ready in zip(self.tasks, self.ready)]
        self.depth, self.timeout = depth, timeout
        self.submitted = self.consumed = 0
        self.pending = {}
        for p in self.processes:
            p.start()
        deadline = time.monotonic() + timeout
        try:
            while not all(e.is_set() for e in self.ready):
                if time.monotonic() > deadline or any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("host prefetch initialization failed")
                time.sleep(.01)
        except BaseException:
            self.close()
            raise

    def submit(self, rows):
        if self.submitted - self.consumed >= self.depth:
            raise RuntimeError("prefetch admission bound exceeded")
        seq = self.submitted
        # Own the source bytes even if the caller immediately releases its ring slot.
        self.tasks[seq % len(self.tasks)].put((seq, np.array(rows, copy=True)), timeout=self.timeout)
        self.submitted += 1

    def get(self):
        if self.consumed >= self.submitted:
            raise RuntimeError("no submitted stage")
        deadline = time.monotonic() + self.timeout
        while self.consumed not in self.pending:
            try:
                seq, result, error = self.results.get(timeout=min(0.1, max(0.001, deadline-time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline or any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("host prefetch worker exited or stalled")
                continue
            if error:
                raise RuntimeError(f"host input preparation failed:\n{error}")
            if seq in self.pending or not self.consumed <= seq < self.submitted:
                raise RuntimeError("duplicate or out-of-range prepared stage")
            self.pending[seq] = result
        result = self.pending.pop(self.consumed)
        self.consumed += 1
        return result

    def try_get(self):
        """Take the next ready stage without waiting, for optional H2D lookahead."""
        if self.consumed >= self.submitted:
            return None
        while True:
            try:
                seq, result, error = self.results.get_nowait()
            except queue.Empty:
                break
            if error:
                raise RuntimeError(f"host input preparation failed:\n{error}")
            if seq in self.pending or not self.consumed <= seq < self.submitted:
                raise RuntimeError("duplicate or out-of-range prepared stage")
            self.pending[seq] = result
        if self.consumed not in self.pending:
            return None
        result = self.pending.pop(self.consumed)
        self.consumed += 1
        return result

    def close(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
        for q in [*self.tasks, self.results]:
            q.cancel_join_thread()
            q.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ResourceMonitor:
    """Sample parent + all child RSS/CPU; summed RSS may double-count shared pages."""
    def __init__(self, interval=.25):
        import psutil
        import threading
        self.parent=psutil.Process()
        self.interval=interval
        self.peak_rss=0
        self.cpu_start={}
        self.cpu_last={}
        self._stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)

    def _sample(self):
        import psutil
        total=0
        for p in [self.parent,*self.parent.children(recursive=True)]:
            try:
                total+=p.memory_info().rss
                t=p.cpu_times(); value=t.user+t.system
                self.cpu_start.setdefault(p.pid,value); self.cpu_last[p.pid]=value
            except psutil.Error:
                continue
        self.peak_rss=max(self.peak_rss,total)

    def _run(self):
        while not self._stop.wait(self.interval): self._sample()

    def start(self):
        self._sample(); self.thread.start()

    def finish(self):
        self._stop.set(); self.thread.join(); self._sample()
        return dict(sampled_peak_process_tree_rss_bytes=self.peak_rss,
                    process_tree_cpu_s=sum(v-self.cpu_start[k] for k,v in self.cpu_last.items()),
                    resource_sample_interval_s=self.interval,
                    memory_scope='sum of parent and child RSS, including workers; shared pages may be counted repeatedly')


def memory_estimate(backend, grids, agents, ring_depth, workers, depth):
    import psutil
    n=len(grids); b=n*agents
    # Dense directed graph is a safe upper bound for the host-ready arrays.
    ready=b*4*13*13*4+n*agents*agents*(2*8+3*4)+b*25+(n+1)*8 if backend=='magat' else b*(256*4+8+1)
    cache=n*agents*grids.shape[1]*grids.shape[2]*8
    raw=b*(8 if backend=='magat' else 13)*2*ring_depth
    upper=ready*(depth+workers+2)+cache*max(workers,1)+raw
    available=psutil.virtual_memory().available
    if upper>available*.8: raise MemoryError('estimated host input buffers exceed 80% of available RAM')
    return dict(ready_stage_upper_bytes=ready,cache_array_upper_per_worker_bytes=cache,
                host_input_array_upper_bytes=upper,available_host_ram_bytes=available,
                estimate_scope='array storage bound; excludes Python/container overhead and model/runtime memory')
