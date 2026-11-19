"""Hash-verified finite expert records and the existing complete-stage ring."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from mapf_cuda.training.host_inputs import validate_stage


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_records(output, *, backend, grids, stages, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stages = np.asarray(stages)
    if not np.issubdtype(stages.dtype, np.integer) or np.any(stages < 0) or np.any(stages > 65535):
        raise ValueError("records must contain uint16-representable integers")
    stages = stages.astype(np.uint16)
    features = 8 if backend == "magat" else 13
    if stages.ndim != 3 or not len(stages) or stages.shape[1] % len(grids):
        raise ValueError("invalid stage archive")
    agents = stages.shape[1] // len(grids)
    for rows in stages:
        validate_stage(rows, grids, agents, features)
    np.save(output / "grids.npy", np.asarray(grids, dtype=np.uint8), allow_pickle=False)
    np.save(output / "stages.npy", stages, allow_pickle=False)
    manifest = dict(schema_version=1, backend=backend, agents=agents, stages=len(stages),
                    features=features, metadata=metadata,
                    hashes={name: sha256_file(output/name) for name in ("grids.npy", "stages.npy")})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def load_records(root):
    root = Path(root)
    m = json.loads((root/"manifest.json").read_text())
    if m.get("schema_version") != 1 or set(m.get("hashes", {})) != {"grids.npy", "stages.npy"}:
        raise ValueError("unsupported or incomplete record manifest")
    for name, expected in m["hashes"].items():
        if sha256_file(root/name) != expected:
            raise ValueError(f"record hash mismatch: {name}")
    grids = np.load(root/"grids.npy", allow_pickle=False)
    stages = np.load(root/"stages.npy", mmap_mode="r", allow_pickle=False)
    if stages.shape != (m["stages"], len(grids)*m["agents"], m["features"]):
        raise ValueError("record shape differs from manifest")
    for rows in stages:
        validate_stage(rows, grids, m["agents"], m["features"])
    return m, grids, stages


class _WorkerGate:
    def __init__(self, ready, go, stop):
        self.ready, self.go, self.stop = ready, go, stop

    def mark_ready(self):
        self.ready.set()
        self.go.wait()

    def mark_complete_and_wait(self):
        self.stop.wait()

    def begin_expert_call(self, **kwargs):
        pass

    def finish_expert_call(self):
        pass

    def mark_published(self):
        pass


def _produce(e, ring, backend, config, records, gate):
    from expert.expert_running import configure_cpu_thread_limits
    configure_cpu_thread_limits()
    if records is not None:
        stages = np.load(Path(records)/"stages.npy", mmap_mode="r", allow_pickle=False)
        a = config["num_agents"]
        gate.mark_ready()
        for stage in stages[:config["num_steps"]]:
            ring.reserve_and_write(np.array(stage[e*a:(e+1)*a], copy=True))
        gate.mark_complete_and_wait()
    else:
        kwargs = dict(maps_path=config["maps_path"], map_name=config["map_names"][e],
                      num_agents=config["num_agents"], num_steps=config["num_steps"],
                      seed=config["seed"]+e, max_episode_steps=config["max_episode_steps"], health_handle=gate)
        if backend == "magat":
            from mapf_cuda.training.topology_async import _topology_async_training_expert_worker_loop
            _topology_async_training_expert_worker_loop(e, ring, **kwargs)
        else:
            from expert.mapf_gpt_online import mapf_gpt_expert_worker_loop
            mapf_gpt_expert_worker_loop(e, ring, **kwargs)


class RingStageSource:
    """One worker per logical environment; original expert code and ring API."""
    def __init__(self, ring, backend, config, *, records=None, timeout=120):
        self.ring, self.timeout, self.seq = ring, timeout, 0
        self.count = config["num_steps"]
        self.go, self.stop = mp.Event(), mp.Event()
        self.ready = [mp.Event() for _ in config["map_names"]]
        self.processes = [mp.Process(target=_produce, args=(e, ring, backend, config, records,
                          _WorkerGate(ready, self.go, self.stop)), daemon=True) for e, ready in enumerate(self.ready)]
        for p in self.processes:
            p.start()
        try:
            deadline = time.monotonic()+timeout
            while not all(e.is_set() for e in self.ready):
                self.check()
                if time.monotonic() > deadline:
                    raise TimeoutError("expert initialization timeout")
                time.sleep(.01)
        except BaseException:
            self.close()
            raise

    def check(self):
        if any(p.exitcode is not None for p in self.processes):
            raise RuntimeError(f"expert process exited: {[p.exitcode for p in self.processes]}")

    def start(self):
        self.go.set()

    def next_rows(self):
        if self.seq >= self.count:
            raise StopIteration
        deadline = time.monotonic()+self.timeout
        while not self.ring.is_stage_ready(self.seq):
            self.check()
            if time.monotonic() > deadline:
                raise TimeoutError("expert failed to publish a complete stage")
            time.sleep(.0005)
        lo, hi = self.ring._stage_row_bounds(self.seq)
        result = self.ring.cpu_buffer[lo:hi].numpy().astype(np.uint16, copy=True)
        self.ring.release_stage(self.seq)
        self.seq += 1
        self.ring.shared_compute_ptr.value = self.seq * self.ring.stage_rows
        return result

    def close(self):
        self.stop.set()
        self.go.set()
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
