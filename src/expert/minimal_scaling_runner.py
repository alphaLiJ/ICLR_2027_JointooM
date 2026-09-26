"""Minimal simulator-scaling runner.

This path intentionally measures throughput only.  It does not perform the
canonical CPU replay, backend round-trip hashes, snapshots, parity checks,
health sampling, or formal archive validation used by the A2 runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np

from expert.benchmark_contract import FrozenTransitionBatch, load_frozen_transition_batch


class GpuMemoryTracker:
    """Sample process GPU memory outside the measured transition region."""

    def __init__(self, backend: str, *, pid: int | None = None):
        if backend not in {"cuda", "jax"}:
            raise ValueError(f"GPU memory tracking is unsupported for {backend!r}")
        self.backend = backend
        self.pid = int(os.getpid() if pid is None else pid)
        self.samples: list[dict[str, Any]] = []
        self._torch = None

    def start(self) -> None:
        if self.backend == "cuda":
            import torch

            self._torch = torch
            # PyTorch 2.11 may reject reset_peak_memory_stats before the CUDA
            # primary context exists (notably on Blackwell).  Initialize the
            # context explicitly; this remains outside the measured loop.
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(0)
        self.sample("before_adapter")

    def _nvml_process_memory_bytes(self) -> int | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        used_mib = 0
        found = False
        for line in completed.stdout.splitlines():
            pid_text, separator, memory_text = line.partition(",")
            if not separator:
                continue
            try:
                row_pid = int(pid_text.strip())
                row_memory_mib = int(memory_text.strip())
            except ValueError:
                continue
            if row_pid == self.pid:
                found = True
                used_mib += row_memory_mib
        return used_mib * 1024 * 1024 if found else None

    def sample(self, phase: str) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "phase": str(phase),
            "nvml_process_memory_bytes": self._nvml_process_memory_bytes(),
        }
        if self._torch is not None:
            sample.update(
                torch_allocated_bytes=int(self._torch.cuda.memory_allocated(0)),
                torch_reserved_bytes=int(self._torch.cuda.memory_reserved(0)),
                torch_peak_allocated_bytes=int(
                    self._torch.cuda.max_memory_allocated(0)
                ),
                torch_peak_reserved_bytes=int(
                    self._torch.cuda.max_memory_reserved(0)
                ),
            )
        self.samples.append(sample)
        return sample

    def summary(self) -> dict[str, Any]:
        keys = (
            "nvml_process_memory_bytes",
            "torch_allocated_bytes",
            "torch_reserved_bytes",
            "torch_peak_allocated_bytes",
            "torch_peak_reserved_bytes",
        )
        result: dict[str, Any] = {
            "backend": self.backend,
            "pid": self.pid,
            "samples": list(self.samples),
        }
        for key in keys:
            values = [
                int(sample[key])
                for sample in self.samples
                if sample.get(key) is not None
            ]
            result[f"max_{key}"] = max(values) if values else None
        return result


def load_scan_cell_with_source(manifest_path: Path, scan_cell: str):
    """Resolve a frozen workload and retain its complete source pool."""

    resolved_manifest = manifest_path.expanduser().resolve()
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    cells = [cell for cell in manifest["a2_cells"] if cell["cell_id"] == scan_cell]
    if len(cells) != 1:
        raise ValueError(f"scan cell must resolve exactly once: {scan_cell}")
    cell = cells[0]
    entries = [
        entry
        for entry in manifest["entries"]
        if entry.get("pool_name") == cell.get("pool_name")
        and entry.get("role") == "a2_pool"
    ]
    if len(entries) != 1:
        raise ValueError(f"pool must resolve exactly once: {cell.get('pool_name')}")
    batch_path = Path(entries[0]["frozen_batch_path"]).expanduser()
    if not batch_path.is_absolute():
        batch_path = resolved_manifest.parent / batch_path
    batch = load_frozen_transition_batch(batch_path)
    return cell, batch, batch.take_envs(int(cell["env_count"]))


def load_scan_cell(manifest_path: Path, scan_cell: str):
    """Resolve one frozen workload without invoking the A2 acceptance stack."""

    cell, _, selected = load_scan_cell_with_source(manifest_path, scan_cell)
    return cell, selected


def _slice_batch(batch: FrozenTransitionBatch, indices: np.ndarray):
    indices = np.asarray(indices, dtype=np.int64)
    return FrozenTransitionBatch(
        instance_ids=batch.instance_ids[indices],
        grids=batch.grids[indices],
        positions=batch.positions[indices],
        goals=batch.goals[indices],
        arrived=batch.arrived[indices],
        active=batch.active[indices],
        actions=batch.actions[:, indices],
        horizon=batch.horizon,
    )


def _run_adapter_trajectory(adapter, batch, perf_counter=time.perf_counter) -> float:
    adapter.reset_device_state()
    prepared = adapter.prepare_actions(batch.actions)
    adapter.synchronize_transition()
    started = perf_counter()
    for handle in prepared:
        adapter.step_transition(handle)
    adapter.synchronize_transition()
    elapsed = perf_counter() - started
    if elapsed <= 0:
        raise RuntimeError("measured trajectory duration must be positive")
    return elapsed


def run_single_backend(
    *,
    backend: str,
    batch,
    warmup_trajectories: int,
    repetitions: int,
    adapter_factory: Callable[..., Any] | None = None,
    perf_counter: Callable[[], float] = time.perf_counter,
    memory_tracker: GpuMemoryTracker | None = None,
) -> list[dict[str, Any]]:
    if backend not in {"cuda", "jax", "pogema"}:
        raise ValueError(f"unsupported single-process backend: {backend}")
    if warmup_trajectories < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    if adapter_factory is None:
        from expert.transition_adapters import make_transition_adapter

        adapter_factory = make_transition_adapter
    if memory_tracker is not None:
        memory_tracker.start()
    device = "cuda:0" if backend in {"cuda", "jax"} else "cpu"
    adapter = adapter_factory(
        backend,
        batch,
        device,
        verify_consumption=False,
    )
    if memory_tracker is not None:
        memory_tracker.sample("after_adapter")
    for _ in range(int(warmup_trajectories)):
        _run_adapter_trajectory(adapter, batch, perf_counter)
    if memory_tracker is not None:
        memory_tracker.sample("after_warmup")

    nominal_env_steps = int(batch.num_envs * batch.horizon)
    nominal_agent_steps = int(nominal_env_steps * batch.num_agents)
    rows = []
    for repetition in range(int(repetitions)):
        elapsed = _run_adapter_trajectory(adapter, batch, perf_counter)
        rows.append(
            {
                "repetition": repetition,
                "measured_duration_s": elapsed,
                "nominal_env_steps": nominal_env_steps,
                "nominal_agent_steps": nominal_agent_steps,
                "env_steps_s": nominal_env_steps / elapsed,
                "agent_steps_s": nominal_agent_steps / elapsed,
            }
        )
        if memory_tracker is not None:
            memory_tracker.sample(f"after_repetition_{repetition}")
    return rows


def _receive_phase(connections, phase: str, repetition: int | None, timeout_s: float):
    from multiprocessing.connection import wait

    pending = set(connections)
    deadline = time.monotonic() + float(timeout_s)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for POGEMA workers: phase={phase}")
        ready = wait(pending, timeout=remaining)
        for connection in ready:
            message = connection.recv()
            if message[0] == "ERROR":
                raise RuntimeError(f"POGEMA worker failed: {message[2]}")
            if message[0] != phase:
                raise RuntimeError(f"expected {phase}, received {message!r}")
            if repetition is not None and int(message[2]) != int(repetition):
                raise RuntimeError(
                    f"stale POGEMA worker message: expected={repetition}, got={message!r}"
                )
            pending.remove(connection)


def run_pogema_multiprocess(
    *,
    batch,
    num_workers: int,
    warmup_trajectories: int,
    repetitions: int,
    worker_timeout_s: float = 600.0,
    perf_counter: Callable[[], float] = time.perf_counter,
) -> list[dict[str, Any]]:
    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    if warmup_trajectories < 0 or repetitions <= 0:
        raise ValueError("warmup must be non-negative and repetitions positive")
    actual_workers = min(int(num_workers), int(batch.num_envs))
    splits = [
        indices
        for indices in np.array_split(np.arange(batch.num_envs), actual_workers)
        if len(indices)
    ]
    worker_batches = [
        _slice_batch(batch, indices) for indices in splits
    ]

    from expert.minimal_scaling_worker import pogema_throughput_worker

    context = mp.get_context("spawn")
    connections = []
    processes = []
    try:
        for worker_id, worker_batch in enumerate(worker_batches):
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=pogema_throughput_worker,
                args=(child, worker_id, worker_batch),
                name=f"minimal-pogema-{worker_id}",
            )
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        _receive_phase(connections, "BOOT", None, worker_timeout_s)

        def execute(repetition: int) -> float:
            for connection in connections:
                connection.send(("RESET", repetition))
            _receive_phase(connections, "READY", repetition, worker_timeout_s)
            started = perf_counter()
            for connection in connections:
                connection.send(("GO", repetition))
            _receive_phase(connections, "DONE", repetition, worker_timeout_s)
            elapsed = perf_counter() - started
            if elapsed <= 0:
                raise RuntimeError("measured POGEMA wall duration must be positive")
            return elapsed

        for warmup in range(int(warmup_trajectories)):
            execute(-(warmup + 1))

        nominal_env_steps = int(batch.num_envs * batch.horizon)
        nominal_agent_steps = int(nominal_env_steps * batch.num_agents)
        rows = []
        for repetition in range(int(repetitions)):
            elapsed = execute(repetition)
            rows.append(
                {
                    "repetition": repetition,
                    "measured_duration_s": elapsed,
                    "nominal_env_steps": nominal_env_steps,
                    "nominal_agent_steps": nominal_agent_steps,
                    "env_steps_s": nominal_env_steps / elapsed,
                    "agent_steps_s": nominal_agent_steps / elapsed,
                }
            )
        return rows
    finally:
        for connection in connections:
            try:
                connection.send(("STOP",))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in connections:
            connection.close()


def _write_results(
    output_dir: Path,
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    values = [float(row["env_steps_s"]) for row in rows]
    result = {
        "backend": config["backend"],
        "num_envs": config["num_envs"],
        "num_agents": config["num_agents"],
        "horizon": config["horizon"],
        "repetitions": len(rows),
        "median_env_steps_s": statistics.median(values),
        "min_env_steps_s": min(values),
        "max_env_steps_s": max(values),
        "timed_region": "fixed_trajectory_transition_only",
        "correctness_validation": False,
        "memory": memory,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "stdout.log").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_failure(
    output_dir: Path,
    *,
    config: dict[str, Any],
    error: Exception,
    memory: dict[str, Any] | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    message = str(error)
    lowered = message.lower()
    capacity_markers = (
        "out of memory",
        "cudaerrormemoryallocation",
        "memory allocation failed",
        "failed to allocate",
    )
    capacity_failure = any(marker in lowered for marker in capacity_markers)
    result = {
        "backend": config["backend"],
        "num_envs": config["num_envs"],
        "num_agents": config["num_agents"],
        "horizon": config["horizon"],
        "status": "capacity_failure" if capacity_failure else "failed",
        "exception_type": type(error).__name__,
        "exception_message": message,
        "memory": memory,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "stdout.log").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal MAPF throughput row.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("cuda", "jax", "pogema", "pogema_mp"),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-trajectories", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    cell, batch = load_scan_cell(args.manifest, args.scan_cell)
    config = {
        "backend": args.backend,
        "scan_cell": args.scan_cell,
        "cell": cell,
        "manifest": str(args.manifest.expanduser().resolve()),
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
        "horizon": batch.horizon,
        "num_workers": (
            min(args.num_workers, batch.num_envs)
            if args.backend == "pogema_mp"
            else 1
        ),
        "warmup_trajectories": args.warmup_trajectories,
        "repetitions": args.repetitions,
        "correctness_validation": False,
        "counting": "nominal_env_slots=num_envs*horizon",
    }
    memory_tracker = (
        GpuMemoryTracker(args.backend)
        if args.backend in {"cuda", "jax"}
        else None
    )
    try:
        if args.backend == "pogema_mp":
            rows = run_pogema_multiprocess(
                batch=batch,
                num_workers=args.num_workers,
                warmup_trajectories=args.warmup_trajectories,
                repetitions=args.repetitions,
                worker_timeout_s=args.worker_timeout_s,
            )
        else:
            rows = run_single_backend(
                backend=args.backend,
                batch=batch,
                warmup_trajectories=args.warmup_trajectories,
                repetitions=args.repetitions,
                memory_tracker=memory_tracker,
            )
    except Exception as error:
        memory = memory_tracker.summary() if memory_tracker is not None else None
        result = _write_failure(
            args.output_dir,
            config=config,
            error=error,
            memory=memory,
        )
        print(json.dumps(result, sort_keys=True))
        return 2
    memory = memory_tracker.summary() if memory_tracker is not None else None
    result = _write_results(
        args.output_dir,
        config=config,
        rows=rows,
        memory=memory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "GpuMemoryTracker",
    "load_scan_cell",
    "run_pogema_multiprocess",
    "run_single_backend",
]
