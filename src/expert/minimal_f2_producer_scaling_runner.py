"""F2 producer-process scaling with a fixed logical training workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any, Callable, Sequence


NUM_EXPERTS = 8
ENVS_PER_OPTIMIZER_UPDATE = 4
MAP_NAMES = (
    "mazes-s0_wc8_od55",
    "mazes-s1_wc3_od70",
    "mazes-s2_wc8_od25",
    "mazes-s3_wc6_od35",
    "mazes-s4_wc3_od45",
    "mazes-s5_wc6_od45",
    "mazes-s27_wc7_od60",
    "mazes-s28_wc2_od35",
)
PRODUCER_COUNTS = (1, 2, 4, 8)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_f2_scaling(
    *,
    num_producers: int,
    num_agents: int,
    seed: int,
    frontier_steps: int,
    output_dir: Path,
    ring_capacity_steps: int = 128,
    device: str = "cuda:0",
    capture_stage_hashes: bool = False,
    experiment_label: str = "F2",
    envs_per_optimizer_update: int = ENVS_PER_OPTIMIZER_UPDATE,
    map_names: Sequence[str] = MAP_NAMES,
    transfer_mode: str = "async",
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if num_producers not in PRODUCER_COUNTS:
        raise ValueError(f"num_producers must be one of {PRODUCER_COUNTS}")
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")
    if frontier_steps <= 0:
        raise ValueError("frontier_steps must be positive")
    if ring_capacity_steps <= 0:
        raise ValueError("ring_capacity_steps must be positive")
    if not experiment_label:
        raise ValueError("experiment_label must be non-empty")
    envs_per_optimizer_update = int(envs_per_optimizer_update)
    if envs_per_optimizer_update <= 0:
        raise ValueError("envs_per_optimizer_update must be positive")
    if NUM_EXPERTS % envs_per_optimizer_update != 0:
        raise ValueError(
            "envs_per_optimizer_update must divide the eight logical experts"
        )
    map_names = tuple(str(name) for name in map_names)
    if len(map_names) != NUM_EXPERTS:
        raise ValueError(f"map_names must contain exactly {NUM_EXPERTS} entries")
    transfer_mode = str(transfer_mode).strip().lower()
    if transfer_mode not in {"async", "sync"}:
        raise ValueError("transfer_mode must be either 'async' or 'sync'")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"F2 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    training_log = output_dir / "train.log"
    train_batch_size = int(num_agents) * envs_per_optimizer_update
    full_frontier_rows = int(num_agents) * NUM_EXPERTS
    optimizer_steps_per_frontier = full_frontier_rows // train_batch_size
    expected_optimizer_steps = frontier_steps * optimizer_steps_per_frontier
    expected_samples = frontier_steps * full_frontier_rows
    protocol = {
        "experiment": str(experiment_label),
        "num_agents": int(num_agents),
        "num_logical_experts": NUM_EXPERTS,
        "num_producer_processes": int(num_producers),
        "frontier_rows": full_frontier_rows,
        "train_batch_size": train_batch_size,
        "envs_per_optimizer_update": envs_per_optimizer_update,
        "optimizer_steps_per_frontier": optimizer_steps_per_frontier,
        "frontier_steps": int(frontier_steps),
        "expected_optimizer_steps": expected_optimizer_steps,
        "map_names": list(map_names),
        "seed": int(seed),
        "max_episode_steps": 256,
        "ring_capacity_steps": int(ring_capacity_steps),
        "capture_stage_hashes": bool(capture_stage_hashes),
        "transfer_mode": transfer_mode,
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": None,
        "pyg_builder_mode": "local_gather",
        "pyg_local_gather_impl": "auto",
        "train_on_arrived_agents": True,
    }
    kwargs = {
        "num_steps": int(frontier_steps),
        "num_agents": int(num_agents),
        "num_experts": NUM_EXPERTS,
        "num_producer_processes": int(num_producers),
        "batch_threshold": full_frontier_rows,
        "train_batch_size": train_batch_size,
        "seed": int(seed),
        "device": device,
        "max_episode_steps": 256,
        "maps_path": "maps/maps.yaml",
        "map_names": map_names,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": expected_optimizer_steps + 1,
        "top_k_checkpoints": 1,
        "checkpoint_selection_mode": "latest",
        "validation_trajectories": None,
        "async_ring_buffer_steps": int(ring_capacity_steps),
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": None,
        "pyg_builder_mode": "local_gather",
        "pyg_local_gather_impl": "auto",
        "train_on_arrived_agents": True,
        "capture_stage_hashes": bool(capture_stage_hashes),
        "transfer_mode": transfer_mode,
        "health_sample_interval_s": 1_000_000_000.0,
        "inline_health_sampling": False,
        "log_file": str(training_log),
        "loss_log_interval": expected_optimizer_steps + 1,
    }

    peak_gpu = 0
    if engine is None:
        # Build the shared LaCAM library once in the parent before producers
        # start, then initialize CUDA outside the timed engine region.
        from real_expert_alg.lacam import inference as _lacam_inference  # noqa: F401
        from mapf_cuda.training.topology_async import (
            benchmark_topology_async_training_system,
        )
        import torch

        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        metrics = dict(benchmark_topology_async_training_system(**kwargs))
        torch.cuda.synchronize(torch.device(device))
        peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
    else:
        metrics = dict(engine(**kwargs))

    observed_optimizer_steps = int(metrics.get("num_optimizer_steps", 0))
    observed_samples = int(metrics.get("samples_processed", 0))
    worker_pids = [int(pid) for pid in metrics.get("worker_pids", [])]
    worker_exitcodes = list(metrics.get("worker_exitcodes", []))
    complete = (
        observed_optimizer_steps == expected_optimizer_steps
        and observed_samples == expected_samples
        and len(worker_pids) == num_producers
        and all(code == 0 for code in worker_exitcodes)
    )
    wall_s = float(metrics["total_wall_s"])
    result = {
        **metrics,
        "schema_version": 1,
        "status": "ok" if complete else "incomplete",
        "accepted": bool(complete),
        "experiment": str(experiment_label),
        "num_agents": int(num_agents),
        "seed": int(seed),
        "frontier_steps": int(frontier_steps),
        "num_producer_processes": int(num_producers),
        "num_logical_experts": NUM_EXPERTS,
        "frontier_rows": full_frontier_rows,
        "train_batch_size": train_batch_size,
        "envs_per_optimizer_update": envs_per_optimizer_update,
        "transfer_mode": transfer_mode,
        "optimizer_steps_per_frontier": optimizer_steps_per_frontier,
        "expected_optimizer_steps": expected_optimizer_steps,
        "expected_samples": expected_samples,
        "optimizer_steps_s": observed_optimizer_steps / wall_s,
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_host_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "validation_in_timed_region": False,
        "checkpoint_in_timed_region": False,
        "monitoring_in_timed_region": False,
        "environment_note": (
            "NVML unavailable because loaded NVIDIA kernel module 595.71.05 "
            "differs from userspace 595.84; CUDA compute remained operational "
            "and NVML was excluded from the timed region."
        ),
        "timing_scope": (
            "steady_training_pipeline_excludes_initialization_validation_"
            "checkpoint_logging_health_sampling_and_teardown"
        ),
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not training_log.exists():
        training_log.write_text(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "num_producer_processes",
                        "frontier_steps",
                        "num_optimizer_steps",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                    )
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-producers", type=int, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frontier-steps", type=int, required=True)
    parser.add_argument("--ring-capacity-steps", type=int, default=128)
    parser.add_argument("--capture-stage-hashes", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_f2_scaling(
            num_producers=args.num_producers,
            num_agents=args.num_agents,
            seed=args.seed,
            frontier_steps=args.frontier_steps,
            ring_capacity_steps=args.ring_capacity_steps,
            capture_stage_hashes=args.capture_stage_hashes,
            device=args.device,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "num_producer_processes",
                        "frontier_steps",
                        "num_optimizer_steps",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                        "optimizer_steps_s",
                        "consumer_wait_s",
                    )
                },
                sort_keys=True,
            )
        )
        return 0 if result["status"] == "ok" else 2
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "experiment": "F2",
            "num_producer_processes": args.num_producers,
            "num_agents": args.num_agents,
            "seed": args.seed,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MAP_NAMES",
    "PRODUCER_COUNTS",
    "run_f2_scaling",
]
