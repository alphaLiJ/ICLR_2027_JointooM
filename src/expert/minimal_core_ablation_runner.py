"""Run one clean MAGAT component-ablation cell.

The matrix crosses representation/materialization with producer-ring depth:

* S0: host-expanded/full-transfer, one-stage backpressure;
* S1: compact transfer + CUDA replay, one-stage backpressure;
* S2: host-expanded/full-transfer, deep producer ring;
* S3: compact transfer + CUDA replay, deep producer ring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any, Callable, Sequence


CELLS = {
    "s0_full_one_stage": ("full_host_materialized", 1),
    "s1_compact_one_stage": ("compact_cuda_replay", 1),
    "s2_full_deep_ring": ("full_host_materialized", 128),
    "s3_compact_deep_ring": ("compact_cuda_replay", 128),
}
DEFAULT_MAP_NAMES = (
    "mazes-s0_wc8_od55",
    "mazes-s1_wc3_od70",
    "mazes-s27_wc7_od60",
    "mazes-s28_wc2_od35",
)
ADVISORY_HEALTH_CODES = frozenset({"collector_error", "gpu_metrics_missing"})


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _health_disposition(metrics: dict[str, Any]) -> str:
    validation = metrics.get("health_report", {}).get("validation", {})
    if bool(validation.get("valid")):
        return "pass"
    codes = {
        str(item.get("code"))
        for item in validation.get("violations", [])
        if isinstance(item, dict)
    }
    if codes and codes <= ADVISORY_HEALTH_CODES:
        return "telemetry_unavailable"
    return "fail"


def _engine_for_representation(
    representation: str,
) -> Callable[..., dict[str, Any]]:
    if representation == "full_host_materialized":
        from expert.training_baseline_engines import run_magat_strong_online_engine

        return run_magat_strong_online_engine
    if representation == "compact_cuda_replay":
        from expert.training_baseline_engines import (
            run_magat_compact_sync_matched_engine,
        )

        return run_magat_compact_sync_matched_engine
    raise ValueError(f"unsupported representation: {representation}")


def _phase_metrics(
    representation: str, metrics: dict[str, Any]
) -> dict[str, float]:
    if representation == "full_host_materialized":
        return {
            "consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
            "host_materialization_and_h2d_s": float(
                metrics.get("consumer_host_builder_and_h2d_s", 0.0)
            ),
            "cuda_replay_s": 0.0,
            "train_s": float(metrics.get("consumer_train_step_s", 0.0)),
        }
    return {
        "consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
        "compact_h2d_s": float(metrics.get("consumer_dma_sync_s", 0.0)),
        "cuda_replay_s": float(metrics.get("consumer_gpu_builder_s", 0.0)),
        "train_s": float(metrics.get("consumer_train_step_s", 0.0)),
    }


def run_core_ablation(
    *,
    cell: str,
    seed: int,
    num_steps: int,
    output_dir: Path,
    device: str = "cuda:0",
    capture_stage_hashes: bool = False,
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if cell not in CELLS:
        raise ValueError(f"unsupported core-ablation cell: {cell}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    representation, ring_buffer_steps = CELLS[cell]
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"core-ablation output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    protocol = {
        "experiment": "CORE-ABLATION",
        "cell": cell,
        "representation": representation,
        "ring_buffer_steps": ring_buffer_steps,
        "stage_release": "after_optimizer_step",
        "num_agents": 256,
        "num_experts": 4,
        "batch_threshold": 1024,
        "train_batch_size": 1024,
        "num_steps": int(num_steps),
        "seed": int(seed),
        "map_names": list(DEFAULT_MAP_NAMES),
        "max_episode_steps": 256,
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": None,
        "pyg_builder_mode": "local_gather",
        "pyg_local_gather_impl": "auto",
        "train_on_arrived_agents": True,
        "checkpoint_interval": int(num_steps) + 1,
        "checkpoint_selection_mode": "latest",
        "validation": "disabled",
        "health_sample_interval_s": 1_000_000_000.0,
        "inline_health_sampling": False,
    }
    kwargs = {
        "num_steps": int(num_steps),
        "num_agents": 256,
        "num_experts": 4,
        "batch_threshold": 1024,
        "train_batch_size": 1024,
        "seed": int(seed),
        "device": device,
        "max_episode_steps": 256,
        "maps_path": "maps/maps.yaml",
        "map_names": DEFAULT_MAP_NAMES,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": int(num_steps) + 1,
        "top_k_checkpoints": 1,
        "checkpoint_selection_mode": "latest",
        "validation_trajectories": (),
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": None,
        "pyg_builder_mode": "local_gather",
        "pyg_local_gather_impl": "auto",
        "train_on_arrived_agents": True,
        "capture_stage_hashes": bool(capture_stage_hashes),
        "health_sample_interval_s": 1_000_000_000.0,
        "inline_health_sampling": False,
        "ring_buffer_steps": ring_buffer_steps,
    }

    # Avoid concurrent first-import builds in expert workers.
    from real_expert_alg.lacam import inference as _lacam_inference  # noqa: F401

    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch_device)
    runner = _engine_for_representation(representation) if engine is None else engine
    metrics = dict(runner(**kwargs))
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        peak_gpu = int(torch.cuda.max_memory_allocated(torch_device))
    else:
        peak_gpu = 0
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    expected_samples = int(num_steps) * 1024
    health_disposition = _health_disposition(metrics)
    accepted = (
        health_disposition in {"pass", "telemetry_unavailable"}
        and int(metrics.get("num_optimizer_steps", 0)) == int(num_steps)
        and int(metrics.get("samples_processed", 0)) == expected_samples
        and int(metrics.get("ring_buffer_steps", 0)) == ring_buffer_steps
    )
    result = {
        **metrics,
        "schema_version": 1,
        "status": "ok" if accepted else "acceptance_failed",
        "accepted": accepted,
        "health_disposition": health_disposition,
        "experiment": "CORE-ABLATION",
        "cell": cell,
        "representation": representation,
        "ring_buffer_steps": ring_buffer_steps,
        "expected_samples": expected_samples,
        "expected_optimizer_steps": int(num_steps),
        "seed": int(seed),
        "num_steps": int(num_steps),
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_host_memory_bytes": peak_host,
        "phase_metrics": _phase_metrics(representation, metrics),
        "timing_scope": (
            "steady_training_pipeline_excludes_initialization_validation_"
            "checkpoint_logging_health_sampling_and_teardown"
        ),
        "validation_in_timed_region": False,
        "monitoring_in_timed_region": False,
        "checkpoint_in_timed_region": False,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "train.log").write_text(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "cell",
                    "seed",
                    "num_optimizer_steps",
                    "samples_processed",
                    "total_wall_s",
                    "samples_s",
                    "final_loss",
                    "h2d_bytes",
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
    parser.add_argument("--cell", required=True, choices=tuple(CELLS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-steps", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-stage-hashes", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_core_ablation(
        cell=args.cell,
        seed=args.seed,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        device=args.device,
        capture_stage_hashes=args.capture_stage_hashes,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "cell",
                    "seed",
                    "num_optimizer_steps",
                    "samples_processed",
                    "total_wall_s",
                    "samples_s",
                    "h2d_bytes",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CELLS", "DEFAULT_MAP_NAMES", "run_core_ablation"]
