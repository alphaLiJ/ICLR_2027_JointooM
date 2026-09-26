"""Run one lightweight E5 MAGAT matched-training row."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any, Callable, Sequence


MODES = ("strong_online", "compact_sync", "proposed_async")
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


def _peak_nvml_bytes(metrics: dict[str, Any]) -> int | None:
    values = [
        (event.get("gpu") or {}).get("memory_used_bytes")
        for event in metrics.get("health_report", {}).get("events", [])
    ]
    observed = [int(value) for value in values if value is not None]
    return max(observed) if observed else None


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


def _phase_metrics(mode: str, metrics: dict[str, Any]) -> dict[str, float]:
    if mode == "strong_online":
        return {
            "producer_or_consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
            "dma_or_host_builder_s": float(
                metrics.get("consumer_host_builder_and_h2d_s", 0.0)
            ),
            "gpu_builder_s": 0.0,
            "trainer_s": float(metrics.get("consumer_train_step_s", 0.0)),
        }
    return {
        "producer_or_consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
        "dma_or_host_builder_s": float(metrics.get("consumer_dma_sync_s", 0.0)),
        "gpu_builder_s": float(metrics.get("consumer_gpu_builder_s", 0.0)),
        "trainer_s": float(metrics.get("consumer_train_step_s", 0.0)),
    }


def _engine_for_mode(mode: str) -> Callable[..., dict[str, Any]]:
    if mode == "strong_online":
        from expert.training_baseline_engines import run_magat_strong_online_engine

        return run_magat_strong_online_engine
    if mode == "compact_sync":
        from expert.training_baseline_engines import (
            run_magat_compact_sync_matched_engine,
        )

        return run_magat_compact_sync_matched_engine
    if mode == "proposed_async":
        from mapf_cuda.training.topology_async import (
            benchmark_topology_async_training_system,
        )

        return benchmark_topology_async_training_system
    raise ValueError(f"unsupported E5 MAGAT mode: {mode}")


def run_e5_magat(
    *,
    mode: str,
    seed: int,
    num_steps: int,
    output_dir: Path,
    validation_trajectories: Sequence[Path],
    device: str = "cuda:0",
    capture_stage_hashes: bool = False,
    engine: Callable[..., dict[str, Any]] | None = None,
    max_training_wall_s: float | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported E5 MAGAT mode: {mode}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if max_training_wall_s is not None and max_training_wall_s <= 0:
        raise ValueError("max_training_wall_s must be positive")
    validation = [str(path.expanduser().resolve()) for path in validation_trajectories]
    if not validation:
        raise ValueError("at least one frozen validation trajectory is required")
    missing = [path for path in validation if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation trajectories: {missing}")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"E5 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    training_log = output_dir / "train.log"
    protocol = {
        "backend": "magat",
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
        "checkpoint_selection_mode": (
            "latest" if max_training_wall_s is not None else "validation_accuracy"
        ),
        "validation_trajectories": validation,
        "health_sample_interval_s": 1_000_000_000.0,
        "inline_health_sampling": False,
        "timed_validation": False,
        "max_training_wall_s": max_training_wall_s,
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
        "top_k_checkpoints": 3,
        "checkpoint_selection_mode": (
            "latest" if max_training_wall_s is not None else "validation_accuracy"
        ),
        "validation_trajectories": validation,
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": None,
        "pyg_builder_mode": "local_gather",
        "pyg_local_gather_impl": "auto",
        "train_on_arrived_agents": True,
        "capture_stage_hashes": bool(capture_stage_hashes),
        "health_sample_interval_s": 1_000_000_000.0,
        "inline_health_sampling": False,
        "max_training_wall_s": max_training_wall_s,
    }
    if mode == "proposed_async":
        kwargs.update(
            {
                "async_ring_buffer_steps": 128,
                "log_file": str(training_log),
                "loss_log_interval": int(num_steps) + 1,
            }
        )

    # Import once in the parent before workers start.  The LaCAM wrapper builds
    # its shared library on first import when absent; allowing four workers to
    # perform that build concurrently corrupts the shared CMake directory.
    from real_expert_alg.lacam import inference as _lacam_inference  # noqa: F401

    import torch

    torch.cuda.set_device(torch.device(device))
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    runner = _engine_for_mode(mode) if engine is None else engine
    metrics = dict(runner(**kwargs))
    torch.cuda.synchronize(torch.device(device))
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    health_disposition = _health_disposition(metrics)
    if max_training_wall_s is None:
        training_complete = int(metrics.get("num_optimizer_steps", 0)) == int(num_steps)
    else:
        training_complete = bool(metrics.get("wall_budget_reached")) and int(
            metrics.get("num_optimizer_steps", 0)
        ) > 0
    accepted = (
        bool(metrics.get("accepted", True))
        or health_disposition == "telemetry_unavailable"
    ) and training_complete
    samples = int(
        metrics.get(
            "samples_processed",
            int(metrics.get("num_optimizer_steps", 0)) * 1024,
        )
    )
    h2d_bytes = metrics.get("h2d_bytes")
    d2h_bytes = metrics.get("d2h_bytes")
    transfer_accounting = "measured"
    if h2d_bytes is None:
        h2d_bytes = samples * 8 * 2
        d2h_bytes = int(metrics.get("num_optimizer_steps", 0)) * 4
        transfer_accounting = "payload_exact_control_excluded"

    result = {
        **metrics,
        "schema_version": 1,
        "status": "ok" if accepted else "health_failed",
        "accepted": accepted,
        "health_disposition": health_disposition,
        "experiment": "E5-MAGAT",
        "mode": mode,
        "seed": int(seed),
        "num_steps": int(num_steps),
        "max_training_wall_s": max_training_wall_s,
        "training_budget_type": (
            "optimizer_steps" if max_training_wall_s is None else "steady_training_wall"
        ),
        "samples_processed": samples,
        "optimizer_steps_s": float(metrics["num_optimizer_steps"])
        / float(metrics["total_wall_s"]),
        "h2d_bytes": int(h2d_bytes),
        "d2h_bytes": int(d2h_bytes or 0),
        "transfer_accounting": transfer_accounting,
        "timing_scope": (
            "steady_training_pipeline_excludes_initialization_validation_"
            "checkpoint_logging_health_sampling_and_teardown"
        ),
        "validation_in_timed_region": False,
        "excluded_validation_s": float(
            sum(
                float(row.get("validation_elapsed_s", 0.0))
                for row in metrics.get("checkpoint_history", [])
            )
        ),
        "peak_gpu_memory_bytes": peak_gpu,
        "peak_nvml_gpu_memory_bytes": _peak_nvml_bytes(metrics),
        "peak_host_memory_bytes": peak_host,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "phase_metrics": _phase_metrics(mode, metrics),
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not training_log.exists():
        training_log.write_text(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "mode",
                        "seed",
                        "num_steps",
                        "num_optimizer_steps",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                        "mean_loss",
                        "final_loss",
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
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--num-steps", required=True, type=int)
    parser.add_argument("--max-training-wall-s", type=float)
    parser.add_argument("--validation-trajectories", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-stage-hashes", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    validation = [
        Path(value) for value in args.validation_trajectories.split(",") if value
    ]
    result = run_e5_magat(
        mode=args.mode,
        seed=args.seed,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        validation_trajectories=validation,
        device=args.device,
        capture_stage_hashes=args.capture_stage_hashes,
        max_training_wall_s=args.max_training_wall_s,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "mode",
                    "seed",
                    "num_optimizer_steps",
                    "samples_processed",
                    "total_wall_s",
                    "samples_s",
                    "final_loss",
                    "peak_gpu_memory_bytes",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_MAP_NAMES", "MODES", "run_e5_magat"]
