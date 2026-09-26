"""Run one lightweight E5 MAPF-GPT matched-training row."""

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
            "consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
            "dma_or_host_pipeline_s": float(
                metrics.get("consumer_host_builder_and_h2d_s", 0.0)
            ),
            "cuda_pipeline_s": float(metrics.get("consumer_train_step_s", 0.0)),
        }
    if mode == "compact_sync":
        return {
            "consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
            "dma_or_host_pipeline_s": float(
                metrics.get("consumer_dma_sync_s", 0.0)
            ),
            "cuda_pipeline_s": float(metrics.get("consumer_pipeline_s", 0.0)),
        }
    return {
        "consumer_wait_s": float(metrics.get("consumer_wait_s", 0.0)),
        "dma_or_host_pipeline_s": 0.0,
        "cuda_pipeline_s": float(metrics.get("consumer_pipeline_s", 0.0)),
    }


def _engine_for_mode(mode: str) -> Callable[..., dict[str, Any]]:
    if mode == "strong_online":
        from expert.training_baseline_engines import (
            run_mapf_gpt_strong_online_engine,
        )

        return run_mapf_gpt_strong_online_engine
    if mode == "compact_sync":
        from expert.training_baseline_engines import (
            run_mapf_gpt_compact_sync_matched_engine,
        )

        return run_mapf_gpt_compact_sync_matched_engine
    if mode == "proposed_async":
        from expert.mapf_gpt_online import run_topology_mapf_gpt_online

        return run_topology_mapf_gpt_online
    raise ValueError(f"unsupported E5 MAPF-GPT mode: {mode}")


def run_e5_mapf_gpt(
    *,
    mode: str,
    seed: int,
    num_steps: int,
    output_dir: Path,
    validation_datasets: Sequence[Path],
    device: str = "cuda:0",
    capture_stage_hashes: bool = False,
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported E5 MAPF-GPT mode: {mode}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    validation = [str(path.expanduser().resolve()) for path in validation_datasets]
    if not validation:
        raise ValueError("at least one frozen validation dataset is required")
    missing = [path for path in validation if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing validation datasets: {missing}")

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"E5 MAPF-GPT output directory already exists: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output_dir / "checkpoints"
    protocol = {
        "backend": "mapf_gpt",
        "model_size": "2M",
        "num_agents": 256,
        "num_experts": 4,
        "train_batch_size": 1024,
        "microbatch_size": 256,
        "shuffle_capacity": 8192,
        "num_steps": int(num_steps),
        "seed": int(seed),
        "map_names": list(DEFAULT_MAP_NAMES),
        "max_episode_steps": 256,
        "learning_rate": 6e-4,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "grad_clip": 1.0,
        "history_length": 5,
        "train_on_arrived_agents": True,
        "checkpoint_interval": 1000,
        "checkpoint_selection_mode": "validation_accuracy",
        "validation_datasets": validation,
    }
    kwargs = {
        "num_steps": int(num_steps),
        "num_agents": 256,
        "maps_path": "maps/maps.yaml",
        "map_names": DEFAULT_MAP_NAMES,
        "model_size": "2M",
        "train_batch_size": 1024,
        "microbatch_size": 256,
        "shuffle_capacity": 8192,
        "seed": int(seed),
        "device": device,
        "max_episode_steps": 256,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_interval": 1000,
        "top_k_checkpoints": 3,
        "checkpoint_selection_mode": "validation_accuracy",
        "validation_datasets": validation,
        "validation_batch_size": 256,
        "train_on_arrived_agents": True,
        "capture_stage_hashes": bool(capture_stage_hashes),
    }
    if mode == "proposed_async":
        kwargs["ring_capacity_steps"] = 4

    # Avoid concurrent first-import builds in the four spawned workers.
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
    optimizer_steps = int(
        metrics.get(
            "num_optimizer_steps",
            metrics.get("run_optimizer_steps", 0),
        )
    )
    samples = int(
        metrics.get(
            "samples_processed",
            metrics.get("run_samples_processed", optimizer_steps * 1024),
        )
    )
    training_complete = optimizer_steps == int(num_steps)
    accepted = (
        bool(metrics.get("accepted", True))
        or health_disposition == "telemetry_unavailable"
    ) and training_complete
    wall_s = float(metrics.get("total_wall_s", metrics.get("wall_s")))
    result = {
        **metrics,
        "schema_version": 1,
        "status": "ok" if accepted else "health_failed",
        "accepted": accepted,
        "health_disposition": health_disposition,
        "experiment": "E5-MAPF-GPT",
        "mode": mode,
        "seed": int(seed),
        "num_steps": int(num_steps),
        "num_optimizer_steps": optimizer_steps,
        "samples_processed": samples,
        "total_wall_s": wall_s,
        "samples_s": float(samples / wall_s),
        "optimizer_steps_s": float(optimizer_steps / wall_s),
        "h2d_bytes": int(metrics.get("h2d_bytes", 0)),
        "d2h_bytes": int(metrics.get("d2h_bytes", 0)),
        "transfer_accounting": str(
            metrics.get(
                "transfer_accounting",
                "model_ready_payload_only_control_excluded",
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
    (output_dir / "train.log").write_text(
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-datasets", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capture-stage-hashes", action="store_true")
    args = parser.parse_args(argv)
    validation = [
        Path(path) for path in args.validation_datasets.split(",") if path
    ]
    result = run_e5_mapf_gpt(
        mode=args.mode,
        seed=args.seed,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        validation_datasets=validation,
        device=args.device,
        capture_stage_hashes=args.capture_stage_hashes,
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
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
