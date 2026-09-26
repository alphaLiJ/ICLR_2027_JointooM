"""Run one fresh-process row of the S2 matched-wall multi-seed experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


MAP_NAMES = (
    "mazes-s0_wc8_od55",
    "mazes-s1_wc3_od70",
    "mazes-s2_wc8_od25",
    "mazes-s3_wc6_od35",
)
METHODS = ("proposed_async", "conventional_sync")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _source_manifest(repo: Path) -> list[dict[str, str]]:
    relative_paths = (
        "experiments/runners/run_s2_multiseed_magat_training.py",
        "src/expert/checkpoint_manager.py",
        "src/expert/training_baseline_engines.py",
        "src/mapf_cuda/training/topology_async.py",
        "src/expert/fixed_magat_plus_runtime.py",
    )
    return [
        {"path": relative, "sha256": sha256_file(repo / relative)}
        for relative in relative_paths
    ]


def build_protocol(
    *,
    method: str,
    seed: int,
    validation_trajectories: Sequence[Path],
    wall_budget_s: float | None,
    num_steps: int,
    checkpoint_interval_s: float | None,
    capture_stage_hashes: bool,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    repo = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "experiment": "S2-matched-wall-MAGAT-multiseed",
        "method": method,
        "seed": int(seed),
        "num_agents": 512,
        "num_experts": 4,
        "map_names": list(MAP_NAMES),
        "num_steps_cap": int(num_steps),
        "max_episode_steps": 256,
        "batch_threshold": 2048,
        "train_batch_size": 2048,
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": "cosine-annealing",
        "scheduler_total_steps": 100_000,
        "grad_clip_norm": None,
        "train_on_arrived_agents": True,
        "wall_budget_s": wall_budget_s,
        "checkpoint_interval_s": checkpoint_interval_s,
        "checkpoint_interval_steps_fallback": 100_001,
        "checkpoint_selection_mode": (
            "validation_accuracy" if validation_trajectories else "latest"
        ),
        "capture_stage_hashes": bool(capture_stage_hashes),
        "timing_boundary": (
            "starts_after_worker_cuda_model_validation_initialization; "
            "includes_periodic_validation_and_checkpointing; "
            "excludes_final_forced_validation_checkpoint_and_teardown"
        ),
        "nvml_sampling": False,
        "validation_trajectories": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in validation_trajectories
        ],
        "git_commit": _git_commit(repo),
        "source_manifest": _source_manifest(repo),
    }


def run_row(
    *,
    method: str,
    seed: int,
    output_dir: Path,
    validation_trajectories: Sequence[Path] = (),
    wall_budget_s: float | None = None,
    num_steps: int = 100_000,
    checkpoint_interval_s: float | None = 300.0,
    capture_stage_hashes: bool = False,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if wall_budget_s is not None and wall_budget_s <= 0:
        raise ValueError("wall_budget_s must be positive")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    validation = [path.expanduser().resolve() for path in validation_trajectories]
    if any(not path.is_file() for path in validation):
        raise FileNotFoundError("all validation trajectories must exist")

    output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoints" if validation else None
    protocol = build_protocol(
        method=method,
        seed=seed,
        validation_trajectories=validation,
        wall_budget_s=wall_budget_s,
        num_steps=num_steps,
        checkpoint_interval_s=checkpoint_interval_s,
        capture_stage_hashes=capture_stage_hashes,
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Build LaCAM before spawning producers so workers never race in CMake.
    from real_expert_alg.lacam import inference as _lacam_inference  # noqa: F401

    common = dict(
        num_steps=int(num_steps),
        num_agents=512,
        num_experts=4,
        batch_threshold=2048,
        train_batch_size=2048,
        seed=int(seed),
        device=device,
        max_episode_steps=256,
        maps_path="maps/maps.yaml",
        map_names=MAP_NAMES,
        checkpoint_dir=(None if checkpoint_dir is None else str(checkpoint_dir)),
        checkpoint_interval=100_001,
        checkpoint_interval_s=checkpoint_interval_s,
        top_k_checkpoints=3,
        checkpoint_selection_mode=(
            "validation_accuracy" if validation else "latest"
        ),
        validation_trajectories=[str(path) for path in validation],
        lr_start=1e-3,
        lr_end=1e-6,
        lr_scheduler="cosine-annealing",
        scheduler_total_steps=100_000,
        grad_clip_norm=None,
        train_on_arrived_agents=True,
        health_sample_interval_s=1_000_000_000.0,
        inline_health_sampling=False,
        max_training_wall_s=wall_budget_s,
        capture_stage_hashes=bool(capture_stage_hashes),
    )
    if method == "proposed_async":
        from mapf_cuda.training.topology_async import (
            benchmark_topology_async_training_system,
        )

        metrics = benchmark_topology_async_training_system(
            **common,
            pyg_builder_mode="local_gather",
            pyg_local_gather_impl="auto",
            async_ring_buffer_steps=128,
            transfer_mode="async",
            num_producer_processes=4,
            log_file=str(output_dir / "engine.log"),
            loss_log_interval=1000,
        )
    else:
        from expert.training_baseline_engines import run_magat_strong_online_engine

        metrics = run_magat_strong_online_engine(
            **common,
            ring_buffer_steps=1,
            system_mode="s2_multiseed_conventional_sync",
        )

    completed_steps = int(metrics.get("num_optimizer_steps", 0))
    completed = bool(metrics.get("wall_budget_reached")) or completed_steps == int(
        num_steps
    )
    result = {
        **metrics,
        "schema_version": 1,
        "experiment": "S2-matched-wall-MAGAT-multiseed",
        "method": method,
        "seed": int(seed),
        "status": "ok" if completed else "incomplete",
        "protocol": protocol,
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-trajectories", default="")
    parser.add_argument("--wall-budget-s", type=float)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--checkpoint-interval-s", type=float, default=300.0)
    parser.add_argument("--capture-stage-hashes", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    result = run_row(
        method=args.method,
        seed=args.seed,
        output_dir=args.output_dir,
        validation_trajectories=[
            Path(value)
            for value in args.validation_trajectories.split(",")
            if value
        ],
        wall_budget_s=args.wall_budget_s,
        num_steps=args.num_steps,
        checkpoint_interval_s=args.checkpoint_interval_s,
        capture_stage_hashes=args.capture_stage_hashes,
        device=args.device,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "method",
                    "seed",
                    "initial_model_state_sha256",
                    "num_optimizer_steps",
                    "samples_processed",
                    "total_wall_s",
                    "samples_s",
                    "validation_metrics",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
