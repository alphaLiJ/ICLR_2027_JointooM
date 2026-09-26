"""Train the conventional MAGAT+ path under the 512-agent run's wall budget.

This runner is deliberately narrow.  It reproduces the task/model/optimizer
protocol of the retained 512-agent 100k topology-async run while replacing the
training data path with the conventional synchronous path:

1. four LaCAM workers generate one environment timestep each;
2. the consumer waits for the complete four-environment stage;
3. MAGAT+ observations and PyG graphs are reconstructed on the CPU;
4. model-ready tensors are copied to CUDA and one optimizer step is executed;
5. only then may the workers publish the next stage.

The wall-clock limit begins after workers, CUDA, the model, and validation
artifacts have initialized, matching ``total_wall_s`` in the retained run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


DEFAULT_MAP_NAMES = (
    "mazes-s0_wc8_od55",
    "mazes-s1_wc3_od70",
    "mazes-s2_wc8_od25",
    "mazes-s3_wc6_od35",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_equal_wall_baseline(
    *,
    output_dir: Path,
    validation_trajectories: Sequence[Path],
    wall_budget_s: float,
    device: str = "cuda:0",
    seed: int = 0,
    checkpoint_interval: int = 250,
) -> dict:
    if wall_budget_s <= 0:
        raise ValueError("wall_budget_s must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    validation = [path.expanduser().resolve() for path in validation_trajectories]
    if not validation or any(not path.is_file() for path in validation):
        raise FileNotFoundError("all frozen validation trajectories must exist")

    output_dir.mkdir(parents=True)
    checkpoint_dir = output_dir / "checkpoints"
    protocol = {
        "baseline": "conventional_cpu_build_sync",
        "num_agents": 512,
        "num_experts": 4,
        "map_names": list(DEFAULT_MAP_NAMES),
        "num_steps_cap": 100_000,
        "max_episode_steps": 256,
        "batch_threshold": 2048,
        "train_batch_size": 2048,
        "ring_buffer_steps": 1,
        "seed": int(seed),
        "lr_start": 1e-3,
        "lr_end": 1e-6,
        "lr_scheduler": "cosine-annealing",
        "scheduler_total_steps": 100_000,
        "grad_clip_norm": None,
        "train_on_arrived_agents": True,
        "checkpoint_interval": int(checkpoint_interval),
        "checkpoint_selection_mode": "validation_accuracy",
        "wall_budget_s": float(wall_budget_s),
        "timing_boundary": (
            "after_worker_cuda_model_validation_initialization; includes_periodic_"
            "validation_and_checkpointing; excludes_final_forced_checkpoint_and_teardown"
        ),
        "validation_trajectories": [
            {"path": str(path), "sha256": _sha256(path)} for path in validation
        ],
        "reference_run": (
            "artifacts/reference-training/"
            "train.log"
        ),
        "reference_total_wall_s": 3035.921933,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Build LaCAM once before multiprocessing workers start.  Concurrent first
    # imports otherwise race in the shared CMake build directory.
    from real_expert_alg.lacam import inference as _lacam_inference  # noqa: F401
    from expert.training_baseline_engines import run_magat_strong_online_engine

    metrics = dict(
        run_magat_strong_online_engine(
            num_steps=100_000,
            num_agents=512,
            num_experts=4,
            batch_threshold=2048,
            train_batch_size=2048,
            seed=int(seed),
            device=device,
            max_episode_steps=256,
            maps_path="maps/maps.yaml",
            map_names=DEFAULT_MAP_NAMES,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_interval=int(checkpoint_interval),
            top_k_checkpoints=3,
            checkpoint_selection_mode="validation_accuracy",
            validation_trajectories=[str(path) for path in validation],
            lr_start=1e-3,
            lr_end=1e-6,
            lr_scheduler="cosine-annealing",
            scheduler_total_steps=100_000,
            grad_clip_norm=None,
            train_on_arrived_agents=True,
            ring_buffer_steps=1,
            health_sample_interval_s=1_000_000_000.0,
            inline_health_sampling=False,
            max_training_wall_s=float(wall_budget_s),
            system_mode="s2_equal_wall_basic",
        )
    )
    result = {
        **metrics,
        "schema_version": 1,
        "experiment": "S2-equal-wall-MAGAT-training",
        "status": "ok" if metrics.get("wall_budget_reached") else "incomplete",
        "protocol": protocol,
        "output_dir": str(output_dir),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-trajectories", required=True)
    parser.add_argument("--wall-budget-s", required=True, type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--checkpoint-interval", default=250, type=int)
    args = parser.parse_args(argv)
    result = run_equal_wall_baseline(
        output_dir=args.output_dir,
        validation_trajectories=[
            Path(value)
            for value in args.validation_trajectories.split(",")
            if value
        ],
        wall_budget_s=args.wall_budget_s,
        device=args.device,
        seed=args.seed,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "num_optimizer_steps",
                    "samples_processed",
                    "total_wall_s",
                    "samples_s",
                    "final_loss",
                    "validation_metrics",
                    "saved_checkpoints",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

