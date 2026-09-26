"""F4 optimizer-batch granularity at a consumer-saturating producer count."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from expert.minimal_f2_producer_scaling_runner import run_f2_scaling


NUM_AGENTS = 256
NUM_PRODUCERS = 8
FRONTIER_STEPS = 500
RING_DEPTH = 128
BATCH_ENVS = (1, 2, 4, 8)


def run_f4_batch_granularity(
    *,
    batch_envs: int,
    seed: int,
    output_dir: Path,
    frontier_steps: int = FRONTIER_STEPS,
    device: str = "cuda:0",
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if batch_envs not in BATCH_ENVS:
        raise ValueError(f"batch_envs must be one of {BATCH_ENVS}")
    result = run_f2_scaling(
        num_producers=NUM_PRODUCERS,
        num_agents=NUM_AGENTS,
        seed=seed,
        frontier_steps=frontier_steps,
        output_dir=output_dir,
        ring_capacity_steps=RING_DEPTH,
        device=device,
        experiment_label="F4",
        envs_per_optimizer_update=batch_envs,
        engine=engine,
    )
    result["batch_envs"] = int(batch_envs)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-envs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frontier-steps", type=int, default=FRONTIER_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_f4_batch_granularity(
            batch_envs=args.batch_envs,
            seed=args.seed,
            frontier_steps=args.frontier_steps,
            device=args.device,
            output_dir=args.output_dir,
        )
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "status",
                        "batch_envs",
                        "train_batch_size",
                        "num_optimizer_steps",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                        "consumer_wait_s",
                        "consumer_train_step_s",
                        "peak_gpu_memory_bytes",
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
            "experiment": "F4",
            "batch_envs": args.batch_envs,
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
