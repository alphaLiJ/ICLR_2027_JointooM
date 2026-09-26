"""F5 homogeneous-topology control for the F3 ring-depth result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from expert.minimal_f2_producer_scaling_runner import run_f2_scaling


NUM_AGENTS = 256
NUM_PRODUCERS = 8
FRONTIER_STEPS = 500
BATCH_ENVS = 4
RING_DEPTHS = (8, 32, 128)
HOMOGENEOUS_MAP = "mazes-s0_wc8_od55"
HOMOGENEOUS_MAP_NAMES = (HOMOGENEOUS_MAP,) * 8


def run_f5_ring_skew_control(
    *,
    ring_depth: int,
    seed: int,
    output_dir: Path,
    frontier_steps: int = FRONTIER_STEPS,
    device: str = "cuda:0",
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if ring_depth not in RING_DEPTHS:
        raise ValueError(f"ring_depth must be one of {RING_DEPTHS}")
    result = run_f2_scaling(
        num_producers=NUM_PRODUCERS,
        num_agents=NUM_AGENTS,
        seed=seed,
        frontier_steps=frontier_steps,
        output_dir=output_dir,
        ring_capacity_steps=ring_depth,
        device=device,
        experiment_label="F5",
        envs_per_optimizer_update=BATCH_ENVS,
        map_names=HOMOGENEOUS_MAP_NAMES,
        engine=engine,
    )
    result["map_condition"] = "homogeneous_topology"
    result["ring_depth"] = int(ring_depth)
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-depth", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frontier-steps", type=int, default=FRONTIER_STEPS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_f5_ring_skew_control(
            ring_depth=args.ring_depth,
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
                        "map_condition",
                        "ring_depth",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                        "consumer_wait_s",
                        "worker_ringbuffer_write_s_max",
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
            "experiment": "F5",
            "ring_depth": args.ring_depth,
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
