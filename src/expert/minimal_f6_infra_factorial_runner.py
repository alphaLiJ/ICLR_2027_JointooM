"""F6 ring-depth by transfer-mode infrastructure factorial."""

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
RING_DEPTHS = (2, 128)
TRANSFER_MODES = ("sync", "async")


def run_f6_infra_factorial(
    *,
    ring_depth: int,
    transfer_mode: str,
    seed: int,
    output_dir: Path,
    frontier_steps: int = FRONTIER_STEPS,
    capture_stage_hashes: bool = False,
    device: str = "cuda:0",
    engine: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if ring_depth not in RING_DEPTHS:
        raise ValueError(f"ring_depth must be one of {RING_DEPTHS}")
    transfer_mode = str(transfer_mode).strip().lower()
    if transfer_mode not in TRANSFER_MODES:
        raise ValueError(f"transfer_mode must be one of {TRANSFER_MODES}")
    result = run_f2_scaling(
        num_producers=NUM_PRODUCERS,
        num_agents=NUM_AGENTS,
        seed=seed,
        frontier_steps=frontier_steps,
        output_dir=output_dir,
        ring_capacity_steps=ring_depth,
        device=device,
        capture_stage_hashes=capture_stage_hashes,
        experiment_label="F6",
        envs_per_optimizer_update=BATCH_ENVS,
        transfer_mode=transfer_mode,
        engine=engine,
    )
    result["ring_depth"] = int(ring_depth)
    result["transfer_mode"] = transfer_mode
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ring-depth", type=int, required=True)
    parser.add_argument("--transfer-mode", choices=TRANSFER_MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frontier-steps", type=int, default=FRONTIER_STEPS)
    parser.add_argument("--capture-stage-hashes", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_f6_infra_factorial(
            ring_depth=args.ring_depth,
            transfer_mode=args.transfer_mode,
            seed=args.seed,
            frontier_steps=args.frontier_steps,
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
                        "transfer_mode",
                        "ring_depth",
                        "samples_processed",
                        "total_wall_s",
                        "samples_s",
                        "consumer_wait_s",
                        "consumer_dma_sync_s",
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
            "experiment": "F6",
            "transfer_mode": args.transfer_mode,
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
