"""Lightweight fixed-step MAGAT deployment benchmark for E4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expert.minimal_scaling_runner import load_scan_cell


MODES = ("cpu_single", "cpu_mp_gpu_model", "gpu_stateful")


def _peak_nvml_from_health(result: dict[str, Any]) -> int | None:
    values = [
        (event.get("gpu") or {}).get("memory_used_bytes")
        for event in result.get("health_report", {}).get("events", [])
    ]
    observed = [int(value) for value in values if value is not None]
    return max(observed) if observed else None


def run_e4_deployment(
    *,
    manifest: Path,
    scan_cell: str,
    checkpoint: Path,
    mode: str,
    num_steps: int,
    num_workers: int = 8,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported E4 mode: {mode}")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    cell, batch = load_scan_cell(manifest, scan_cell)
    if batch.num_agents != 256:
        raise ValueError("E4 currently requires exactly 256 agents")
    if num_steps > batch.horizon:
        raise ValueError("num_steps exceeds frozen input horizon")

    from expert.a4_runtime import (
        _frozen_state_parity,
        _make_runtime,
        _run_cpu_single_engine,
        _run_gpu_stateful_engine,
        _slice_batch,
    )

    runtime = _make_runtime(str(checkpoint.expanduser().resolve()), device)
    # One frozen environment is enough to validate and warm the shared builder
    # and checkpoint outside the timed deployment region.
    parity_batch = _slice_batch(batch, np.asarray([0], dtype=np.int64))
    parity = _frozen_state_parity(runtime, parity_batch, device=device)

    if mode == "cpu_single":
        result = _run_cpu_single_engine(
            runtime,
            batch,
            max_episode_steps=num_steps,
            device=device,
        )
    elif mode == "gpu_stateful":
        result = _run_gpu_stateful_engine(
            runtime,
            batch,
            max_episode_steps=num_steps,
            device=device,
        )
    else:
        from expert.a4_cpu_mp import run_cpu_mp_engine

        result = run_cpu_mp_engine(
            runtime,
            batch,
            max_episode_steps=num_steps,
            device=device,
            num_workers=num_workers,
        )

    result.update(
        {
            "schema_version": 1,
            "status": "ok",
            "mode": mode,
            "scan_cell": scan_cell,
            "cell": cell,
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "input_semantic_sha256": batch.semantic_sha256,
            "num_envs": batch.num_envs,
            "num_agents": batch.num_agents,
            "fixed_steps_requested": num_steps,
            "num_workers": min(num_workers, batch.num_envs)
            if mode == "cpu_mp_gpu_model"
            else 0,
            "warmup_and_parity_envs": 1,
            "parity": parity,
            "peak_nvml_gpu_memory_bytes": _peak_nvml_from_health(result),
            "timed_region": "fixed_step_complete_closed_loop",
            "correctness_replay_in_timed_region": False,
        }
    )
    return result


def _write(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "stdout.log").write_text(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "mode",
                    "num_envs",
                    "num_agents",
                    "executed_steps",
                    "wall_ms_per_batched_step",
                    "env_steps_s",
                    "agent_steps_s",
                    "peak_gpu_memory_bytes",
                    "peak_host_memory_bytes",
                )
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one lightweight E4 row.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_e4_deployment(
            manifest=args.manifest,
            scan_cell=args.scan_cell,
            checkpoint=args.checkpoint,
            mode=args.mode,
            num_steps=args.num_steps,
            num_workers=args.num_workers,
            device=args.device,
        )
        _write(args.output_dir, report)
        print((args.output_dir / "stdout.log").read_text(encoding="utf-8").strip())
        return 0
    except Exception as error:
        message = str(error)
        report = {
            "schema_version": 1,
            "status": (
                "capacity_failure"
                if "out of memory" in message.lower()
                else "failed"
            ),
            "mode": args.mode,
            "scan_cell": args.scan_cell,
            "exception_type": type(error).__name__,
            "exception_message": message,
            "pid": os.getpid(),
        }
        _write(args.output_dir, report)
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MODES", "run_e4_deployment"]
