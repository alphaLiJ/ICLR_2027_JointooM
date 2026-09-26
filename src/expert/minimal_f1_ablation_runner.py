"""Minimal F1 data-path ablation for MAGAT deployment.

The formal F1 report reuses the already-frozen E4 ``cpu_single`` and
``gpu_stateful`` endpoints.  This runner adds the missing middle path:

    CPU POGEMA state -> pinned compact rows -> CUDA stateless builder
    -> MAGAT model -> pinned action readback -> CPU POGEMA transition.

Checkpoint loading, parity validation, simulator allocation, and optional
monitoring are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expert.minimal_scaling_runner import load_scan_cell


MODE = "compact_h2d"


def _fill_compact_rows(episodes, rows: np.ndarray, *, refresh: bool) -> None:
    """Materialize the fixed-width deployment boundary into pinned storage."""

    num_envs = episodes.batch.num_envs
    num_agents = episodes.batch.num_agents
    shaped = rows.reshape(num_envs, num_agents, 8)
    shaped[:, :, 0] = np.arange(num_envs, dtype=np.int16)[:, None]
    shaped[:, :, 1] = np.arange(num_agents, dtype=np.int16)[None, :]
    for env_id, env in enumerate(episodes.envs):
        positions = np.asarray(
            env.grid.get_agents_xy(ignore_borders=True), dtype=np.int16
        )
        shaped[env_id, :, 2:4] = positions
    shaped[:, :, 4:6] = episodes.batch.goals.astype(np.int16, copy=False)
    shaped[:, :, 6] = 0
    shaped[:, :, 7] = 1 if refresh else 0


def _event_seconds(pairs) -> float:
    return float(sum(start.elapsed_time(end) for start, end in pairs) / 1000.0)


def _run_compact_h2d_engine(
    runtime,
    batch,
    *,
    max_episode_steps: int,
    device: str,
) -> dict[str, Any]:
    import torch
    import grid_world_cpp as ext

    from expert.a4_runtime import (
        _CpuEpisodeSet,
        _digest_arrays,
        _final_state_digest,
        _quality_metrics,
    )
    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator

    episodes = _CpuEpisodeSet(batch, max_episode_steps)
    grids, _ = stack_grids_for_compiled_simulator(
        list(batch.grids), device=device
    )
    builder = ext.StatelessGridWorldSimulator(grids, batch.num_agents, 3)
    builder.pyg_builder_mode = "local_gather"
    builder.pyg_local_gather_impl = "auto"

    num_rows = batch.num_envs * batch.num_agents
    compact_host = torch.empty(
        (num_rows, 8), dtype=torch.int16, pin_memory=True
    )
    compact_device = torch.empty(
        (num_rows, 8), dtype=torch.int16, device=device
    )
    action_host = torch.empty(
        (batch.num_envs, batch.num_agents),
        dtype=torch.uint8,
        pin_memory=True,
    )
    trajectory = np.zeros(
        (max_episode_steps, batch.num_envs, batch.num_agents), dtype=np.uint8
    )
    compact_numpy = compact_host.numpy()

    h2d_events = []
    builder_events = []
    model_events = []
    argmax_events = []
    d2h_events = []

    def event_pair():
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    cpu_pack_s = 0.0
    cpu_transition_s = 0.0
    executed_steps = 0

    for step in range(max_episode_steps):
        if np.all(episodes.terminated | episodes.truncated):
            break

        stage_started = time.perf_counter()
        _fill_compact_rows(episodes, compact_numpy, refresh=(step == 0))
        cpu_pack_s += time.perf_counter() - stage_started

        h2d_pair = event_pair()
        h2d_pair[0].record()
        compact_device.copy_(compact_host, non_blocking=True)
        h2d_pair[1].record()
        h2d_events.append(h2d_pair)

        builder_pair = event_pair()
        builder_pair[0].record()
        if step == 0:
            builder.update_energy_maps(compact_device, num_rows)
        builder.refresh_compact_state_from_raw_batch(compact_device)
        data = runtime.batch_builder.build(
            builder, compact_device, materialize_edges=False
        )
        builder_pair[1].record()
        builder_events.append(builder_pair)

        model_pair = event_pair()
        model_pair[0].record()
        with torch.no_grad():
            logits = runtime.model(data.x, data)
        model_pair[1].record()
        model_events.append(model_pair)

        argmax_pair = event_pair()
        argmax_pair[0].record()
        actions = logits.argmax(dim=-1).reshape(
            batch.num_envs, batch.num_agents
        ).to(torch.uint8)
        argmax_pair[1].record()
        argmax_events.append(argmax_pair)

        d2h_pair = event_pair()
        d2h_pair[0].record()
        action_host.copy_(actions, non_blocking=True)
        d2h_pair[1].record()
        d2h_events.append(d2h_pair)
        torch.cuda.current_stream(torch.device(device)).synchronize()

        actions_numpy = action_host.numpy()
        trajectory[step] = actions_numpy
        stage_started = time.perf_counter()
        episodes.step(actions_numpy, step)
        cpu_transition_s += time.perf_counter() - stage_started
        executed_steps = step + 1

    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started
    cpu_process_s = time.process_time() - cpu_started
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
    result = episodes.result()
    result["executed_steps"] = int(executed_steps)

    h2d_s = _event_seconds(h2d_events)
    gpu_builder_s = _event_seconds(builder_events)
    model_forward_s = _event_seconds(model_events)
    argmax_s = _event_seconds(argmax_events)
    d2h_s = _event_seconds(d2h_events)
    h2d_bytes = int(executed_steps * compact_host.numel() * compact_host.element_size())
    d2h_bytes = int(executed_steps * action_host.numel() * action_host.element_size())

    return {
        **_quality_metrics(result, max_episode_steps),
        "total_wall_s": float(wall_s),
        "wall_ms_per_batched_step": (
            None if executed_steps == 0 else wall_s / executed_steps * 1000.0
        ),
        "executed_steps": int(executed_steps),
        "env_steps": int(result["env_steps"]),
        "agent_steps": int(result["agent_steps"]),
        "env_steps_s": float(result["env_steps"] / wall_s),
        "agent_steps_s": float(result["agent_steps"] / wall_s),
        "num_envs": int(batch.num_envs),
        "num_agents": int(batch.num_agents),
        "h2d_bytes": h2d_bytes,
        "d2h_bytes": d2h_bytes,
        "compact_bytes_per_batched_step": int(
            compact_host.numel() * compact_host.element_size()
        ),
        "peak_host_memory_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "peak_gpu_memory_bytes": peak_gpu,
        "cpu_process_utilization_pct": float(cpu_process_s / wall_s * 100.0),
        "cpu_pack_s": float(cpu_pack_s),
        "compact_h2d_s": h2d_s,
        "gpu_builder_s": gpu_builder_s,
        "model_forward_s": model_forward_s,
        "argmax_s": argmax_s,
        "action_d2h_s": d2h_s,
        "cpu_transition_s": float(cpu_transition_s),
        "timed_health_sampling": False,
        "accepted": True,
        "trajectory_action_sha256": _digest_arrays(
            actions=trajectory[:executed_steps]
        ),
        "final_state_sha256": _final_state_digest(result, batch.goals),
    }


def run_f1_ablation(
    *,
    manifest: Path,
    scan_cell: str,
    checkpoint: Path,
    num_steps: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    cell, batch = load_scan_cell(manifest, scan_cell)
    if batch.num_agents != 256:
        raise ValueError("F1 currently requires exactly 256 agents")
    if num_steps > batch.horizon:
        raise ValueError("num_steps exceeds frozen input horizon")

    from expert.a4_runtime import _frozen_state_parity, _make_runtime, _slice_batch

    runtime = _make_runtime(str(checkpoint.expanduser().resolve()), device)
    parity_batch = _slice_batch(batch, np.asarray([0], dtype=np.int64))
    parity = _frozen_state_parity(runtime, parity_batch, device=device)
    result = _run_compact_h2d_engine(
        runtime,
        batch,
        max_episode_steps=num_steps,
        device=device,
    )
    result.update(
        {
            "schema_version": 1,
            "status": "ok",
            "experiment": "F1",
            "mode": MODE,
            "scan_cell": scan_cell,
            "cell": cell,
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "input_semantic_sha256": batch.semantic_sha256,
            "fixed_steps_requested": int(num_steps),
            "parity": parity,
            "timed_region": "fixed_step_compact_h2d_closed_loop",
            "validation_in_timed_region": False,
            "monitoring_in_timed_region": False,
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
                    "h2d_bytes",
                )
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one F1 compact-H2D row.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_f1_ablation(
            manifest=args.manifest,
            scan_cell=args.scan_cell,
            checkpoint=args.checkpoint,
            num_steps=args.num_steps,
            device=args.device,
        )
        _write(args.output_dir, report)
        print((args.output_dir / "stdout.log").read_text(encoding="utf-8").strip())
        return 0
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": (
                "capacity_failure"
                if "out of memory" in str(error).lower()
                else "failed"
            ),
            "experiment": "F1",
            "mode": MODE,
            "scan_cell": args.scan_cell,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "pid": os.getpid(),
        }
        _write(args.output_dir, report)
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MODE", "_fill_compact_rows", "run_f1_ablation"]
