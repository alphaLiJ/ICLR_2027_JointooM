"""Long-horizon MAGAT resident-loop parity and performance experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from expert.minimal_scaling_runner import load_scan_cell


def _gpu_state(simulator) -> dict[str, np.ndarray]:
    import torch

    return {
        "positions": torch.stack(
            (simulator.cur_x, simulator.cur_y), dim=-1
        ).to(torch.uint16).cpu().numpy(),
        "arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "terminated": simulator.terminated.to(torch.bool).cpu().numpy(),
        "truncated": simulator.truncated.to(torch.bool).cpu().numpy(),
        "step_counts": simulator.step_counts.cpu().numpy(),
    }


def _assert_state_equal(
    cpu: dict[str, Any], gpu: dict[str, np.ndarray], *, step: int
) -> None:
    for key in ("positions", "arrived", "terminated", "truncated", "step_counts"):
        expected = np.asarray(cpu[key])
        actual = np.asarray(gpu[key])
        if not np.array_equal(expected, actual):
            mismatch = int(np.count_nonzero(expected != actual))
            raise RuntimeError(
                f"P1 shared-action parity failed at step {step} for {key}: "
                f"{mismatch} mismatches"
            )


def _model_actions(runtime, simulator, *, num_envs: int, num_agents: int):
    import torch

    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()
    data = runtime.batch_builder.view_stateful_outputs(
        simulator, materialize_edges=False
    )
    with torch.no_grad():
        logits = runtime.model(data.x, data)
    actions = logits.argmax(dim=-1).reshape(num_envs, num_agents).to(torch.uint8)
    active_envs = ~(
        simulator.terminated.to(torch.bool)
        | simulator.truncated.to(torch.bool)
    )
    active_agents = (
        ~simulator.arrived.to(torch.bool) & active_envs[:, None]
    )
    actions.masked_fill_(~active_agents, 0)
    return actions, active_envs, active_agents


def run_shared_action_parity(
    runtime,
    batch,
    *,
    horizon: int,
    parity_envs: int,
    device: str,
) -> dict[str, Any]:
    import torch

    from expert.a4_runtime import (
        _CpuEpisodeSet,
        _build_stateful_simulator,
        _digest_arrays,
        _slice_batch,
    )

    count = min(int(parity_envs), batch.num_envs)
    if count <= 0:
        return {"steps": 0, "num_envs": 0, "state_mismatches": 0}
    parity_batch = _slice_batch(batch, np.arange(count, dtype=np.int64))
    cpu = _CpuEpisodeSet(parity_batch, horizon)
    simulator = _build_stateful_simulator(
        parity_batch, device=device, max_episode_steps=horizon
    )
    action_trajectory = np.empty(
        (horizon, count, parity_batch.num_agents), dtype=np.uint8
    )
    for step in range(horizon):
        actions, _, _ = _model_actions(
            runtime,
            simulator,
            num_envs=count,
            num_agents=parity_batch.num_agents,
        )
        actions_host = actions.cpu().numpy()
        action_trajectory[step] = actions_host
        cpu.step(actions_host, step)
        simulator.update_actions(actions)
        simulator.step_sim_only()
        torch.cuda.synchronize(torch.device(device))
        _assert_state_equal(cpu.result(), _gpu_state(simulator), step=step + 1)
    return {
        "steps": horizon,
        "num_envs": count,
        "num_agents": parity_batch.num_agents,
        "state_mismatches": 0,
        "action_trajectory_sha256": _digest_arrays(actions=action_trajectory),
        "final_state_sha256": _digest_arrays(**cpu.result()),
    }


def run_fixed_resident_loop(
    runtime,
    batch,
    *,
    horizon: int,
    device: str,
) -> dict[str, Any]:
    import torch

    from expert.a4_runtime import (
        _build_stateful_simulator,
        _digest_arrays,
        _final_state_digest,
    )

    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=horizon
    )
    trajectory = torch.empty(
        (horizon, batch.num_envs, batch.num_agents),
        dtype=torch.uint8,
        device=device,
    )
    requested_moves = torch.zeros((), dtype=torch.int64, device=device)
    blocked_moves = torch.zeros((), dtype=torch.int64, device=device)
    active_env_steps = torch.zeros((), dtype=torch.int64, device=device)
    active_agent_steps = torch.zeros((), dtype=torch.int64, device=device)
    previous_x = torch.empty_like(simulator.cur_x)
    previous_y = torch.empty_like(simulator.cur_y)

    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_started = time.perf_counter()
    for step in range(horizon):
        previous_x.copy_(simulator.cur_x)
        previous_y.copy_(simulator.cur_y)
        actions, active_envs, active_agents = _model_actions(
            runtime,
            simulator,
            num_envs=batch.num_envs,
            num_agents=batch.num_agents,
        )
        trajectory[step].copy_(actions)
        active_env_steps.add_(active_envs.sum())
        active_agent_steps.add_(active_agents.sum())
        requested = active_agents & actions.ne(0)
        requested_moves.add_(requested.sum())
        simulator.update_actions(actions)
        simulator.step_sim_only()
        moved = (simulator.cur_x != previous_x) | (simulator.cur_y != previous_y)
        blocked_moves.add_((requested & ~moved).sum())
    end_event.record()
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started
    gpu_ms = float(start_event.elapsed_time(end_event))
    peak_allocated = int(torch.cuda.max_memory_allocated(torch.device(device)))
    peak_reserved = int(torch.cuda.max_memory_reserved(torch.device(device)))

    state = _gpu_state(simulator)
    trajectory_host = trajectory.cpu().numpy()
    requested = int(requested_moves.item())
    blocked = int(blocked_moves.item())
    env_steps = int(active_env_steps.item())
    agent_steps = int(active_agent_steps.item())
    return {
        "wall_s": wall_s,
        "gpu_total_ms": gpu_ms,
        "wall_ms_per_batched_step": wall_s / horizon * 1000.0,
        "gpu_ms_per_batched_step": gpu_ms / horizon,
        "nominal_env_steps_s": batch.num_envs * horizon / wall_s,
        "nominal_agent_steps_s": (
            batch.num_envs * batch.num_agents * horizon / wall_s
        ),
        "active_env_steps": env_steps,
        "active_agent_steps": agent_steps,
        "active_env_steps_s": env_steps / wall_s,
        "active_agent_steps_s": agent_steps / wall_s,
        "requested_moves": requested,
        "blocked_moves": blocked,
        "blocked_move_rate": 0.0 if requested == 0 else blocked / requested,
        "peak_gpu_memory_allocated_bytes": peak_allocated,
        "peak_gpu_memory_reserved_bytes": peak_reserved,
        "resident_loop_h2d_bytes": 0,
        "resident_loop_d2h_bytes": 0,
        "validation_in_timed_region": False,
        "monitoring_in_timed_region": False,
        "cold_derived_state_in_timed_region": True,
        "individual_success_rate": float(state["arrived"].mean()),
        "complete_success_rate": float(state["terminated"].mean()),
        "truncation_rate": float(state["truncated"].mean()),
        "step_count_mean": float(state["step_counts"].mean()),
        "trajectory_action_sha256": _digest_arrays(actions=trajectory_host),
        "final_state_sha256": _final_state_digest(state, batch.goals),
    }


def run_p1_long(
    *,
    manifest: Path,
    scan_cell: str,
    checkpoint: Path,
    horizon: int = 120,
    parity_envs: int = 1,
    device: str = "cuda:0",
) -> dict[str, Any]:
    from expert.a4_runtime import _frozen_state_parity, _make_runtime, _slice_batch

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    cell, batch = load_scan_cell(manifest, scan_cell)
    if batch.num_agents != 256:
        raise ValueError("P1 long-horizon experiment requires 256 agents")
    if horizon > batch.horizon:
        raise ValueError("horizon exceeds frozen input horizon")
    runtime = _make_runtime(str(checkpoint.expanduser().resolve()), device)
    one_step = _frozen_state_parity(
        runtime,
        _slice_batch(batch, np.asarray([0], dtype=np.int64)),
        device=device,
    )
    temporal = run_shared_action_parity(
        runtime,
        batch,
        horizon=horizon,
        parity_envs=parity_envs,
        device=device,
    )
    performance = run_fixed_resident_loop(
        runtime, batch, horizon=horizon, device=device
    )
    return {
        "schema_version": 1,
        "status": "ok",
        "experiment": "P1-MAGAT-long-resident-loop",
        "scan_cell": scan_cell,
        "cell": cell,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
        "horizon": horizon,
        "input_semantic_sha256": batch.semantic_sha256,
        "action_selection": "argmax",
        "one_step_model_parity": one_step,
        "temporal_shared_action_parity": temporal,
        **performance,
    }


def _write(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    keys = (
        "status",
        "num_envs",
        "num_agents",
        "horizon",
        "wall_ms_per_batched_step",
        "nominal_env_steps_s",
        "active_env_steps_s",
        "individual_success_rate",
        "complete_success_rate",
        "truncation_rate",
        "exception_type",
        "exception_message",
    )
    summary = {key: report.get(key) for key in keys if report.get(key) is not None}
    (output_dir / "stdout.log").write_text(
        json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one long-horizon MAGAT resident-loop row."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--scan-cell", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--parity-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_p1_long(
            manifest=args.manifest,
            scan_cell=args.scan_cell,
            checkpoint=args.checkpoint,
            horizon=args.horizon,
            parity_envs=args.parity_envs,
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
            "scan_cell": args.scan_cell,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
        }
        _write(args.output_dir, report)
        print(json.dumps(report, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_assert_state_equal",
    "run_fixed_resident_loop",
    "run_p1_long",
    "run_shared_action_parity",
]
