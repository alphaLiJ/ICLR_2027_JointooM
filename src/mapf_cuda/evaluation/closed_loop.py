"""Closed-loop MAGAT evaluation on immutable standard-MAPF batches."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np


def _optional_stat(values: np.ndarray, reducer) -> float | None:
    return None if values.size == 0 else float(reducer(values))


def summarize_closed_loop(
    *,
    instance_ids: np.ndarray,
    num_agents: int,
    horizon: int,
    initial_arrived: np.ndarray,
    final_arrived: np.ndarray,
    arrival_times: np.ndarray,
    arrival_counts: np.ndarray,
    completion_steps: np.ndarray,
    last_arrival_steps: np.ndarray,
    last_motion_steps: np.ndarray,
    requested_moves: np.ndarray,
    blocked_moves: np.ndarray,
) -> dict[str, Any]:
    """Summarize fixed-horizon progress without treating initial arrivals as skill."""

    initial_arrived = np.asarray(initial_arrived, dtype=np.bool_)
    final_arrived = np.asarray(final_arrived, dtype=np.bool_)
    arrival_times = np.asarray(arrival_times, dtype=np.int32)
    arrival_counts = np.asarray(arrival_counts, dtype=np.int32)
    completion_steps = np.asarray(completion_steps, dtype=np.int32)
    last_arrival_steps = np.asarray(last_arrival_steps, dtype=np.int32)
    last_motion_steps = np.asarray(last_motion_steps, dtype=np.int32)
    requested_moves = np.asarray(requested_moves, dtype=np.int64)
    blocked_moves = np.asarray(blocked_moves, dtype=np.int64)
    instance_ids = np.asarray(instance_ids, dtype=np.int64)

    num_envs = int(instance_ids.size)
    expected_agents = (num_envs, int(num_agents))
    if initial_arrived.shape != expected_agents or final_arrived.shape != expected_agents:
        raise ValueError("arrived arrays must have shape [num_envs, num_agents]")
    if arrival_times.shape != expected_agents:
        raise ValueError("arrival_times must have shape [num_envs, num_agents]")
    if arrival_counts.shape != (int(horizon) + 1, num_envs):
        raise ValueError("arrival_counts must have shape [horizon + 1, num_envs]")
    for name, values in (
        ("completion_steps", completion_steps),
        ("last_arrival_steps", last_arrival_steps),
        ("last_motion_steps", last_motion_steps),
        ("requested_moves", requested_moves),
        ("blocked_moves", blocked_moves),
    ):
        if values.shape != (num_envs,):
            raise ValueError(f"{name} must have shape [num_envs]")

    initial_counts = initial_arrived.sum(axis=1, dtype=np.int32)
    final_counts = final_arrived.sum(axis=1, dtype=np.int32)
    unresolved_initial = int(num_envs * num_agents - initial_counts.sum())
    newly_arrived = final_arrived & ~initial_arrived
    solved = completion_steps >= 0
    unresolved_episodes = ~solved
    tail_no_arrival = np.where(
        unresolved_episodes, int(horizon) - last_arrival_steps, 0
    )
    tail_no_motion = np.where(
        unresolved_episodes, int(horizon) - last_motion_steps, 0
    )

    arrival_fraction_curve = arrival_counts.astype(np.float64) / float(num_agents)
    arrival_auc = float(arrival_fraction_curve[1:].mean())
    initially_unresolved_per_env = num_agents - initial_counts
    incremental_curve = np.divide(
        arrival_counts[1:] - initial_counts[None, :],
        initially_unresolved_per_env[None, :],
        out=np.ones((horizon, num_envs), dtype=np.float64),
        where=initially_unresolved_per_env[None, :] > 0,
    )
    requested_total = int(requested_moves.sum())
    blocked_total = int(blocked_moves.sum())
    newly_arrived_times = arrival_times[newly_arrived]
    completed_steps = completion_steps[solved]
    unresolved_count = int((~final_arrived).sum())
    unresolved_episode_count = int(unresolved_episodes.sum())

    per_instance = []
    for env_index, instance_id in enumerate(instance_ids.tolist()):
        requested = int(requested_moves[env_index])
        blocked = int(blocked_moves[env_index])
        per_instance.append(
            {
                "instance_id": int(instance_id),
                "initial_arrived": int(initial_counts[env_index]),
                "final_arrived": int(final_counts[env_index]),
                "individual_success_rate": float(final_counts[env_index] / num_agents),
                "complete_success": bool(solved[env_index]),
                "completion_step": (
                    int(completion_steps[env_index]) if solved[env_index] else None
                ),
                "unresolved_agents": int(num_agents - final_counts[env_index]),
                "tail_without_arrival_steps": int(tail_no_arrival[env_index]),
                "tail_without_motion_steps": int(tail_no_motion[env_index]),
                "no_arrival_in_final_16": bool(
                    unresolved_episodes[env_index] and tail_no_arrival[env_index] >= 16
                ),
                "no_motion_in_final_16": bool(
                    unresolved_episodes[env_index] and tail_no_motion[env_index] >= 16
                ),
                "blocked_requested_move_rate": (
                    0.0 if requested == 0 else float(blocked / requested)
                ),
            }
        )

    return {
        "num_envs": num_envs,
        "num_agents": int(num_agents),
        "horizon": int(horizon),
        "initial_individual_success_rate": float(initial_arrived.mean()),
        "individual_success_rate": float(final_arrived.mean()),
        "normalized_arrival_gain": (
            1.0
            if unresolved_initial == 0
            else float(newly_arrived.sum() / unresolved_initial)
        ),
        "arrival_auc": arrival_auc,
        "incremental_arrival_auc": float(incremental_curve.mean()),
        "complete_success_rate": float(solved.mean()),
        "unresolved_agent_fraction": float(unresolved_count / (num_envs * num_agents)),
        "newly_arrived_agent_count": int(newly_arrived.sum()),
        "newly_arrived_step_mean": _optional_stat(newly_arrived_times, np.mean),
        "newly_arrived_step_median": _optional_stat(newly_arrived_times, np.median),
        "success_conditioned_completion_step_mean": _optional_stat(
            completed_steps, np.mean
        ),
        "success_conditioned_completion_step_median": _optional_stat(
            completed_steps, np.median
        ),
        "unresolved_episode_count": unresolved_episode_count,
        "no_arrival_in_final_16_rate": float(
            np.mean(unresolved_episodes & (tail_no_arrival >= 16))
        ),
        "no_motion_in_final_16_rate": float(
            np.mean(unresolved_episodes & (tail_no_motion >= 16))
        ),
        "unresolved_conditional_no_arrival_in_final_16_rate": (
            0.0
            if unresolved_episode_count == 0
            else float(np.mean(tail_no_arrival[unresolved_episodes] >= 16))
        ),
        "unresolved_conditional_no_motion_in_final_16_rate": (
            0.0
            if unresolved_episode_count == 0
            else float(np.mean(tail_no_motion[unresolved_episodes] >= 16))
        ),
        "requested_moves": requested_total,
        "blocked_moves": blocked_total,
        "blocked_requested_move_rate": (
            0.0 if requested_total == 0 else float(blocked_total / requested_total)
        ),
        "mean_arrival_fraction_by_step": arrival_fraction_curve.mean(axis=1).tolist(),
        "per_instance": per_instance,
    }


def run_magat_closed_loop(
    runtime,
    batch,
    *,
    horizon: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run deterministic argmax MAGAT inference in the resident CUDA simulator."""

    import torch

    from expert.a4_runtime import _build_stateful_simulator

    if horizon <= 0 or horizon > batch.horizon:
        raise ValueError("horizon must be positive and not exceed the frozen input")
    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=horizon
    )
    initial_arrived = simulator.arrived.to(torch.bool).clone()
    arrival_times = torch.full(
        (batch.num_envs, batch.num_agents),
        -1,
        dtype=torch.int32,
        device=device,
    )
    arrival_times.masked_fill_(initial_arrived, 0)
    arrival_counts = torch.empty(
        (horizon + 1, batch.num_envs), dtype=torch.int32, device=device
    )
    arrival_counts[0].copy_(initial_arrived.sum(dim=1, dtype=torch.int32))
    completion_steps = torch.full(
        (batch.num_envs,), -1, dtype=torch.int32, device=device
    )
    completion_steps.masked_fill_(simulator.terminated.to(torch.bool), 0)
    last_arrival_steps = torch.zeros(
        batch.num_envs, dtype=torch.int32, device=device
    )
    last_motion_steps = torch.zeros_like(last_arrival_steps)
    requested_moves = torch.zeros(
        batch.num_envs, dtype=torch.int64, device=device
    )
    blocked_moves = torch.zeros_like(requested_moves)
    previous_x = torch.empty_like(simulator.cur_x)
    previous_y = torch.empty_like(simulator.cur_y)

    torch.cuda.synchronize(torch.device(device))
    wall_started = time.perf_counter()
    executed_steps = 0
    for step in range(horizon):
        terminated_before = simulator.terminated.to(torch.bool)
        truncated_before = simulator.truncated.to(torch.bool)
        active_envs = ~(terminated_before | truncated_before)
        arrived_before = simulator.arrived.to(torch.bool).clone()
        active_agents = ~arrived_before & active_envs[:, None]
        previous_x.copy_(simulator.cur_x)
        previous_y.copy_(simulator.cur_y)

        simulator.update_derived_state()
        simulator.build_magat_plus_inputs()
        data = runtime.batch_builder.view_stateful_outputs(
            simulator, materialize_edges=False
        )
        with torch.no_grad():
            logits = runtime.model(data.x, data)
        actions = logits.argmax(dim=-1).reshape(
            batch.num_envs, batch.num_agents
        ).to(torch.uint8)
        actions.masked_fill_(~active_agents, 0)
        requested = active_agents & actions.ne(0)
        requested_moves.add_(requested.sum(dim=1, dtype=torch.int64))

        simulator.update_actions(actions)
        simulator.step_sim_only()
        arrived_after = simulator.arrived.to(torch.bool)
        moved = active_agents & (
            (simulator.cur_x != previous_x) | (simulator.cur_y != previous_y)
        )
        blocked_moves.add_((requested & ~moved).sum(dim=1, dtype=torch.int64))
        newly_arrived = arrived_after & ~arrived_before
        arrival_times.masked_fill_(newly_arrived, step + 1)
        arrival_progress = newly_arrived.any(dim=1)
        motion_progress = moved.any(dim=1)
        last_arrival_steps.masked_fill_(arrival_progress, step + 1)
        last_motion_steps.masked_fill_(motion_progress, step + 1)
        arrival_counts[step + 1].copy_(
            arrived_after.sum(dim=1, dtype=torch.int32)
        )
        newly_completed = simulator.terminated.to(torch.bool) & ~terminated_before
        completion_steps.masked_fill_(newly_completed, step + 1)
        executed_steps = step + 1
        if bool(
            torch.all(
                simulator.terminated.to(torch.bool)
                | simulator.truncated.to(torch.bool)
            ).item()
        ):
            break

    if executed_steps < horizon:
        arrival_counts[executed_steps + 1 :].copy_(
            arrival_counts[executed_steps].expand(horizon - executed_steps, -1)
        )
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started

    details = {
        "instance_ids": np.asarray(batch.instance_ids, dtype=np.int64),
        "initial_arrived": initial_arrived.cpu().numpy(),
        "final_arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "arrival_times": arrival_times.cpu().numpy(),
        "arrival_counts": arrival_counts.cpu().numpy(),
        "completion_steps": completion_steps.cpu().numpy(),
        "last_arrival_steps": last_arrival_steps.cpu().numpy(),
        "last_motion_steps": last_motion_steps.cpu().numpy(),
        "requested_moves": requested_moves.cpu().numpy(),
        "blocked_moves": blocked_moves.cpu().numpy(),
    }
    summary = summarize_closed_loop(
        instance_ids=details["instance_ids"],
        num_agents=batch.num_agents,
        horizon=horizon,
        initial_arrived=details["initial_arrived"],
        final_arrived=details["final_arrived"],
        arrival_times=details["arrival_times"],
        arrival_counts=details["arrival_counts"],
        completion_steps=details["completion_steps"],
        last_arrival_steps=details["last_arrival_steps"],
        last_motion_steps=details["last_motion_steps"],
        requested_moves=details["requested_moves"],
        blocked_moves=details["blocked_moves"],
    )
    summary.update(
        {
            "executed_batched_steps": int(executed_steps),
            "loop_wall_s_diagnostic": float(wall_s),
            "loop_wall_ms_per_batched_step_diagnostic": (
                None if executed_steps == 0 else wall_s * 1000.0 / executed_steps
            ),
        }
    )
    return summary, details


def run_magat_closed_loop_microbatched(
    runtime,
    batch,
    *,
    horizon: int,
    device: str,
    max_envs_per_batch: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one frozen cell in bounded-memory environment microbatches."""

    from expert.a4_runtime import _slice_batch

    if max_envs_per_batch <= 0:
        raise ValueError("max_envs_per_batch must be positive")
    chunks: list[dict[str, np.ndarray]] = []
    chunk_summaries: list[Mapping[str, Any]] = []
    for start in range(0, batch.num_envs, max_envs_per_batch):
        stop = min(start + max_envs_per_batch, batch.num_envs)
        indices = np.arange(start, stop, dtype=np.int64)
        chunk_summary, chunk_details = run_magat_closed_loop(
            runtime,
            _slice_batch(batch, indices),
            horizon=horizon,
            device=device,
        )
        chunk_summaries.append(chunk_summary)
        chunks.append(chunk_details)

    axis_one = {"arrival_counts"}
    details = {
        key: np.concatenate(
            [chunk[key] for chunk in chunks], axis=1 if key in axis_one else 0
        )
        for key in chunks[0]
    }
    summary = summarize_closed_loop(
        instance_ids=details["instance_ids"],
        num_agents=batch.num_agents,
        horizon=horizon,
        initial_arrived=details["initial_arrived"],
        final_arrived=details["final_arrived"],
        arrival_times=details["arrival_times"],
        arrival_counts=details["arrival_counts"],
        completion_steps=details["completion_steps"],
        last_arrival_steps=details["last_arrival_steps"],
        last_motion_steps=details["last_motion_steps"],
        requested_moves=details["requested_moves"],
        blocked_moves=details["blocked_moves"],
    )
    wall_s = float(sum(float(row["loop_wall_s_diagnostic"]) for row in chunk_summaries))
    executed = int(
        sum(int(row["executed_batched_steps"]) for row in chunk_summaries)
    )
    summary.update(
        {
            "evaluation_microbatch_envs": int(max_envs_per_batch),
            "evaluation_microbatch_count": len(chunks),
            "executed_batched_steps_total": executed,
            "loop_wall_s_diagnostic": wall_s,
            "loop_wall_ms_per_batched_step_diagnostic": (
                None if executed == 0 else wall_s * 1000.0 / executed
            ),
        }
    )
    return summary, details


__all__ = [
    "run_magat_closed_loop",
    "run_magat_closed_loop_microbatched",
    "summarize_closed_loop",
]
