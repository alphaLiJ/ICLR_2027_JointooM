"""Shared execution engines for the MAGAT A4 deployment study."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from expert.benchmark_contract import FrozenTransitionBatch


def _digest_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        for component in (
            name.encode("ascii"),
            value.dtype.str.encode("ascii"),
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
            value.tobytes(order="C"),
        ):
            digest.update(len(component).to_bytes(8, "little"))
            digest.update(component)
    return digest.hexdigest()


def _slice_batch(batch: FrozenTransitionBatch, indices: np.ndarray) -> FrozenTransitionBatch:
    indices = np.asarray(indices, dtype=np.int64)
    return FrozenTransitionBatch(
        instance_ids=batch.instance_ids[indices],
        grids=batch.grids[indices],
        positions=batch.positions[indices],
        goals=batch.goals[indices],
        arrived=batch.arrived[indices],
        active=batch.active[indices],
        actions=batch.actions[:, indices],
        horizon=batch.horizon,
    )


def _make_pogema_envs(batch: FrozenTransitionBatch, max_episode_steps: int):
    from pogema.envs import PogemaCoopFinish
    from pogema.grid_config import GridConfig

    envs = []
    for env_index in range(batch.num_envs):
        config = GridConfig(
            map=batch.grids[env_index].tolist(),
            agents_xy=batch.positions[env_index].tolist(),
            targets_xy=batch.goals[env_index].tolist(),
            num_agents=batch.num_agents,
            on_target="nothing",
            collision_system="soft",
            observation_type="MAPF",
            max_episode_steps=int(max_episode_steps),
            empty_outside=True,
        )
        env = PogemaCoopFinish(grid_config=config)
        env._initialize_grid()
        env.update_was_on_goal()
        if not hasattr(env, "num_agents"):
            env.num_agents = env.get_num_agents()
        envs.append(env)
    return envs


def _compact_rows_for_env(env, env_id: int) -> np.ndarray:
    positions = np.asarray(env.grid.get_agents_xy(ignore_borders=True), dtype=np.uint16)
    goals = np.asarray(env.grid.get_targets_xy(ignore_borders=True), dtype=np.uint16)
    rows = np.zeros((positions.shape[0], 8), dtype=np.uint16)
    rows[:, 0] = int(env_id)
    rows[:, 1] = np.arange(positions.shape[0], dtype=np.uint16)
    rows[:, 2:4] = positions
    rows[:, 4:6] = goals
    return rows


class _CpuEpisodeSet:
    def __init__(self, batch: FrozenTransitionBatch, max_episode_steps: int):
        from magat_plus.additional_data.cost_to_go_calculator import CostToGoCalculator
        from expert.fixed_magat_plus_runtime import OBS_RADIUS

        self.batch = batch
        self.horizon = int(max_episode_steps)
        self.envs = _make_pogema_envs(batch, self.horizon)
        self.calculators = [
            CostToGoCalculator(
                env=env,
                obs_radius=OBS_RADIUS,
                dtype="float32",
                pad_cost_to_go=True,
                clamp_value=1.0,
                clamp_values_doubled=False,
            )
            for env in self.envs
        ]
        self.arrived = np.array(batch.arrived, dtype=np.bool_, copy=True)
        self.arrival_times = np.where(self.arrived, 0, self.horizon).astype(np.int32)
        self.terminated = np.all(self.arrived, axis=1)
        self.truncated = np.zeros(batch.num_envs, dtype=np.bool_)
        self.step_counts = np.zeros(batch.num_envs, dtype=np.int32)
        self.requested_moves = 0
        self.blocked_moves = 0
        self.env_steps = 0
        self.agent_steps = 0

    def build_references(self):
        from expert.magat_reference_oracle import build_magat_reference_batch_from_env

        return [
            build_magat_reference_batch_from_env(
                env,
                _compact_rows_for_env(env, env_id),
                cost_to_go_calculator=self.calculators[env_id],
            )
            for env_id, env in enumerate(self.envs)
        ]

    def step(self, actions: np.ndarray, step_index: int) -> None:
        for env_id, env in enumerate(self.envs):
            if self.terminated[env_id] or self.truncated[env_id]:
                continue
            before = np.asarray(
                env.grid.get_agents_xy(ignore_borders=True), dtype=np.int64
            )
            before_arrived = self.arrived[env_id].copy()
            effective_actions = np.array(actions[env_id], dtype=np.uint8, copy=True)
            effective_actions[before_arrived] = 0
            requested = (~before_arrived) & (effective_actions != 0)
            self.requested_moves += int(requested.sum())
            self.agent_steps += int((~before_arrived).sum())
            self.env_steps += 1
            _, _, terminated, truncated, _ = env.step(
                effective_actions.astype(np.int64, copy=False)
            )
            after = np.asarray(
                env.grid.get_agents_xy(ignore_borders=True), dtype=np.int64
            )
            self.blocked_moves += int((requested & np.all(after == before, axis=1)).sum())
            now_arrived = before_arrived | np.all(
                after == self.batch.goals[env_id], axis=1
            )
            newly_arrived = now_arrived & ~before_arrived
            self.arrival_times[env_id, newly_arrived] = int(step_index) + 1
            self.arrived[env_id] = now_arrived
            self.step_counts[env_id] += 1
            self.terminated[env_id] = bool(all(terminated))
            # These fixed-instance environments intentionally bypass Pogema's
            # outer MultiTimeLimit wrapper, so reproduce its horizon contract
            # explicitly for parity with the resident CUDA transition kernel.
            reached_horizon = self.step_counts[env_id] >= self.horizon
            self.truncated[env_id] = bool(
                (all(truncated) or reached_horizon) and not self.terminated[env_id]
            )

    def final_positions(self) -> np.ndarray:
        return np.asarray(
            [env.grid.get_agents_xy(ignore_borders=True) for env in self.envs],
            dtype=np.uint16,
        )

    def result(self) -> dict[str, Any]:
        return {
            "arrival_times": self.arrival_times,
            "arrived": self.arrived,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "step_counts": self.step_counts,
            "positions": self.final_positions(),
            "requested_moves": int(self.requested_moves),
            "blocked_moves": int(self.blocked_moves),
            "env_steps": int(self.env_steps),
            "agent_steps": int(self.agent_steps),
        }


def _to_device(array: np.ndarray, *, device: str, dtype=None):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.pin_memory().to(device, non_blocking=True)


def _batch_references(references, arrived: np.ndarray, *, device: str):
    import torch

    from expert.fixed_magat_plus_runtime import RuntimePyGBatch

    num_envs = len(references)
    num_agents = int(references[0].x.shape[0])
    x = np.concatenate([item.x for item in references], axis=0).astype(np.float32, copy=False)
    edge_parts = [
        item.edge_index + env_id * num_agents
        for env_id, item in enumerate(references)
    ]
    edge_index = np.concatenate(edge_parts, axis=1).astype(np.int64, copy=False)
    edge_attr = np.concatenate([item.edge_attr for item in references], axis=0).astype(
        np.float32, copy=False
    )
    batch_index = np.repeat(np.arange(num_envs, dtype=np.int64), num_agents)
    ptr = np.arange(0, num_envs * num_agents + 1, num_agents, dtype=np.int64)
    arrived_flat = np.asarray(arrived, dtype=np.bool_).reshape(-1)
    data = RuntimePyGBatch(
        x=_to_device(x, device=device),
        edge_index=_to_device(edge_index, device=device, dtype=torch.int64),
        edge_attr=_to_device(edge_attr, device=device),
        edge_index_storage=None,
        edge_attr_storage=None,
        num_edges=torch.tensor([edge_index.shape[1]], dtype=torch.int64, device=device),
        batch=_to_device(batch_index, device=device, dtype=torch.int64),
        ptr=_to_device(ptr, device=device, dtype=torch.int64),
        y=torch.zeros(num_envs * num_agents, dtype=torch.int64, device=device),
        terminated=torch.zeros(num_envs * num_agents, dtype=torch.bool, device=device),
        arrived=_to_device(arrived_flat, device=device, dtype=torch.bool),
    )
    h2d_bytes = sum(
        value.nbytes
        for value in (x, edge_index, edge_attr, batch_index, ptr, arrived_flat)
    )
    return data, int(h2d_bytes)


def _tensor_input_digest(data) -> str:
    return _digest_arrays(
        x=data.x.detach().cpu().numpy(),
        edge_index=data.edge_index.detach().cpu().numpy(),
        edge_attr=data.edge_attr.detach().cpu().numpy(),
        batch=data.batch.detach().cpu().numpy(),
        ptr=data.ptr.detach().cpu().numpy(),
    )


def _make_runtime(checkpoint_path: str, device: str):
    from mapf_cuda.models.checkpoints import load_magat_checkpoint
    from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter

    runtime = MAGATRuntimeAdapter(device=device)
    load_magat_checkpoint(runtime, checkpoint_path, device)
    runtime.model.eval()
    return runtime


def _build_stateful_simulator(
    batch: FrozenTransitionBatch, *, device: str, max_episode_steps: int | None = None
):
    import torch
    import grid_world_cpp as ext

    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator

    grids, extents = stack_grids_for_compiled_simulator(
        list(batch.grids), device=device
    )
    simulator = ext.GridWorldSimulator(
        grids,
        batch.num_agents,
        0,
        int(grids.shape[0] * grids.shape[1] * grids.shape[2]),
        20260720,
        "standard_mapf",
        int(batch.horizon if max_episode_steps is None else max_episode_steps),
    )
    simulator.set_real_map_extents(extents)
    simulator.load_state(
        torch.as_tensor(batch.positions.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.goals.astype(np.int16), device=device).contiguous(),
        torch.as_tensor(batch.arrived.astype(np.uint8), device=device).contiguous(),
        torch.zeros(batch.num_envs, dtype=torch.int32, device=device),
    )
    simulator.pyg_builder_mode = "local_gather"
    simulator.pyg_local_gather_impl = "auto"
    return simulator


def _frozen_state_parity(runtime, batch: FrozenTransitionBatch, *, device: str):
    import torch

    cpu_set = _CpuEpisodeSet(batch, batch.horizon)
    cpu_data, _ = _batch_references(
        cpu_set.build_references(), cpu_set.arrived, device=device
    )
    simulator = _build_stateful_simulator(batch, device=device)
    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()
    gpu_data = runtime.batch_builder.view_stateful_outputs(
        simulator, materialize_edges=True
    )
    torch.cuda.synchronize(torch.device(device))
    input_max_error = float(torch.max(torch.abs(cpu_data.x - gpu_data.x)).item())
    # The goal-energy channel is normalized independently by NumPy and CUDA.
    # Both start from identical uint8 distances, but float32 division can differ
    # by one ULP; require numerical identity rather than byte identity here.
    if not torch.allclose(cpu_data.x, gpu_data.x, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "A4 CPU/GPU model input parity failed for node features: "
            f"max_abs={input_max_error}"
        )
    if not torch.equal(cpu_data.edge_index, gpu_data.edge_index):
        raise RuntimeError("A4 CPU/GPU model input parity failed for graph edges")
    if not torch.equal(cpu_data.edge_attr, gpu_data.edge_attr):
        raise RuntimeError("A4 CPU/GPU model input parity failed for edge attributes")
    with torch.no_grad():
        cpu_logits = runtime.model(cpu_data.x, cpu_data)
        gpu_logits = runtime.model(gpu_data.x, gpu_data)
    torch.cuda.synchronize(torch.device(device))
    max_error = float(torch.max(torch.abs(cpu_logits - gpu_logits)).item())
    if not torch.allclose(cpu_logits, gpu_logits, rtol=1e-5, atol=1e-3):
        raise RuntimeError(f"A4 CPU/GPU logit parity failed: max_abs={max_error}")
    cpu_actions = cpu_logits.argmax(dim=-1)
    gpu_actions = gpu_logits.argmax(dim=-1)
    if not torch.equal(cpu_actions, gpu_actions):
        raise RuntimeError("A4 CPU/GPU argmax action parity failed")
    return {
        "state_sha256": batch.semantic_sha256,
        "model_input_sha256": _tensor_input_digest(gpu_data),
        "logits_sha256": _digest_arrays(logits=gpu_logits.detach().cpu().numpy()),
        "actions_sha256": _digest_arrays(actions=gpu_actions.detach().cpu().numpy()),
        "max_abs_model_input_error": input_max_error,
        "max_abs_logit_error": max_error,
    }


def _quality_metrics(result: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    arrival_times = np.asarray(result["arrival_times"], dtype=np.int32)
    arrived = np.asarray(result["arrived"], dtype=np.bool_)
    success = np.all(arrived, axis=1)
    per_episode_soc = arrival_times.sum(axis=1, dtype=np.int64)
    per_episode_makespan = arrival_times.max(axis=1)
    requested = int(result["requested_moves"])
    blocked = int(result["blocked_moves"])
    return {
        "csr": float(success.mean()),
        "isr": float(arrived.mean()),
        "soc_mean": float(per_episode_soc.mean()),
        "soc_total": int(per_episode_soc.sum()),
        "makespan_mean": float(per_episode_makespan.mean()),
        "success_conditioned_makespan_mean": (
            None if not np.any(success) else float(per_episode_makespan[success].mean())
        ),
        "blocked_no_move_rate": 0.0 if requested == 0 else blocked / requested,
        "truncation_rate": float((~success).mean()),
        "horizon": int(horizon),
    }


def _final_state_digest(result: Mapping[str, Any], goals: np.ndarray) -> str:
    return _digest_arrays(
        positions=np.asarray(result["positions"], dtype=np.uint16),
        goals=np.asarray(goals, dtype=np.uint16),
        arrived=np.asarray(result["arrived"], dtype=np.bool_),
        terminated=np.asarray(result["terminated"], dtype=np.bool_),
        truncated=np.asarray(result["truncated"], dtype=np.bool_),
        step_counts=np.asarray(result["step_counts"], dtype=np.int32),
    )


def _gpu_utilization_from_health(health: Mapping[str, Any]) -> float | None:
    values = [
        float(event["gpu"]["utilization_pct"])
        for event in health.get("events", [])
        if isinstance(event.get("gpu"), Mapping)
        and event["gpu"].get("utilization_pct") is not None
    ]
    return None if not values else float(sum(values) / len(values))


def _start_health(process_provider):
    from expert.benchmark_health import HealthCollector
    from mapf_cuda.observability.pipeline import query_gpu_snapshot

    collector = HealthCollector(
        process_provider=process_provider,
        gpu_provider=lambda: {
            **query_gpu_snapshot(device_index=0),
            "progress_counter": 0,
        },
        pipeline_provider=lambda: {
            "applicable": False,
            "not_applicable_reason": "A4 deployment has no training ring or DMA pipeline",
        },
        sample_interval_s=1.0,
    )
    collector.start()
    return collector


def _finish_health(collector, *, expected_workers: int):
    from expert.benchmark_health import build_live_health_report

    collector.sample_once(reason="deployment_complete")
    collector.stop()
    return build_live_health_report(
        collector.events,
        expected_workers=int(expected_workers),
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
        sample_interval_s=1.0,
    )


def _common_performance_metrics(
    *, wall_s: float, batch: FrozenTransitionBatch, result: Mapping[str, Any],
    h2d_bytes: int, d2h_bytes: int, peak_gpu_memory_bytes: int,
    cpu_process_s: float, health: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "total_wall_s": float(wall_s),
        "executed_steps": int(result["executed_steps"]),
        "wall_ms_per_batched_step": (
            None
            if int(result["executed_steps"]) == 0
            else float(wall_s / result["executed_steps"] * 1000.0)
        ),
        "env_steps": int(result["env_steps"]),
        "agent_steps": int(result["agent_steps"]),
        "env_steps_s": float(result["env_steps"] / wall_s),
        "agent_steps_s": float(result["agent_steps"] / wall_s),
        "h2d_bytes": int(h2d_bytes),
        "d2h_bytes": int(d2h_bytes),
        "peak_host_memory_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
        "cpu_process_utilization_pct": float(cpu_process_s / wall_s * 100.0),
        "gpu_utilization_pct": _gpu_utilization_from_health(health),
        "num_envs": batch.num_envs,
        "num_agents": batch.num_agents,
    }


def _default_deployment_executor(**kwargs) -> Mapping[str, Any]:
    return _execute_magat_mode(**kwargs)


def _run_cpu_single_engine(
    runtime,
    batch: FrozenTransitionBatch,
    *,
    max_episode_steps: int,
    device: str,
) -> dict[str, Any]:
    import torch

    episodes = _CpuEpisodeSet(batch, max_episode_steps)
    trajectory = np.zeros(
        (max_episode_steps, batch.num_envs, batch.num_agents), dtype=np.uint8
    )
    collector = _start_health(
        lambda: {"parent_pid": os.getpid(), "workers": []}
    )
    h2d_bytes = d2h_bytes = 0
    cpu_builder_s = model_s = cpu_transition_s = 0.0
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    cpu_started = time.process_time()
    wall_started = time.perf_counter()
    executed_steps = 0
    for step in range(max_episode_steps):
        if np.all(episodes.terminated | episodes.truncated):
            break
        stage_started = time.perf_counter()
        references = episodes.build_references()
        cpu_builder_s += time.perf_counter() - stage_started
        data, copied = _batch_references(references, episodes.arrived, device=device)
        h2d_bytes += copied
        stage_started = time.perf_counter()
        with torch.no_grad():
            logits = runtime.model(data.x, data)
        actions = logits.argmax(dim=-1).reshape(batch.num_envs, batch.num_agents)
        actions_host = actions.to(torch.uint8).cpu().numpy()
        torch.cuda.synchronize(torch.device(device))
        model_s += time.perf_counter() - stage_started
        d2h_bytes += int(actions_host.nbytes)
        trajectory[step] = actions_host
        stage_started = time.perf_counter()
        episodes.step(actions_host, step)
        cpu_transition_s += time.perf_counter() - stage_started
        executed_steps = step + 1
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started
    cpu_process_s = time.process_time() - cpu_started
    result = episodes.result()
    result["executed_steps"] = int(executed_steps)
    health = _finish_health(collector, expected_workers=0)
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
    return {
        **_quality_metrics(result, max_episode_steps),
        **_common_performance_metrics(
            wall_s=wall_s,
            batch=batch,
            result=result,
            h2d_bytes=h2d_bytes,
            d2h_bytes=d2h_bytes,
            peak_gpu_memory_bytes=peak_gpu,
            cpu_process_s=cpu_process_s,
            health=health,
        ),
        "accepted": bool(health["validation"]["valid"]),
        "observed_worker_pids": [],
        "health_report": health,
        "cpu_builder_s": cpu_builder_s,
        "model_and_action_readback_s": model_s,
        "cpu_transition_s": cpu_transition_s,
        "trajectory_action_sha256": _digest_arrays(
            actions=trajectory[:executed_steps]
        ),
        "final_state_sha256": _final_state_digest(result, batch.goals),
    }


def _run_gpu_stateful_engine(
    runtime,
    batch: FrozenTransitionBatch,
    *,
    max_episode_steps: int,
    device: str,
) -> dict[str, Any]:
    import torch

    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=max_episode_steps
    )
    trajectory = torch.empty(
        (max_episode_steps, batch.num_envs, batch.num_agents),
        dtype=torch.uint8,
        device=device,
    )
    arrival_times = torch.full(
        (batch.num_envs, batch.num_agents),
        int(max_episode_steps),
        dtype=torch.int32,
        device=device,
    )
    arrival_times.masked_fill_(simulator.arrived.to(torch.bool), 0)
    requested_moves = torch.zeros((), dtype=torch.int64, device=device)
    blocked_moves = torch.zeros((), dtype=torch.int64, device=device)
    env_steps = torch.zeros((), dtype=torch.int64, device=device)
    agent_steps = torch.zeros((), dtype=torch.int64, device=device)
    previous_x = torch.empty_like(simulator.cur_x)
    previous_y = torch.empty_like(simulator.cur_y)
    collector = _start_health(
        lambda: {"parent_pid": os.getpid(), "workers": []}
    )
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cpu_started = time.process_time()
    start_event.record()
    wall_started = time.perf_counter()
    executed_steps = 0
    for step in range(max_episode_steps):
        active_env = ~(simulator.terminated.to(torch.bool) | simulator.truncated.to(torch.bool))
        active_agents = (
            ~simulator.arrived.to(torch.bool)
            & active_env.reshape(batch.num_envs, 1)
        )
        env_steps.add_(active_env.sum())
        agent_steps.add_(active_agents.sum())
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
        trajectory[step].copy_(actions)
        requested = active_agents & actions.ne(0)
        requested_moves.add_(requested.sum())
        simulator.update_actions(actions)
        simulator.step_sim_only()
        moved = (simulator.cur_x != previous_x) | (simulator.cur_y != previous_y)
        blocked_moves.add_((requested & ~moved).sum())
        newly_arrived = simulator.arrived.to(torch.bool) & active_agents
        arrival_times.masked_fill_(newly_arrived, int(step) + 1)
        executed_steps = step + 1
        if bool(
            torch.all(
                simulator.terminated.to(torch.bool)
                | simulator.truncated.to(torch.bool)
            ).item()
        ):
            break
    end_event.record()
    torch.cuda.synchronize(torch.device(device))
    wall_s = time.perf_counter() - wall_started
    gpu_total_ms = float(start_event.elapsed_time(end_event))
    cpu_process_s = time.process_time() - cpu_started
    health = _finish_health(collector, expected_workers=0)
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))
    final_positions = torch.stack((simulator.cur_x, simulator.cur_y), dim=-1)
    result = {
        "arrival_times": arrival_times.cpu().numpy(),
        "arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "terminated": simulator.terminated.to(torch.bool).cpu().numpy(),
        "truncated": simulator.truncated.to(torch.bool).cpu().numpy(),
        "step_counts": simulator.step_counts.cpu().numpy(),
        "positions": final_positions.to(torch.uint16).cpu().numpy(),
        "requested_moves": int(requested_moves.item()),
        "blocked_moves": int(blocked_moves.item()),
        "env_steps": int(env_steps.item()),
        "agent_steps": int(agent_steps.item()),
        "executed_steps": int(executed_steps),
    }
    trajectory_host = trajectory[:executed_steps].cpu().numpy()
    return {
        **_quality_metrics(result, max_episode_steps),
        **_common_performance_metrics(
            wall_s=wall_s,
            batch=batch,
            result=result,
            h2d_bytes=0,
            d2h_bytes=executed_steps,
            peak_gpu_memory_bytes=peak_gpu,
            cpu_process_s=cpu_process_s,
            health=health,
        ),
        "gpu_total_ms": gpu_total_ms,
        "accepted": bool(health["validation"]["valid"]),
        "observed_worker_pids": [],
        "health_report": health,
        "trajectory_action_sha256": _digest_arrays(actions=trajectory_host),
        "final_state_sha256": _final_state_digest(result, batch.goals),
        "resident_loop_h2d_bytes": 0,
        "resident_loop_d2h_bytes": int(executed_steps),
        "diagnostic_post_loop_d2h_bytes": int(
            trajectory_host.nbytes
            + sum(np.asarray(result[key]).nbytes for key in (
                "arrival_times", "arrived", "terminated", "truncated",
                "step_counts", "positions",
            ))
        ),
    }


def _execute_magat_mode(
    *,
    mode: str,
    checkpoint_path: str,
    checkpoint_info: Mapping[str, Any],
    batch: FrozenTransitionBatch,
    max_episode_steps: int,
    device: str,
    num_workers: int,
) -> Mapping[str, Any]:
    del checkpoint_info
    runtime = _make_runtime(checkpoint_path, device)
    parity = _frozen_state_parity(runtime, batch, device=device)
    if mode == "cpu_single":
        result = _run_cpu_single_engine(
            runtime,
            batch,
            max_episode_steps=max_episode_steps,
            device=device,
        )
    elif mode == "gpu_stateful":
        result = _run_gpu_stateful_engine(
            runtime,
            batch,
            max_episode_steps=max_episode_steps,
            device=device,
        )
    elif mode == "cpu_mp_gpu_model":
        from expert.a4_cpu_mp import run_cpu_mp_engine

        result = run_cpu_mp_engine(
            runtime,
            batch,
            max_episode_steps=max_episode_steps,
            device=device,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"unsupported A4 mode {mode!r}")
    return {**result, "parity": parity}

execute_magat_mode = _execute_magat_mode


__all__ = ["execute_magat_mode"]
