"""Matched JAX/Torch complete-step control for E4b.

The control deliberately composes a JAX/XLA materialized observation path with
eager PyTorch graph construction while keeping state, model inputs, policy
inference, actions, and the transition on the accelerator.  It is a causal
control for MAPF-CUDA's compact resident representation, not a pure-JAX MAGAT
implementation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Iterator

import numpy as np


E4B_PROFILE_RANGE_PREFIX = "e4b.profile"
E4B_STAGE_RANGE_PREFIX = "e4b.stage"


@contextmanager
def _optional_nvtx_range(
    enabled: bool,
    name: str,
    *,
    nvtx_api=None,
) -> Iterator[None]:
    """Push one diagnostic NVTX range without changing synchronization.

    Formal E4b rows leave ``enabled`` false.  The injectable API keeps the
    balance/error behavior unit-testable without importing CUDA in tests.
    """

    if not enabled:
        yield
        return
    if nvtx_api is None:
        import torch

        nvtx_api = torch.cuda.nvtx
    nvtx_api.range_push(name)
    try:
        yield
    finally:
        nvtx_api.range_pop()


@contextmanager
def _optional_cuda_profiler(
    enabled: bool,
    *,
    cudart_api=None,
) -> Iterator[None]:
    """Bound Nsight collection to the diagnostic step without synchronizing."""

    if not enabled:
        yield
        return
    if cudart_api is None:
        import torch

        cudart_api = torch.cuda.cudart()
    cudart_api.cudaProfilerStart()
    try:
        yield
    finally:
        cudart_api.cudaProfilerStop()


@dataclass(frozen=True)
class EagerBatchStatics:
    """Shape-only PyG fields that are constant across policy steps."""

    batch: Any
    ptr: Any
    y: Any
    terminated: Any


def make_eager_batch_statics(
    num_envs: int,
    num_agents: int,
    *,
    device: str = "cuda:0",
) -> EagerBatchStatics:
    import torch

    num_nodes = int(num_envs) * int(num_agents)
    return EagerBatchStatics(
        batch=torch.arange(
            int(num_envs), device=device, dtype=torch.int64
        ).repeat_interleave(int(num_agents)),
        ptr=torch.arange(
            0,
            num_nodes + 1,
            int(num_agents),
            device=device,
            dtype=torch.int64,
        ),
        y=torch.zeros(num_nodes, device=device, dtype=torch.int64),
        terminated=torch.zeros(num_nodes, device=device, dtype=torch.bool),
    )


def build_eager_magat_batch(
    observations,
    positions,
    arrived,
    *,
    communication_radius: float = 7.0,
    statics: EagerBatchStatics | None = None,
):
    """Materialize the official MAGAT graph with eager PyTorch operations."""

    import torch

    from expert.fixed_magat_plus_runtime import RuntimePyGBatch

    if observations.ndim != 5:
        raise ValueError("observations must have shape [E, A, C, H, W]")
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [E, A, 2]")
    if arrived.shape != positions.shape[:2]:
        raise ValueError("arrived must have shape [E, A]")
    if tuple(observations.shape[:2]) != tuple(positions.shape[:2]):
        raise ValueError("observation and position batches must align")
    if not observations.is_contiguous():
        raise ValueError("DLPack observations must be contiguous")
    if not positions.is_contiguous():
        raise ValueError("DLPack positions must be contiguous")

    num_envs, num_agents = map(int, positions.shape[:2])
    positions_i32 = positions.to(dtype=torch.int32)
    deltas = positions_i32[:, :, None, :] - positions_i32[:, None, :, :]
    distance_sq = (deltas * deltas).sum(dim=-1)
    radius_sq = float(communication_radius) * float(communication_radius)
    adjacency = (distance_sq > 0) & (distance_sq.to(torch.float32) <= radius_sq)
    # torch.nonzero on [E,A,A] is lexicographic: environment, source, target.
    # That matches np.argwhere in the official oracle and the CUDA prefix order.
    triples = torch.nonzero(adjacency, as_tuple=False)
    env_index, source, target = triples.unbind(dim=1)
    global_source = env_index * num_agents + source
    global_target = env_index * num_agents + target
    edge_index = torch.stack((global_source, global_target), dim=0).to(torch.int64)
    edge_delta = deltas[env_index, source, target].to(torch.float32)
    edge_attr = torch.cat(
        (edge_delta, edge_delta.abs().sum(dim=1, keepdim=True)), dim=1
    )

    if statics is None:
        statics = make_eager_batch_statics(
            num_envs, num_agents, device=str(observations.device)
        )
    num_edges = torch.full(
        (1,), int(edge_index.shape[1]), device=observations.device, dtype=torch.int64
    )
    return RuntimePyGBatch(
        x=observations.reshape(
            num_envs * num_agents,
            observations.shape[2],
            observations.shape[3],
            observations.shape[4],
        ),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_index_storage=edge_index,
        edge_attr_storage=edge_attr,
        num_edges=num_edges,
        batch=statics.batch,
        ptr=statics.ptr,
        y=statics.y,
        terminated=statics.terminated,
        arrived=arrived.reshape(-1).to(torch.bool),
    )


def _torch_from_jax(value, *, device: str = "cuda:0"):
    import torch

    return torch.utils.dlpack.from_dlpack(value, device=device, copy=False)


def _jax_from_torch(value):
    import jax

    return jax.dlpack.from_dlpack(value, copy=False)


def _hash_arrays(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _make_config(batch, *, max_episode_steps: int | None = None):
    from baselines.jax_sim_baseline import EnvConfig

    return EnvConfig(
        height=int(batch.grids.shape[1]),
        width=int(batch.grids.shape[2]),
        num_agents=int(batch.num_agents),
        max_steps=int(batch.horizon),
        fov_radius=5,
        clamp_value=1.0,
        energy_iters=254,
        task_mode="standard_mapf",
        max_episode_steps=int(
            batch.horizon if max_episode_steps is None else max_episode_steps
        ),
    )


def _device_inputs(batch, config):
    import jax
    import jax.numpy as jnp

    from baselines.jax_sim_baseline import reset_frozen_jit

    grids = jax.device_put(np.asarray(batch.grids, dtype=np.int32))
    positions = jax.device_put(np.asarray(batch.positions, dtype=np.int32))
    goals = jax.device_put(np.asarray(batch.goals, dtype=np.int32))
    arrived = jax.device_put(np.asarray(batch.arrived, dtype=np.bool_))
    rng_keys = jax.random.split(jax.random.PRNGKey(20260723), batch.num_envs)
    state = reset_frozen_jit(
        config, positions, goals, arrived, grids, rng_keys
    )
    jax.block_until_ready((grids, positions, goals, arrived, state))
    return grids, positions, goals, arrived, state


def _compile_jax_kernels(
    config,
    state,
    grids,
    goals,
    *,
    energy_chunk_envs: int,
):
    """Compile all exact-shape XLA executables without executing the step."""

    import jax
    import jax.numpy as jnp

    from baselines.jax_sim_baseline import (
        build_batched_aligned_observations_from_energy_maps,
        build_batched_energy_maps,
        step_transition_jit,
    )

    energy_jit = jax.jit(build_batched_energy_maps, static_argnums=(0,))
    obs_jit = jax.jit(
        build_batched_aligned_observations_from_energy_maps,
        static_argnums=(0,),
    )
    grid_chunks = tuple(
        grids[start : min(start + energy_chunk_envs, grids.shape[0])]
        for start in range(0, grids.shape[0], energy_chunk_envs)
    )
    goal_chunks = tuple(
        goals[start : min(start + energy_chunk_envs, goals.shape[0])]
        for start in range(0, goals.shape[0], energy_chunk_envs)
    )
    jax.block_until_ready((grid_chunks, goal_chunks))
    chunk_sizes = {int(chunk.shape[0]) for chunk in grid_chunks}
    for chunk_size in sorted(chunk_sizes):
        energy_jit.lower(
            config, grids[:chunk_size], goals[:chunk_size]
        ).compile()
    energy_shape = jax.ShapeDtypeStruct(
        (
            state.pos.shape[0],
            config.num_agents,
            config.height,
            config.width,
        ),
        jnp.uint8,
    )
    obs_jit.lower(config, state, grids, energy_shape).compile()
    actions_shape = jax.ShapeDtypeStruct(
        (state.pos.shape[0], config.num_agents), jnp.int32
    )
    # step_transition_jit wraps an already-jitted function.  Lower the
    # underlying transition once through a small outer jitted callable.
    transition_jit = jax.jit(
        lambda s, a, g: step_transition_jit(config, s, a, g)
    )
    transition_jit.lower(state, actions_shape, grids).compile()
    concat_jit = None
    if len(grid_chunks) > 1:
        concat_jit = jax.jit(lambda *parts: jnp.concatenate(parts, axis=0))
        energy_parts = tuple(
            jax.ShapeDtypeStruct(
                (
                    int(chunk.shape[0]),
                    config.num_agents,
                    config.height,
                    config.width,
                ),
                jnp.uint8,
            )
            for chunk in grid_chunks
        )
        concat_jit.lower(*energy_parts).compile()
    return (
        energy_jit,
        obs_jit,
        transition_jit,
        grid_chunks,
        goal_chunks,
        concat_jit,
    )


def _build_energy_cache_timed(
    energy_jit,
    config,
    grid_chunks,
    goal_chunks,
    concat_jit,
):
    import jax
    import jax.numpy as jnp

    parts = []
    for grid_chunk, goal_chunk in zip(grid_chunks, goal_chunks, strict=True):
        part = energy_jit(config, grid_chunk, goal_chunk)
        # Bound live BFS temporaries.  This synchronization is part of the
        # measured baseline rather than hidden in setup.
        jax.block_until_ready(part)
        parts.append(part)
    energy_maps = parts[0] if len(parts) == 1 else concat_jit(*parts)
    return energy_maps


def _jax_torch_policy_step(
    runtime,
    state,
    grids,
    energy_maps,
    obs_jit,
    transition_jit,
    statics,
    *,
    device: str,
    communication_radius: float,
):
    """Enqueue one observation→model→argmax→transition chain."""

    import torch

    observations = obs_jit(state, grids, energy_maps)
    observations_t = _torch_from_jax(observations, device=device)
    positions_t = _torch_from_jax(state.pos, device=device)
    arrived_t = _torch_from_jax(state.arrived, device=device)
    data = build_eager_magat_batch(
        observations_t,
        positions_t,
        arrived_t,
        communication_radius=communication_radius,
        statics=statics,
    )
    with torch.no_grad():
        logits = runtime.model(data.x, data)
    actions_t = logits.argmax(dim=-1).reshape(
        state.pos.shape[0], state.pos.shape[1]
    ).to(torch.int32).contiguous()
    actions_j = _jax_from_torch(actions_t)
    new_state, _, _, _, _ = transition_jit(state, actions_j, grids)
    return new_state, actions_j


def _jax_torch_policy_step_profiled(
    runtime,
    state,
    grids,
    energy_maps,
    obs_jit,
    transition_jit,
    statics,
    *,
    device: str,
    communication_radius: float,
):
    """Enqueue the same policy step with diagnostic-only NVTX leaves."""

    import torch

    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.view_observation"
    ):
        observations = obs_jit(state, grids, energy_maps)
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.framework_bridge_in"
    ):
        observations_t = _torch_from_jax(observations, device=device)
        positions_t = _torch_from_jax(state.pos, device=device)
        arrived_t = _torch_from_jax(state.arrived, device=device)
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.graph_build"
    ):
        data = build_eager_magat_batch(
            observations_t,
            positions_t,
            arrived_t,
            communication_radius=communication_radius,
            statics=statics,
        )
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.model"
    ):
        with torch.no_grad():
            logits = runtime.model(data.x, data)
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.argmax"
    ):
        actions_t = logits.argmax(dim=-1).reshape(
            state.pos.shape[0], state.pos.shape[1]
        ).to(torch.int32).contiguous()
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.framework_bridge_out"
    ):
        actions_j = _jax_from_torch(actions_t)
    with _optional_nvtx_range(
        True, f"{E4B_STAGE_RANGE_PREFIX}.transition"
    ):
        new_state, _, _, _, _ = transition_jit(state, actions_j, grids)
    return new_state, actions_j


def run_jax_torch_complete(
    runtime,
    batch,
    *,
    device: str = "cuda:0",
    energy_chunk_envs: int = 16,
    communication_radius: float = 7.0,
) -> dict[str, Any]:
    """Measure one cold complete decision step with a matched GPU control."""

    import jax
    import torch

    from expert.a4_runtime import _digest_arrays

    # Retained E4 runs exactly one fixed step and constructs both environments
    # with that measured horizon.  Match its post-step status contract too.
    config = _make_config(batch, max_episode_steps=1)
    grids, _, goals, _, state = _device_inputs(batch, config)
    (
        energy_jit,
        obs_jit,
        transition_jit,
        grid_chunks,
        goal_chunks,
        concat_jit,
    ) = _compile_jax_kernels(
        config,
        state,
        grids,
        goals,
        energy_chunk_envs=int(energy_chunk_envs),
    )
    statics = make_eager_batch_statics(
        batch.num_envs, batch.num_agents, device=device
    )
    # Bind the static config outside the timer so Python orchestration matches
    # the already-compiled transition closure.
    obs_call = lambda s, g, e: obs_jit(config, s, g, e)

    jax.block_until_ready(state)
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
    energy_started = started
    energy_maps = _build_energy_cache_timed(
        energy_jit,
        config,
        grid_chunks,
        goal_chunks,
        concat_jit,
    )
    jax.block_until_ready(energy_maps)
    energy_ready = time.perf_counter()
    new_state, actions_j = _jax_torch_policy_step(
        runtime,
        state,
        grids,
        energy_maps,
        obs_call,
        transition_jit,
        statics,
        device=device,
        communication_radius=communication_radius,
    )
    jax.block_until_ready(new_state)
    elapsed = time.perf_counter() - started
    peak_torch = int(torch.cuda.max_memory_allocated(torch.device(device)))

    actions_host = np.asarray(jax.device_get(actions_j), dtype=np.uint8)
    positions_host = np.asarray(jax.device_get(new_state.pos), dtype=np.uint16)
    arrived_host = np.asarray(jax.device_get(new_state.arrived), dtype=np.bool_)
    terminated_host = np.asarray(
        jax.device_get(new_state.terminated), dtype=np.bool_
    )
    truncated_host = np.asarray(jax.device_get(new_state.truncated), dtype=np.bool_)
    step_counts_host = np.asarray(
        jax.device_get(new_state.step_count), dtype=np.int32
    )
    memory_stats = jax.devices()[0].memory_stats() or {}
    return {
        "schema_version": "e4b-complete-step-v1",
        "status": "ok",
        "mode": "jax_torch_complete",
        "timing_phase": "cold_first",
        "num_envs": int(batch.num_envs),
        "num_agents": int(batch.num_agents),
        "executed_steps": 1,
        "wall_s": float(elapsed),
        "wall_ms_per_batched_step": float(elapsed * 1000.0),
        "env_steps_s": float(batch.num_envs / elapsed),
        "agent_steps_s": float(batch.num_envs * batch.num_agents / elapsed),
        "energy_preparation_ms": float((energy_ready - energy_started) * 1000.0),
        "post_energy_complete_ms": float((started + elapsed - energy_ready) * 1000.0),
        "energy_chunk_envs": int(energy_chunk_envs),
        "energy_cache_dtype": str(energy_maps.dtype),
        "action_selection": "argmax",
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "dlpack_copy": False,
        "peak_torch_allocated_bytes": peak_torch,
        "jax_memory_stats_diagnostic": {
            str(key): int(value) if isinstance(value, (int, np.integer)) else value
            for key, value in memory_stats.items()
        },
        "cross_framework_peak_memory_bytes": None,
        "cross_framework_memory_reason": (
            "NVML unavailable; Torch allocator statistics exclude JAX allocations"
        ),
        "input_semantic_sha256": batch.semantic_sha256,
        "trajectory_action_sha256": _digest_arrays(
            actions=actions_host[None, ...]
        ),
        "final_state_sha256": _digest_arrays(
            positions=positions_host,
            goals=np.asarray(batch.goals, dtype=np.uint16),
            arrived=arrived_host,
            terminated=terminated_host,
            truncated=truncated_host,
            step_counts=step_counts_host,
        ),
    }


def run_jax_torch_steady(
    runtime,
    batch,
    *,
    device: str = "cuda:0",
    energy_chunk_envs: int = 16,
    communication_radius: float = 7.0,
    diagnostic_nvtx: bool = False,
    diagnostic_profiler_api: bool = True,
) -> dict[str, Any]:
    """Measure a complete decision after one full-batch cache warm-up step."""

    import jax
    import torch

    from expert.a4_runtime import _digest_arrays

    config = _make_config(batch, max_episode_steps=2)
    grids, _, goals, _, state = _device_inputs(batch, config)
    (
        energy_jit,
        obs_jit,
        transition_jit,
        grid_chunks,
        goal_chunks,
        concat_jit,
    ) = _compile_jax_kernels(
        config,
        state,
        grids,
        goals,
        energy_chunk_envs=int(energy_chunk_envs),
    )
    statics = make_eager_batch_statics(
        batch.num_envs, batch.num_agents, device=device
    )
    obs_call = lambda s, g, e: obs_jit(config, s, g, e)

    setup_started = time.perf_counter()
    energy_maps = _build_energy_cache_timed(
        energy_jit,
        config,
        grid_chunks,
        goal_chunks,
        concat_jit,
    )
    jax.block_until_ready(energy_maps)
    setup_energy_ms = (time.perf_counter() - setup_started) * 1000.0
    warm_state, _ = _jax_torch_policy_step(
        runtime,
        state,
        grids,
        energy_maps,
        obs_call,
        transition_jit,
        statics,
        device=device,
        communication_radius=communication_radius,
    )
    jax.block_until_ready(warm_state)
    torch.cuda.synchronize(torch.device(device))

    torch.cuda.reset_peak_memory_stats(torch.device(device))
    profile_range = (
        f"{E4B_PROFILE_RANGE_PREFIX}.jax_torch_complete.steady_after_one"
    )
    if diagnostic_nvtx:
        with _optional_cuda_profiler(diagnostic_profiler_api):
            with _optional_nvtx_range(True, profile_range):
                started = time.perf_counter()
                new_state, actions_j = _jax_torch_policy_step_profiled(
                    runtime,
                    warm_state,
                    grids,
                    energy_maps,
                    obs_call,
                    transition_jit,
                    statics,
                    device=device,
                    communication_radius=communication_radius,
                )
                with _optional_nvtx_range(
                    True,
                    f"{E4B_STAGE_RANGE_PREFIX}.completion_sync",
                ):
                    jax.block_until_ready(new_state)
                elapsed = time.perf_counter() - started
    else:
        started = time.perf_counter()
        new_state, actions_j = _jax_torch_policy_step(
            runtime,
            warm_state,
            grids,
            energy_maps,
            obs_call,
            transition_jit,
            statics,
            device=device,
            communication_radius=communication_radius,
        )
        jax.block_until_ready(new_state)
        elapsed = time.perf_counter() - started
    peak_torch = int(torch.cuda.max_memory_allocated(torch.device(device)))

    actions_host = np.asarray(jax.device_get(actions_j), dtype=np.uint8)
    final = {
        "positions": np.asarray(jax.device_get(new_state.pos), dtype=np.uint16),
        "goals": np.asarray(batch.goals, dtype=np.uint16),
        "arrived": np.asarray(jax.device_get(new_state.arrived), dtype=np.bool_),
        "terminated": np.asarray(
            jax.device_get(new_state.terminated), dtype=np.bool_
        ),
        "truncated": np.asarray(
            jax.device_get(new_state.truncated), dtype=np.bool_
        ),
        "step_counts": np.asarray(
            jax.device_get(new_state.step_count), dtype=np.int32
        ),
    }
    memory_stats = jax.devices()[0].memory_stats() or {}
    return {
        "schema_version": "e4b-complete-step-v1",
        "status": "ok",
        "mode": "jax_torch_complete",
        "timing_phase": "steady_after_one",
        "num_envs": int(batch.num_envs),
        "num_agents": int(batch.num_agents),
        "warmup_policy_steps": 1,
        "executed_steps": 1,
        "final_state_step": 2,
        "wall_s": float(elapsed),
        "wall_ms_per_batched_step": float(elapsed * 1000.0),
        "env_steps_s": float(batch.num_envs / elapsed),
        "agent_steps_s": float(batch.num_envs * batch.num_agents / elapsed),
        "energy_preparation_ms": 0.0,
        "post_energy_complete_ms": float(elapsed * 1000.0),
        "setup_energy_preparation_ms": float(setup_energy_ms),
        "energy_chunk_envs": int(energy_chunk_envs),
        "energy_cache_dtype": str(energy_maps.dtype),
        "action_selection": "argmax",
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "dlpack_copy": False,
        "peak_torch_allocated_bytes": peak_torch,
        "jax_memory_stats_diagnostic": {
            str(key): int(value) if isinstance(value, (int, np.integer)) else value
            for key, value in memory_stats.items()
        },
        "cross_framework_peak_memory_bytes": None,
        "cross_framework_memory_reason": (
            "NVML unavailable; Torch allocator statistics exclude JAX allocations"
        ),
        "diagnostic_nvtx_enabled": bool(diagnostic_nvtx),
        "diagnostic_profiler_api_enabled": bool(
            diagnostic_nvtx and diagnostic_profiler_api
        ),
        "diagnostic_nvtx_profile_range": (
            profile_range if diagnostic_nvtx else None
        ),
        "input_semantic_sha256": batch.semantic_sha256,
        "trajectory_action_sha256": _digest_arrays(
            actions=actions_host[None, ...]
        ),
        "final_state_sha256": _digest_arrays(**final),
    }


def run_cuda_resident_complete(
    runtime,
    batch,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Measure the matched minimal CUDA cold complete-step path.

    Unlike the retained A4 runner, this causal-control path does not launch
    quality-statistic kernels, copy a trajectory buffer, or poll termination
    inside the timed interval.  It times exactly the operations also present in
    the JAX/Torch control.
    """

    import torch

    from expert.a4_runtime import _build_stateful_simulator, _digest_arrays

    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=1
    )
    torch.cuda.synchronize(torch.device(device))
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    started = time.perf_counter()
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
    simulator.update_actions(actions)
    simulator.step_sim_only()
    torch.cuda.synchronize(torch.device(device))
    elapsed = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))

    actions_host = actions.cpu().numpy()
    final = {
        "positions": torch.stack((simulator.cur_x, simulator.cur_y), dim=-1)
        .to(torch.uint16)
        .cpu()
        .numpy(),
        "goals": np.asarray(batch.goals, dtype=np.uint16),
        "arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "terminated": simulator.terminated.to(torch.bool).cpu().numpy(),
        "truncated": simulator.truncated.to(torch.bool).cpu().numpy(),
        "step_counts": simulator.step_counts.to(torch.int32).cpu().numpy(),
    }
    return {
        "schema_version": "e4b-complete-step-v1",
        "status": "ok",
        "mode": "cuda_resident",
        "timing_phase": "cold_first",
        "num_envs": int(batch.num_envs),
        "num_agents": int(batch.num_agents),
        "executed_steps": 1,
        "wall_s": float(elapsed),
        "wall_ms_per_batched_step": float(elapsed * 1000.0),
        "env_steps_s": float(batch.num_envs / elapsed),
        "agent_steps_s": float(batch.num_envs * batch.num_agents / elapsed),
        "action_selection": "argmax",
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "peak_gpu_memory_bytes": peak_gpu,
        "input_semantic_sha256": batch.semantic_sha256,
        "trajectory_action_sha256": _digest_arrays(
            actions=actions_host[None, ...]
        ),
        "final_state_sha256": _digest_arrays(**final),
    }


def run_cuda_resident_steady(
    runtime,
    batch,
    *,
    device: str = "cuda:0",
    diagnostic_nvtx: bool = False,
    diagnostic_profiler_api: bool = True,
) -> dict[str, Any]:
    """Measure the minimal resident path after one full-batch warm-up step."""

    import torch

    from expert.a4_runtime import _build_stateful_simulator, _digest_arrays

    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=2
    )

    # The matched warm-up executes a complete policy step on the same frozen
    # state.  It initializes the static energy cache and advances both paths to
    # the identical deterministic second-step state.
    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()
    warm_data = runtime.batch_builder.view_stateful_outputs(
        simulator, materialize_edges=False
    )
    with torch.no_grad():
        warm_logits = runtime.model(warm_data.x, warm_data)
    warm_actions = warm_logits.argmax(dim=-1).reshape(
        batch.num_envs, batch.num_agents
    ).to(torch.uint8)
    simulator.update_actions(warm_actions)
    simulator.step_sim_only()
    torch.cuda.synchronize(torch.device(device))

    torch.cuda.reset_peak_memory_stats(torch.device(device))
    profile_range = f"{E4B_PROFILE_RANGE_PREFIX}.cuda_resident.steady_after_one"
    # Standard-MAPF goals are immutable.  The warm-up has populated the energy
    # cache, so re-running update_derived_state would add an avoidable
    # device-to-host goal-dirty scalar check that the matched JAX cache does not
    # perform.
    if diagnostic_nvtx:
        with _optional_cuda_profiler(diagnostic_profiler_api):
            with _optional_nvtx_range(True, profile_range):
                started = time.perf_counter()
                with _optional_nvtx_range(
                    True,
                    f"{E4B_STAGE_RANGE_PREFIX}.view_graph_build",
                ):
                    simulator.build_magat_plus_inputs()
                with _optional_nvtx_range(
                    True,
                    f"{E4B_STAGE_RANGE_PREFIX}.framework_adapter",
                ):
                    data = runtime.batch_builder.view_stateful_outputs(
                        simulator, materialize_edges=False
                    )
                with _optional_nvtx_range(
                    True, f"{E4B_STAGE_RANGE_PREFIX}.model"
                ):
                    with torch.no_grad():
                        logits = runtime.model(data.x, data)
                with _optional_nvtx_range(
                    True, f"{E4B_STAGE_RANGE_PREFIX}.argmax"
                ):
                    actions = logits.argmax(dim=-1).reshape(
                        batch.num_envs, batch.num_agents
                    ).to(torch.uint8)
                with _optional_nvtx_range(
                    True, f"{E4B_STAGE_RANGE_PREFIX}.transition"
                ):
                    simulator.update_actions(actions)
                    simulator.step_sim_only()
                with _optional_nvtx_range(
                    True,
                    f"{E4B_STAGE_RANGE_PREFIX}.completion_sync",
                ):
                    torch.cuda.synchronize(torch.device(device))
                elapsed = time.perf_counter() - started
    else:
        started = time.perf_counter()
        simulator.build_magat_plus_inputs()
        data = runtime.batch_builder.view_stateful_outputs(
            simulator, materialize_edges=False
        )
        with torch.no_grad():
            logits = runtime.model(data.x, data)
        actions = logits.argmax(dim=-1).reshape(
            batch.num_envs, batch.num_agents
        ).to(torch.uint8)
        simulator.update_actions(actions)
        simulator.step_sim_only()
        torch.cuda.synchronize(torch.device(device))
        elapsed = time.perf_counter() - started
    peak_gpu = int(torch.cuda.max_memory_allocated(torch.device(device)))

    actions_host = actions.cpu().numpy()
    final = {
        "positions": torch.stack((simulator.cur_x, simulator.cur_y), dim=-1)
        .to(torch.uint16)
        .cpu()
        .numpy(),
        "goals": np.asarray(batch.goals, dtype=np.uint16),
        "arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "terminated": simulator.terminated.to(torch.bool).cpu().numpy(),
        "truncated": simulator.truncated.to(torch.bool).cpu().numpy(),
        "step_counts": simulator.step_counts.to(torch.int32).cpu().numpy(),
    }
    return {
        "schema_version": "e4b-complete-step-v1",
        "status": "ok",
        "mode": "cuda_resident",
        "timing_phase": "steady_after_one",
        "num_envs": int(batch.num_envs),
        "num_agents": int(batch.num_agents),
        "warmup_policy_steps": 1,
        "executed_steps": 1,
        "final_state_step": 2,
        "wall_s": float(elapsed),
        "wall_ms_per_batched_step": float(elapsed * 1000.0),
        "env_steps_s": float(batch.num_envs / elapsed),
        "agent_steps_s": float(batch.num_envs * batch.num_agents / elapsed),
        "action_selection": "argmax",
        "timed_h2d_bytes": 0,
        "timed_d2h_bytes": 0,
        "peak_gpu_memory_bytes": peak_gpu,
        "diagnostic_nvtx_enabled": bool(diagnostic_nvtx),
        "diagnostic_profiler_api_enabled": bool(
            diagnostic_nvtx and diagnostic_profiler_api
        ),
        "diagnostic_nvtx_profile_range": (
            profile_range if diagnostic_nvtx else None
        ),
        "input_semantic_sha256": batch.semantic_sha256,
        "trajectory_action_sha256": _digest_arrays(
            actions=actions_host[None, ...]
        ),
        "final_state_sha256": _digest_arrays(**final),
    }


def run_jax_cuda_parity_gate(
    runtime,
    batch,
    *,
    device: str = "cuda:0",
    communication_radius: float = 7.0,
) -> dict[str, Any]:
    """Validate the hybrid control against the resident CUDA model inputs."""

    import jax
    import jax.numpy as jnp
    import torch

    from baselines.jax_sim_baseline import (
        build_batched_aligned_observations_from_energy_maps,
        build_batched_energy_maps,
        step_transition_jit,
    )
    from expert.a4_runtime import _build_stateful_simulator
    from expert.a4_runtime import _digest_arrays

    if batch.num_envs != 1:
        raise ValueError("parity gate requires exactly one frozen environment")
    config = _make_config(batch, max_episode_steps=1)
    grids, _, goals, _, state = _device_inputs(batch, config)
    energy = jax.jit(build_batched_energy_maps, static_argnums=(0,))(
        config, grids, goals
    )
    observations = jax.jit(
        build_batched_aligned_observations_from_energy_maps,
        static_argnums=(0,),
    )(config, state, grids, energy)
    observations_t = _torch_from_jax(observations, device=device)
    positions_t = _torch_from_jax(state.pos, device=device)
    arrived_t = _torch_from_jax(state.arrived, device=device)
    eager_data = build_eager_magat_batch(
        observations_t,
        positions_t,
        arrived_t,
        communication_radius=communication_radius,
    )

    simulator = _build_stateful_simulator(
        batch, device=device, max_episode_steps=1
    )
    simulator.update_derived_state()
    simulator.build_magat_plus_inputs()
    resident_data = runtime.batch_builder.view_stateful_outputs(
        simulator, materialize_edges=True
    )
    torch.cuda.synchronize(torch.device(device))

    input_error = float((eager_data.x - resident_data.x).abs().max().item())
    if not torch.allclose(eager_data.x, resident_data.x, rtol=0.0, atol=1e-7):
        input_diff = (eager_data.x - resident_data.x).abs()
        channel_max = input_diff.amax(dim=(0, 2, 3)).cpu().tolist()
        channel_mismatch_count = (
            input_diff.gt(1e-7).sum(dim=(0, 2, 3)).cpu().tolist()
        )
        first = torch.nonzero(input_diff.gt(1e-7), as_tuple=False)[0]
        first_index = [int(value) for value in first.cpu().tolist()]
        first_eager = float(eager_data.x[tuple(first)].item())
        first_resident = float(resident_data.x[tuple(first)].item())
        raise RuntimeError(
            "E4b observation parity failed: "
            f"max_abs={input_error}, channel_max={channel_max}, "
            f"channel_mismatch_count={channel_mismatch_count}, "
            f"first_index={first_index}, eager={first_eager}, "
            f"resident={first_resident}"
        )
    if not torch.equal(eager_data.edge_index, resident_data.edge_index):
        raise RuntimeError("E4b edge_index parity failed")
    if not torch.equal(eager_data.edge_attr, resident_data.edge_attr):
        raise RuntimeError("E4b edge_attr parity failed")
    with torch.no_grad():
        eager_logits = runtime.model(eager_data.x, eager_data)
        resident_logits = runtime.model(resident_data.x, resident_data)
    torch.cuda.synchronize(torch.device(device))
    logit_error = float((eager_logits - resident_logits).abs().max().item())
    if not torch.allclose(eager_logits, resident_logits, rtol=1e-5, atol=1e-3):
        raise RuntimeError(f"E4b logit parity failed: max_abs={logit_error}")
    eager_actions = eager_logits.argmax(dim=-1).reshape(1, batch.num_agents)
    resident_actions = resident_logits.argmax(dim=-1).reshape(1, batch.num_agents)
    if not torch.equal(eager_actions, resident_actions):
        raise RuntimeError("E4b argmax parity failed")

    actions_u8 = eager_actions.to(torch.uint8).contiguous()
    simulator.update_actions(actions_u8)
    simulator.step_sim_only()
    new_state, _, _, _, _ = step_transition_jit(
        config,
        state,
        _jax_from_torch(eager_actions.to(torch.int32).contiguous()),
        grids,
    )
    jax.block_until_ready(new_state)
    cuda_positions = torch.stack((simulator.cur_x, simulator.cur_y), dim=-1)
    jax_positions = _torch_from_jax(new_state.pos, device=device).to(torch.uint16)
    if not torch.equal(cuda_positions.to(torch.uint16), jax_positions):
        raise RuntimeError("E4b one-step position parity failed")
    jax_arrived = _torch_from_jax(new_state.arrived, device=device).to(torch.bool)
    if not torch.equal(simulator.arrived.to(torch.bool), jax_arrived):
        raise RuntimeError("E4b one-step arrived parity failed")

    actions_host = eager_actions.to(torch.uint8).cpu().numpy()
    jax_final = {
        "positions": jax.device_get(new_state.pos).astype(np.uint16),
        "goals": np.asarray(batch.goals, dtype=np.uint16),
        "arrived": jax.device_get(new_state.arrived).astype(np.bool_),
        "terminated": jax.device_get(new_state.terminated).astype(np.bool_),
        "truncated": jax.device_get(new_state.truncated).astype(np.bool_),
        "step_counts": jax.device_get(new_state.step_count).astype(np.int32),
    }
    cuda_final = {
        "positions": torch.stack((simulator.cur_x, simulator.cur_y), dim=-1)
        .to(torch.uint16)
        .cpu()
        .numpy(),
        "goals": np.asarray(batch.goals, dtype=np.uint16),
        "arrived": simulator.arrived.to(torch.bool).cpu().numpy(),
        "terminated": simulator.terminated.to(torch.bool).cpu().numpy(),
        "truncated": simulator.truncated.to(torch.bool).cpu().numpy(),
        "step_counts": simulator.step_counts.to(torch.int32).cpu().numpy(),
    }
    jax_digest = _digest_arrays(**jax_final)
    cuda_digest = _digest_arrays(**cuda_final)
    if jax_digest != cuda_digest:
        raise RuntimeError(
            "E4b one-step full final-state parity failed: "
            f"jax={jax_digest}, cuda={cuda_digest}"
        )
    return {
        "status": "passed",
        "num_envs": 1,
        "num_agents": int(batch.num_agents),
        "max_abs_model_input_error": input_error,
        "max_abs_logit_error": logit_error,
        "edge_index_exact": True,
        "edge_attr_exact": True,
        "argmax_exact": True,
        "final_positions_exact": True,
        "full_final_state_exact": True,
        "actions_sha256": _hash_arrays(actions=actions_host),
        "final_state_sha256": jax_digest,
    }


__all__ = [
    "E4B_PROFILE_RANGE_PREFIX",
    "E4B_STAGE_RANGE_PREFIX",
    "EagerBatchStatics",
    "build_eager_magat_batch",
    "make_eager_batch_statics",
    "run_cuda_resident_complete",
    "run_cuda_resident_steady",
    "run_jax_cuda_parity_gate",
    "run_jax_torch_complete",
    "run_jax_torch_steady",
]
