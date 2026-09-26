"""Compact frontier-ring training with topology-specific LaCAM producers.

This module is the production MAGAT training path.  It is intentionally
independent of the historical all-in-one benchmark command.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch

try:
    from pogema import GridConfig, pogema_v0
    from pogema.grid_registry import GRID_STR_REGISTRY
    POGEMA_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    GridConfig = None
    pogema_v0 = None
    GRID_STR_REGISTRY = None
    POGEMA_IMPORT_ERROR = exc

from expert.benchmark_training_health import TrainingHealthLifecycle
from expert.checkpoint_manager import TopKCheckpointManager
from expert.expert_running import (
    FEATURE_DIM,
    ExtremeMAPFPipeline,
    LacamExpertPolicy,
    configure_cpu_thread_limits,
    initial_refresh_flags,
    no_refresh_flags,
    normalize_refresh_flags,
    put_maps_into_registry,
    resolve_expert_timeouts,
)
from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter, MAGAT_LR, MAGAT_LR_END
from expert.profiled_topology_loader import (
    load_named_topology_strings_from_yaml,
    normalize_topology_map_family,
    parse_topology_map_name,
)
from mapf_cuda.simulation.grids import (
    build_stateless_simulator as _build_stateless_simulator_from_grids,
    configure_magat_builder as _configure_magat_builder,
)
from mapf_cuda.simulation.pogema_envs import build_topology_standard_env
from mapf_cuda.observability.pipeline import (
    device_index as _device_index_from_string,
    pipeline_snapshot as _pipeline_health_snapshot,
)

ASYNC_RING_BUFFER_STEPS = 16


def _apply_simulator_builder_policy(
    simulator,
    *,
    pyg_builder_mode: str | None = None,
    pyg_local_gather_impl: str | None = None,
):
    return _configure_magat_builder(
        simulator,
        mode=pyg_builder_mode,
        local_gather_impl=pyg_local_gather_impl,
    )


def _default_async_ring_capacity(full_expert_batch: int, capacity_steps: int = ASYNC_RING_BUFFER_STEPS) -> int:
    return max(full_expert_batch * capacity_steps, full_expert_batch + 1)

def _require_pogema_benchmark_dependencies():
    if POGEMA_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "pogema benchmark dependencies are unavailable; install gymnasium/pogema to use this benchmark mode"
        ) from POGEMA_IMPORT_ERROR

def _normalize_obs_coords_array(observations):
    pos = np.array([obs["global_xy"] for obs in observations], dtype=np.int32)
    tgt = np.array([obs["global_target_xy"] for obs in observations], dtype=np.int32)
    pos -= 5
    tgt -= 5
    return pos, tgt

def _build_step_data(observations, actions, env_id: int = 0, reset_flag=0):
    n = len(observations)
    data = np.zeros((n, FEATURE_DIM), dtype=np.uint16)
    data[:, 0] = env_id
    data[:, 1] = np.arange(n, dtype=np.uint16)
    pos, tgt = _normalize_obs_coords_array(observations)
    data[:, 2] = pos[:, 0]
    data[:, 3] = pos[:, 1]
    data[:, 4] = tgt[:, 0]
    data[:, 5] = tgt[:, 1]
    data[:, 6] = actions
    data[:, 7] = normalize_refresh_flags(reset_flag, n)
    return data

def _build_pogema_topology_map_env(
    *,
    num_agents: int,
    map_name: str,
    seed: int,
    max_episode_steps: int = 256,
    collision_system: str = "priority",
):
    return build_topology_standard_env(
        num_agents=num_agents,
        map_name=map_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        collision_system=collision_system,
    )

def _select_topology_training_maps(
    maps_path: str,
    num_experts: int,
    map_names: list[str] | tuple[str, ...] | None = None,
    map_families: list[str] | tuple[str, ...] | None = None,
):
    available_maps = load_named_topology_strings_from_yaml(maps_path)
    available_names = list(available_maps.keys())
    put_maps_into_registry(maps_path)
    normalized_families = None
    if map_families is not None:
        normalized_families = {
            normalize_topology_map_family(family) for family in map_families
        }

    if map_names is not None:
        selected_names = list(map_names)
        missing = [name for name in selected_names if name not in available_names]
        if missing:
            raise ValueError(f"Requested topology maps are not present in {maps_path}: {missing}")
        if normalized_families is not None:
            disallowed = [
                name
                for name in selected_names
                if parse_topology_map_name(name).family not in normalized_families
            ]
            if disallowed:
                raise ValueError(
                    "Requested topology maps do not match the requested families: "
                    f"{disallowed}"
                )
    else:
        selected_names = available_names
        if normalized_families is not None:
            selected_names = [
                name
                for name in selected_names
                if parse_topology_map_name(name).family in normalized_families
            ]
        selected_names = selected_names[:num_experts]

    if len(selected_names) < num_experts:
        raise ValueError(
            f"Need at least {num_experts} topology maps from {maps_path}, found {len(selected_names)}"
        )

    return [
        (name, np.array(GRID_STR_REGISTRY[name].obstacles, copy=True))
        for name in selected_names[:num_experts]
    ]

def _resolve_validation_trajectory_paths(validation_trajectories):
    if validation_trajectories is None:
        return None
    resolved = [str(Path(path).expanduser().resolve()) for path in validation_trajectories if path]
    return resolved or None

def _build_checkpoint_validation_evaluator(validation_trajectories, *, device: str):
    resolved = _resolve_validation_trajectory_paths(validation_trajectories)
    if resolved is None:
        return None, None
    from expert.validation_selection import build_frozen_validation_evaluator

    return build_frozen_validation_evaluator(resolved, device=device), resolved

def _open_training_logger(log_file: str | None, *, announce: bool = True):
    if not log_file:
        return None
    resolved_path = os.path.abspath(log_file)
    parent_dir = os.path.dirname(resolved_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    log_fh = open(resolved_path, "a", encoding="utf-8", buffering=1)
    if announce:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_fh.write(f"\n# === training log opened {timestamp} pid={os.getpid()} ===\n")
        log_fh.flush()
    return log_fh

def _close_training_logger(log_fh):
    if log_fh is not None and not log_fh.closed:
        log_fh.close()

def _log_line(log_fh, line: str, *, also_print: bool = True, flush: bool = True):
    if log_fh is not None:
        log_fh.write(line + "\n")
        if flush:
            log_fh.flush()
    if also_print:
        print(line, flush=flush)

def _topology_async_training_expert_worker_loop(
    expert_id: int,
    ring_buffer,
    timing_list=None,
    *,
    maps_path: str,
    map_name: str,
    num_agents: int,
    num_steps: int,
    seed: int,
    max_episode_steps: int,
    expert_timeouts=None,
    health_handle=None,
):
    configure_cpu_thread_limits()
    put_maps_into_registry(maps_path)
    env = _build_pogema_topology_map_env(
        num_agents=num_agents,
        map_name=map_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        collision_system="soft",
    )
    policy = LacamExpertPolicy(timeouts=resolve_expert_timeouts(expert_timeouts))
    policy.reset_states(env)
    observations = env.env.unwrapped._obs()
    refresh_flags = initial_refresh_flags(len(observations))
    policy_s = 0.0
    env_step_s = 0.0
    ringbuffer_write_s = 0.0
    worker_t0 = time.perf_counter()
    if health_handle is not None:
        health_handle.mark_ready()
    expert_call_timeout_s = float(sum(resolve_expert_timeouts(expert_timeouts)))

    for _ in range(num_steps):
        t0 = time.perf_counter()
        if health_handle is not None:
            health_handle.begin_expert_call(timeout_s=expert_call_timeout_s)
        try:
            actions = policy.act(observations)
        finally:
            if health_handle is not None:
                health_handle.finish_expert_call()
        policy_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        step_data = _build_step_data(
            observations,
            actions,
            env_id=expert_id,
            reset_flag=refresh_flags,
        )
        ring_buffer.reserve_and_write(step_data)
        if health_handle is not None:
            health_handle.mark_published()
        ringbuffer_write_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        _, _, terminated, truncated, _ = env.step(actions.astype(np.int64, copy=False))
        if all(terminated) or all(truncated):
            env.reset()
            observations = env.env.unwrapped._obs()
            policy.reset_states(env)
            refresh_flags = initial_refresh_flags(len(observations))
        else:
            observations = env.env.unwrapped._obs()
            refresh_flags = no_refresh_flags(len(observations))
        env_step_s += time.perf_counter() - t0

    if timing_list is not None:
        timing_list.append(
            {
                "expert_id": expert_id,
                "map_name": map_name,
                "policy_s": policy_s,
                "env_step_s": env_step_s,
                "ringbuffer_write_s": ringbuffer_write_s,
                "worker_total_wall_s": time.perf_counter() - worker_t0,
            }
        )
    if health_handle is not None:
        health_handle.mark_complete_and_wait()

def _partition_expert_assignments(
    num_experts: int, num_producer_processes: int
) -> tuple[tuple[int, ...], ...]:
    """Partition fixed logical experts across a variable process fanout."""

    num_experts = int(num_experts)
    num_producer_processes = int(num_producer_processes)
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if not 1 <= num_producer_processes <= num_experts:
        raise ValueError(
            "num_producer_processes must be in [1, num_experts]: "
            f"num_producer_processes={num_producer_processes}, "
            f"num_experts={num_experts}"
        )
    groups = [[] for _ in range(num_producer_processes)]
    for expert_id in range(num_experts):
        groups[expert_id % num_producer_processes].append(expert_id)
    return tuple(tuple(group) for group in groups)

def _topology_async_training_expert_group_worker_loop(
    producer_id: int,
    expert_ids: tuple[int, ...],
    ring_buffer,
    timing_list=None,
    *,
    maps_path: str,
    map_names: tuple[str, ...],
    num_agents: int,
    num_steps: int,
    seed: int,
    max_episode_steps: int,
    expert_timeouts=None,
    health_handle=None,
):
    """Run multiple logical expert environments sequentially in one process."""

    configure_cpu_thread_limits()
    put_maps_into_registry(maps_path)
    resolved_timeouts = resolve_expert_timeouts(expert_timeouts)
    expert_call_timeout_s = float(sum(resolved_timeouts))
    states = []
    for expert_id in expert_ids:
        map_name = map_names[expert_id]
        env = _build_pogema_topology_map_env(
            num_agents=num_agents,
            map_name=map_name,
            seed=seed + expert_id,
            max_episode_steps=max_episode_steps,
            collision_system="soft",
        )
        policy = LacamExpertPolicy(timeouts=resolved_timeouts)
        policy.reset_states(env)
        observations = env.env.unwrapped._obs()
        states.append(
            {
                "expert_id": expert_id,
                "map_name": map_name,
                "env": env,
                "policy": policy,
                "observations": observations,
                "refresh_flags": initial_refresh_flags(len(observations)),
            }
        )

    policy_s = 0.0
    env_step_s = 0.0
    ringbuffer_write_s = 0.0
    worker_t0 = time.perf_counter()
    if health_handle is not None:
        health_handle.mark_ready()

    for _ in range(num_steps):
        for state in states:
            t0 = time.perf_counter()
            if health_handle is not None:
                health_handle.begin_expert_call(timeout_s=expert_call_timeout_s)
            try:
                actions = state["policy"].act(state["observations"])
            finally:
                if health_handle is not None:
                    health_handle.finish_expert_call()
            policy_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            step_data = _build_step_data(
                state["observations"],
                actions,
                env_id=state["expert_id"],
                reset_flag=state["refresh_flags"],
            )
            ring_buffer.reserve_and_write(step_data)
            if health_handle is not None:
                health_handle.mark_published()
            ringbuffer_write_s += time.perf_counter() - t0

            t0 = time.perf_counter()
            _, _, terminated, truncated, _ = state["env"].step(
                actions.astype(np.int64, copy=False)
            )
            if all(terminated) or all(truncated):
                state["env"].reset()
                state["observations"] = state["env"].env.unwrapped._obs()
                state["policy"].reset_states(state["env"])
                state["refresh_flags"] = initial_refresh_flags(
                    len(state["observations"])
                )
            else:
                state["observations"] = state["env"].env.unwrapped._obs()
                state["refresh_flags"] = no_refresh_flags(
                    len(state["observations"])
                )
            env_step_s += time.perf_counter() - t0

    if timing_list is not None:
        timing_list.append(
            {
                "producer_id": int(producer_id),
                "expert_id": int(expert_ids[0]),
                "expert_ids": [int(value) for value in expert_ids],
                "map_names": [map_names[value] for value in expert_ids],
                "policy_s": policy_s,
                "env_step_s": env_step_s,
                "ringbuffer_write_s": ringbuffer_write_s,
                "worker_total_wall_s": time.perf_counter() - worker_t0,
            }
        )
    if health_handle is not None:
        health_handle.mark_complete_and_wait()

def benchmark_topology_async_training_system(
    num_steps: int = 20,
    num_agents: int = 256,
    num_experts: int = 8,
    batch_threshold: int | None = None,
    train_batch_size: int | None = None,
    expert_timeouts=None,
    maps_path: str = "maps/maps.yaml",
    map_names: list[str] | tuple[str, ...] | None = None,
    map_families: list[str] | tuple[str, ...] | None = None,
    seed: int = 42,
    max_episode_steps: int = 256,
    device: str = "cuda:0",
    pyg_builder_mode: str = "local_gather",
    pyg_local_gather_impl: str = "auto",
    log_file: str | None = None,
    loss_log_interval: int = 1000,
    checkpoint_dir: str | None = None,
    top_k_checkpoints: int = 3,
    checkpoint_interval: int = 1000,
    checkpoint_interval_s: float | None = None,
    checkpoint_selection_mode: str = "latest",
    validation_trajectories: list[str] | tuple[str, ...] | None = None,
    async_ring_buffer_steps: int = ASYNC_RING_BUFFER_STEPS,
    lr_start: float = MAGAT_LR,
    lr_end: float = MAGAT_LR_END,
    lr_scheduler: str | None = None,
    scheduler_total_steps: int | None = None,
    grad_clip_norm: float | None = None,
    train_on_arrived_agents: bool = True,
    capture_stage_hashes: bool = False,
    health_sample_interval_s: float = 5.0,
    inline_health_sampling: bool = True,
    num_producer_processes: int | None = None,
    transfer_mode: str = "async",
    max_training_wall_s: float | None = None,
):
    expert_timeouts = resolve_expert_timeouts(expert_timeouts)
    transfer_mode = str(transfer_mode).strip().lower()
    if transfer_mode not in {"async", "sync"}:
        raise ValueError(
            "transfer_mode must be either 'async' or 'sync', "
            f"got {transfer_mode!r}"
        )
    if max_training_wall_s is not None and float(max_training_wall_s) <= 0:
        raise ValueError("max_training_wall_s must be positive")
    if checkpoint_interval_s is not None and float(checkpoint_interval_s) <= 0:
        raise ValueError("checkpoint_interval_s must be positive")
    selected_maps = _select_topology_training_maps(
        maps_path=maps_path,
        num_experts=num_experts,
        map_names=map_names,
        map_families=map_families,
    )
    selected_map_names = [name for name, _ in selected_maps]
    grids = [grid for _, grid in selected_maps]
    if num_producer_processes is None:
        num_producer_processes = num_experts
    producer_groups = _partition_expert_assignments(
        num_experts, num_producer_processes
    )
    simulator = _build_stateless_simulator_from_grids(grids, num_agents=num_agents, device=device)
    resolved_pyg_builder_impl = _apply_simulator_builder_policy(
        simulator,
        pyg_builder_mode=pyg_builder_mode,
        pyg_local_gather_impl=pyg_local_gather_impl,
    )

    full_expert_batch = num_agents * num_experts
    if batch_threshold is None:
        batch_threshold = full_expert_batch
    batch_threshold = max(1, int(batch_threshold))
    if batch_threshold != full_expert_batch:
        raise ValueError(
            "topology_async_training_system currently requires a full expert batch: "
            f"expected batch_threshold={full_expert_batch}, got {batch_threshold}"
        )
    if train_batch_size is None:
        train_batch_size = full_expert_batch
    train_batch_size = max(1, int(train_batch_size))
    if train_batch_size > full_expert_batch:
        raise ValueError(
            f"train_batch_size must be <= full expert batch ({full_expert_batch}), got {train_batch_size}"
        )
    if full_expert_batch % train_batch_size != 0:
        raise ValueError(
            "train_batch_size must divide the full expert batch: "
            f"full_expert_batch={full_expert_batch}, train_batch_size={train_batch_size}"
        )
    if train_batch_size % num_agents != 0:
        raise ValueError(
            "train_batch_size must align to whole environments: "
            f"num_agents={num_agents}, train_batch_size={train_batch_size}"
        )

    capacity = _default_async_ring_capacity(
        full_expert_batch,
        capacity_steps=max(1, int(async_ring_buffer_steps)),
    )
    pipeline = ExtremeMAPFPipeline(
        simulator,
        capacity=capacity,
        batch_threshold=batch_threshold,
        device=device,
    )
    pipeline.initialize()

    runtime_scheduler_total_steps = scheduler_total_steps if lr_scheduler else None
    if runtime_scheduler_total_steps is None and lr_scheduler:
        runtime_scheduler_total_steps = num_steps * full_expert_batch // train_batch_size
    torch.manual_seed(int(seed))
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    runtime = MAGATRuntimeAdapter(
        device=device,
        lr=lr_start,
        lr_end=lr_end,
        lr_scheduler=lr_scheduler,
        scheduler_total_steps=runtime_scheduler_total_steps,
        grad_clip_norm=grad_clip_norm,
        train_on_arrived_agents=train_on_arrived_agents,
    )
    from expert.checkpoint_manager import model_state_sha256

    initial_model_state_sha256 = model_state_sha256(runtime.model)
    selection_mode = str(checkpoint_selection_mode)
    if selection_mode not in {"latest", "validation_accuracy"}:
        raise ValueError(
            "checkpoint_selection_mode must be 'latest' or 'validation_accuracy', "
            f"got {selection_mode!r}"
        )
    validation_evaluator, resolved_validation_trajectories = _build_checkpoint_validation_evaluator(
        validation_trajectories,
        device=device,
    )
    if selection_mode == "validation_accuracy" and validation_evaluator is None:
        raise ValueError(
            "validation_trajectories are required when checkpoint_selection_mode='validation_accuracy'"
        )
    losses = []
    processes = []
    loss_log_interval = max(1, int(loss_log_interval))
    last_loss_log_step = 0
    validation_metrics = None
    validation_elapsed_values = []
    stage_sha256s = []
    log_fh = _open_training_logger(log_file)
    checkpoint_manager = TopKCheckpointManager(
        checkpoint_dir,
        top_k=top_k_checkpoints,
        save_interval_steps=checkpoint_interval,
        mode_label="topology_async_training_system",
        logger=lambda line: _log_line(log_fh, line),
        selection_mode=selection_mode,
        validation_evaluator=validation_evaluator,
    )
    deadline_seconds = max(5.0, num_steps * num_experts * 2.0)
    total_samples = num_steps * num_agents * num_experts
    samples_processed = 0
    consumer_wait_s = 0.0
    consumer_dma_sync_s = 0.0
    consumer_gpu_builder_s = 0.0
    consumer_train_step_s = 0.0
    manager = mp.Manager()
    worker_timings = manager.list()
    worker_timing_rows = []
    health_report = None
    wall_budget_reached = False
    run_completed = False
    dma_thread = None
    health_lifecycle = TrainingHealthLifecycle(
        expected_workers=num_producer_processes,
        pipeline=pipeline,
        block_size=full_expert_batch,
        optimizer_steps_provider=lambda: len(losses),
        samples_processed_provider=lambda: samples_processed,
        gpu_device_index=_device_index_from_string(device),
        expert_timeout_s=float(sum(expert_timeouts)),
        sample_interval_s=float(health_sample_interval_s),
    )

    _log_line(
        log_fh,
        "[start] "
        f"mode=topology_async_training_system "
        f"num_steps={num_steps} "
        f"num_agents={num_agents} "
        f"num_experts={num_experts} "
        f"num_producer_processes={num_producer_processes} "
        f"maps={','.join(selected_map_names)} "
        f"total_samples={total_samples} "
        f"loss_log_interval={loss_log_interval} "
        f"train_on_arrived_agents={train_on_arrived_agents} "
        f"checkpoint_selection_mode={selection_mode} "
        f"transfer_mode={transfer_mode} "
        f"async_ring_buffer_steps={max(1, int(async_ring_buffer_steps))}",
    )
    try:
        for producer_id, expert_ids in enumerate(producer_groups):
            if len(expert_ids) == 1:
                expert_id = expert_ids[0]
                process = mp.Process(
                    target=_topology_async_training_expert_worker_loop,
                    args=(expert_id, pipeline.ring_buffer, worker_timings),
                    kwargs={
                        "maps_path": maps_path,
                        "map_name": selected_map_names[expert_id],
                        "num_agents": num_agents,
                        "num_steps": num_steps,
                        "seed": seed + expert_id,
                        "max_episode_steps": max_episode_steps,
                        "expert_timeouts": expert_timeouts,
                        "health_handle": health_lifecycle.worker_handles[
                            producer_id
                        ],
                    },
                    daemon=True,
                )
            else:
                process = mp.Process(
                    target=_topology_async_training_expert_group_worker_loop,
                    args=(
                        producer_id,
                        expert_ids,
                        pipeline.ring_buffer,
                        worker_timings,
                    ),
                    kwargs={
                        "maps_path": maps_path,
                        "map_names": tuple(selected_map_names),
                        "num_agents": num_agents,
                        "num_steps": num_steps,
                        "seed": seed,
                        "max_episode_steps": max_episode_steps,
                        "expert_timeouts": expert_timeouts,
                        "health_handle": health_lifecycle.worker_handles[
                            producer_id
                        ],
                    },
                    daemon=True,
                )
            processes.append(process)

        health_lifecycle.register_processes(processes)
        health_lifecycle.start()
        if transfer_mode == "async":
            dma_thread = pipeline.dma_worker.start()
        for process in processes:
            process.start()
        health_lifecycle.wait_for_workers_ready(timeout_s=120.0)

        total_t0 = time.perf_counter()
        next_checkpoint_wall_s = (
            None if checkpoint_interval_s is None else float(checkpoint_interval_s)
        )

        def record_checkpoint(
            *, loss_value: float, optimizer_step: int, current_samples: int, force: bool = False
        ):
            nonlocal next_checkpoint_wall_s
            training_elapsed_s = time.perf_counter() - total_t0
            checkpoint_force = bool(force)
            if not checkpoint_force and next_checkpoint_wall_s is not None:
                if training_elapsed_s < next_checkpoint_wall_s:
                    return None
                checkpoint_force = True
            extra_meta = {
                "samples_processed": int(current_samples),
                "total_samples": total_samples,
                "num_agents": num_agents,
                "num_experts": num_experts,
                "train_batch_size": train_batch_size,
                "map_names": selected_map_names,
                "maps_path": maps_path,
                "training_elapsed_s": training_elapsed_s,
                **runtime.training_config(),
            }
            saved = checkpoint_manager.maybe_save(
                loss=float(loss_value),
                runtime=runtime,
                optimizer_step=int(optimizer_step),
                extra_meta=extra_meta,
                force=checkpoint_force,
            )
            if checkpoint_force and next_checkpoint_wall_s is not None and not force:
                now_elapsed_s = time.perf_counter() - total_t0
                while next_checkpoint_wall_s <= now_elapsed_s:
                    next_checkpoint_wall_s += float(checkpoint_interval_s)
            return saved

        deadline = time.perf_counter() + deadline_seconds
        while samples_processed < total_samples:
            if time.perf_counter() > deadline:
                raise TimeoutError("Timed out waiting for topology async training system benchmark.")
            t_wait = time.perf_counter()
            failed_exitcodes = [
                process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed_exitcodes:
                raise RuntimeError(
                    "topology async expert worker failure: "
                    f"exitcodes={failed_exitcodes}"
                )
            pipeline.dma_read_ptr = pipeline.dma_worker.dma_read_ptr
            available = pipeline.dma_read_ptr - pipeline.compute_ptr
            all_experts_done = all(not process.is_alive() for process in processes)
            should_process = pipeline.has_env_aligned_batch(num_experts)
            if all_experts_done and not should_process:
                raise RuntimeError(
                    "topology frontier pipeline exhausted workers before the next complete "
                    "stage became consumable"
                )
            if not should_process:
                time.sleep(0.001)
                consumer_wait_s += time.perf_counter() - t_wait
                continue
            consumer_wait_s += time.perf_counter() - t_wait

            t0 = time.perf_counter()
            if transfer_mode == "sync":
                copied = pipeline.dma_worker.run_once()
                if copied is None:
                    time.sleep(0.001)
                    consumer_wait_s += time.perf_counter() - t0
                    continue
            pipeline.dma_event.synchronize()
            consumer_dma_sync_s += time.perf_counter() - t0
            pipeline.dma_read_ptr = pipeline.dma_worker.dma_read_ptr
            available = pipeline.dma_read_ptr - pipeline.compute_ptr

            raw_batch = pipeline.extract_env_aligned_batch(
                available,
                agents_per_env=num_agents,
                num_envs=num_experts,
            )
            if raw_batch is None:
                time.sleep(0.001)
                continue
            processed_rows = raw_batch.shape[0]
            if capture_stage_hashes:
                stage_sha256s.append(
                    hashlib.sha256(
                        raw_batch.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                )

            t0 = time.perf_counter()
            agents_to_update = pipeline.gpu_handler.filter_reset_agents(raw_batch)
            pipeline.gpu_handler.process_batch(raw_batch, agents_to_update)
            consumer_gpu_builder_s += time.perf_counter() - t0

            batch_losses = []
            batch_arrived_fracs = []
            validation_elapsed_before = checkpoint_manager.last_validation_metrics()
            validation_elapsed_before = (
                None
                if validation_elapsed_before is None
                else validation_elapsed_before.get("validation_elapsed_s")
            )
            if train_batch_size == full_expert_batch:
                batch_arrived_fracs.append(
                    float(raw_batch[:, 2:4].eq(raw_batch[:, 4:6]).all(dim=1).float().mean().item())
                )
                t0 = time.perf_counter()
                loss = runtime.train_step(pipeline.cuda_simulator, raw_batch)
                consumer_train_step_s += time.perf_counter() - t0
                batch_losses.append(float(loss.item()))
                losses.append(batch_losses[-1])
                record_checkpoint(
                    loss_value=losses[-1],
                    optimizer_step=len(losses),
                    current_samples=samples_processed + processed_rows,
                )
            else:
                full_batch = runtime.build_batch(
                    pipeline.cuda_simulator,
                    raw_batch,
                    materialize_edges=True,
                )
                for start_node in range(0, full_expert_batch, train_batch_size):
                    end_node = start_node + train_batch_size
                    sub_batch = runtime.batch_builder.slice_env_aligned_batch(
                        full_batch,
                        start_node=start_node,
                        end_node=end_node,
                        agents_per_env=num_agents,
                    )
                    batch_arrived_fracs.append(float(sub_batch.arrived.float().mean().item()))
                    t0 = time.perf_counter()
                    loss = runtime.train_step_from_batch(sub_batch)
                    consumer_train_step_s += time.perf_counter() - t0
                    batch_losses.append(float(loss.item()))
                    losses.append(batch_losses[-1])
                    record_checkpoint(
                        loss_value=losses[-1],
                        optimizer_step=len(losses),
                        current_samples=samples_processed + end_node,
                    )
            batch_arrived_frac = statistics.fmean(batch_arrived_fracs)
            validation_metrics_after = checkpoint_manager.last_validation_metrics()
            validation_elapsed_after = (
                None
                if validation_metrics_after is None
                else validation_metrics_after.get("validation_elapsed_s")
            )
            if (
                validation_elapsed_after is not None
                and validation_elapsed_after != validation_elapsed_before
            ):
                validation_elapsed_values.append(float(validation_elapsed_after))
            if len(losses) - last_loss_log_step >= loss_log_interval:
                recent_losses = losses[-loss_log_interval:]
                elapsed = time.perf_counter() - total_t0
                extra_val = ""
                if validation_metrics_after is not None and validation_metrics_after.get("validation_elapsed_s") is not None:
                    extra_val = f" validation_elapsed_s={float(validation_metrics_after['validation_elapsed_s']):.6f}"
                _log_line(
                    log_fh,
                    "[loss] "
                    f"optimizer_steps={len(losses)} "
                    f"avg_last_{loss_log_interval}={statistics.fmean(recent_losses):.6f} "
                    f"latest={losses[-1]:.6f} "
                    f"samples_processed={samples_processed + processed_rows}/{total_samples} "
                    f"arrived_frac_last_batch={batch_arrived_frac:.6f} "
                    f"elapsed_s={elapsed:.3f}"
                    f"{extra_val}",
                )
                last_loss_log_step = len(losses)
            samples_processed += processed_rows
            if inline_health_sampling:
                health_lifecycle.maybe_sample_progress(
                    force=len(losses) == (full_expert_batch // train_batch_size)
                    or len(losses) % 1000 == 0
                )
            if (
                max_training_wall_s is not None
                and time.perf_counter() - total_t0 >= float(max_training_wall_s)
            ):
                wall_budget_reached = True
                break

        torch.cuda.synchronize()
        total_wall_s = time.perf_counter() - total_t0
        if losses:
            record_checkpoint(
                loss_value=losses[-1],
                optimizer_step=len(losses),
                current_samples=samples_processed,
                force=True,
            )
        validation_metrics = checkpoint_manager.last_validation_metrics()
        _log_line(
            log_fh,
            "[done] "
            f"mode=topology_async_training_system "
            f"optimizer_steps={len(losses)} "
            f"samples_processed={samples_processed}/{total_samples}",
        )
        run_completed = True
    finally:
        pipeline.dma_worker.stop()
        health_lifecycle.release_workers()
        for process in processes:
            # A wall-capped run intentionally stops while producers still have
            # work remaining.  Waiting for their natural completion adds up to
            # 15 seconds per producer after the measured region and makes a
            # row appear hung, so request termination before joining them.
            if (wall_budget_reached or not run_completed) and process.is_alive():
                process.terminate()
            process.join(timeout=5 if (wall_budget_reached or not run_completed) else 15)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if dma_thread is not None:
            dma_thread.join(timeout=5.0)
        if health_lifecycle.running:
            health_report = health_lifecycle.finalize(reference_throughput=None)
        worker_timing_rows = sorted(list(worker_timings), key=lambda item: item["expert_id"])
        manager.shutdown()
        _close_training_logger(log_fh)

    mean_loss = float(np.mean(losses)) if losses else None
    min_loss = float(np.min(losses)) if losses else None
    max_loss = float(np.max(losses)) if losses else None
    final_loss = losses[-1] if losses else None
    worker_policy_values = [item["policy_s"] for item in worker_timing_rows]
    worker_env_values = [item["env_step_s"] for item in worker_timing_rows]
    worker_write_values = [item["ringbuffer_write_s"] for item in worker_timing_rows]
    worker_total_values = [item["worker_total_wall_s"] for item in worker_timing_rows]
    health_snapshot = _pipeline_health_snapshot(
        pipeline,
        fallback_block_size=num_agents,
    )
    worker_pids = [getattr(process, "pid", 1000 + index) for index, process in enumerate(processes)]
    worker_exitcodes = [process.exitcode for process in processes]
    return {
        "mode": "topology_async_training_system",
        "health_report": health_report,
        "system": "topology_async_training_system",
        "worker_pids": worker_pids,
        "worker_exitcodes": worker_exitcodes,
        "num_agents": num_agents,
        "num_experts": num_experts,
        "num_producer_processes": int(num_producer_processes),
        "transfer_mode": transfer_mode,
        "producer_groups": [list(group) for group in producer_groups],
        "batch_threshold": batch_threshold,
        "train_batch_size": train_batch_size,
        "expert_timeouts": list(expert_timeouts),
        "num_steps": num_steps,
        "maps_path": maps_path,
        "map_names": selected_map_names,
        "map_families": list(map_families) if map_families is not None else None,
        "pyg_builder_mode": simulator.pyg_builder_mode,
        "pyg_local_gather_impl": simulator.pyg_local_gather_impl,
        "resolved_pyg_builder_impl": resolved_pyg_builder_impl,
        "num_optimizer_steps": len(losses),
        "samples_processed": samples_processed,
        "optimizer_steps_s": len(losses) / total_wall_s,
        "total_wall_s": total_wall_s,
        "samples_s": samples_processed / total_wall_s,
        "mean_loss": mean_loss,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "final_loss": final_loss,
        "loss_history": [float(value) for value in losses],
        "stage_sha256s": stage_sha256s,
        "worker_policy_s_sum": sum(worker_policy_values),
        "worker_policy_s_mean": statistics.fmean(worker_policy_values) if worker_policy_values else 0.0,
        "worker_policy_s_max": max(worker_policy_values) if worker_policy_values else 0.0,
        "worker_env_step_s_sum": sum(worker_env_values),
        "worker_env_step_s_mean": statistics.fmean(worker_env_values) if worker_env_values else 0.0,
        "worker_env_step_s_max": max(worker_env_values) if worker_env_values else 0.0,
        "worker_ringbuffer_write_s_sum": sum(worker_write_values),
        "worker_ringbuffer_write_s_mean": statistics.fmean(worker_write_values) if worker_write_values else 0.0,
        "worker_ringbuffer_write_s_max": max(worker_write_values) if worker_write_values else 0.0,
        "worker_total_wall_s_sum": sum(worker_total_values),
        "worker_total_wall_s_mean": statistics.fmean(worker_total_values) if worker_total_values else 0.0,
        "worker_total_wall_s_max": max(worker_total_values) if worker_total_values else 0.0,
        "consumer_wait_s": consumer_wait_s,
        "consumer_dma_sync_s": consumer_dma_sync_s,
        "consumer_gpu_builder_s": consumer_gpu_builder_s,
        "consumer_train_step_s": consumer_train_step_s,
        **runtime.training_config(),
        "checkpoint_dir": checkpoint_manager.directory,
        "top_k_checkpoints": checkpoint_manager.top_k,
        "checkpoint_interval": checkpoint_manager.save_interval_steps,
        "checkpoint_interval_s": (
            None if checkpoint_interval_s is None else float(checkpoint_interval_s)
        ),
        "checkpoint_selection_mode": selection_mode,
        "validation_trajectories": resolved_validation_trajectories,
        "async_ring_buffer_steps": max(1, int(async_ring_buffer_steps)),
        "saved_checkpoints": checkpoint_manager.snapshot(),
        "checkpoint_history": (
            checkpoint_manager.checkpoint_history()
            if hasattr(checkpoint_manager, "checkpoint_history")
            else []
        ),
        "validation_metrics": validation_metrics,
        "validation_eval_count": len(validation_elapsed_values),
        "validation_elapsed_s_sum": float(sum(validation_elapsed_values)),
        "validation_elapsed_s_mean": (
            float(statistics.fmean(validation_elapsed_values))
            if validation_elapsed_values
            else 0.0
        ),
        "validation_elapsed_s_max": max(validation_elapsed_values) if validation_elapsed_values else 0.0,
        "max_training_wall_s": (
            None if max_training_wall_s is None else float(max_training_wall_s)
        ),
        "wall_budget_reached": wall_budget_reached,
        "initial_model_state_sha256": initial_model_state_sha256,
    }
