from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import copy
import multiprocessing as mp
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch


T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class TorchRNGState:
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class FrozenTrajectory:
    raw_batches: np.ndarray
    metadata: dict
    digest: str


def seed_training_rngs(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_torch_rng_state() -> TorchRNGState:
    cuda_states = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return TorchRNGState(cpu=torch.get_rng_state().clone(), cuda=cuda_states)


def restore_torch_rng_state(state: TorchRNGState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda:
        torch.cuda.set_rng_state_all(list(state.cuda))


def run_paired_rng(first: Callable[[], T], second: Callable[[], U]) -> tuple[T, U]:
    pre_first = capture_torch_rng_state()
    first_result = first()
    post_first = capture_torch_rng_state()
    restore_torch_rng_state(pre_first)
    try:
        second_result = second()
    finally:
        restore_torch_rng_state(post_first)
    return first_result, second_result


def _canonical_metadata_json(metadata: dict) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _trajectory_digest(raw_batches: np.ndarray, metadata_json: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(raw_batches.dtype).encode("ascii"))
    digest.update(json.dumps(list(raw_batches.shape), separators=(",", ":")).encode("ascii"))
    digest.update(np.ascontiguousarray(raw_batches).tobytes(order="C"))
    digest.update(metadata_json.encode("utf-8"))
    return digest.hexdigest()


def _validate_trajectory_contract(raw_batches: np.ndarray, metadata: dict) -> None:
    from expert.expert_running import FEATURE_DIM, RESET_FLAG_COL

    if raw_batches.dtype != np.uint16:
        raise ValueError(f"frozen trajectory dtype must be uint16, got {raw_batches.dtype}")
    if raw_batches.ndim != 3 or raw_batches.shape[2] != FEATURE_DIM:
        raise ValueError(
            "frozen trajectory shape must be "
            f"[num_steps, num_agents, {FEATURE_DIM}], got {raw_batches.shape}"
        )
    expected_shape = (int(metadata["num_steps"]), int(metadata["num_agents"]), FEATURE_DIM)
    if raw_batches.shape != expected_shape:
        raise ValueError(
            f"frozen trajectory shape mismatch: expected {expected_shape}, got {raw_batches.shape}"
        )
    refresh_flags = raw_batches[:, :, RESET_FLAG_COL]
    if not np.all((refresh_flags == 0) | (refresh_flags == 1)):
        raise ValueError("frozen trajectory refresh flags must be binary")


def save_frozen_trajectory(path: str | Path, raw_batches: np.ndarray, metadata: dict) -> str:
    raw_batches = np.ascontiguousarray(raw_batches)
    metadata = dict(metadata)
    _validate_trajectory_contract(raw_batches, metadata)
    metadata_json = _canonical_metadata_json(metadata)
    digest = _trajectory_digest(raw_batches, metadata_json)
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        resolved,
        raw_batches=raw_batches,
        metadata_json=np.asarray(metadata_json),
        sha256=np.asarray(digest),
    )
    return digest


def load_frozen_trajectory(path: str | Path, expected: dict) -> FrozenTrajectory:
    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as payload:
        raw_batches = np.ascontiguousarray(payload["raw_batches"])
        metadata_json = str(payload["metadata_json"].item())
        stored_digest = str(payload["sha256"].item())
    metadata = json.loads(metadata_json)
    _validate_trajectory_contract(raw_batches, metadata)
    for key, expected_value in dict(expected).items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"frozen trajectory metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {metadata.get(key)!r}"
            )
    actual_digest = _trajectory_digest(raw_batches, _canonical_metadata_json(metadata))
    if actual_digest != stored_digest:
        raise ValueError(
            f"frozen trajectory SHA-256 mismatch: expected {stored_digest}, got {actual_digest}"
        )
    return FrozenTrajectory(raw_batches=raw_batches, metadata=metadata, digest=stored_digest)


def collect_frozen_validation_trajectory(
    *,
    output: str | Path,
    maps_path: str,
    map_name: str,
    num_agents: int,
    num_steps: int,
    seed: int,
    max_episode_steps: int,
    expert_timeouts,
    env_builder=None,
    policy_factory=None,
) -> dict:
    from mapf_cuda.training.topology_async import (
        _build_pogema_topology_map_env,
        _build_step_data,
    )
    from expert.expert_running import (
        LacamExpertPolicy,
        initial_refresh_flags,
        no_refresh_flags,
        put_maps_into_registry,
        resolve_expert_timeouts,
    )

    resolved_timeouts = resolve_expert_timeouts(expert_timeouts)
    if env_builder is None:
        put_maps_into_registry(maps_path)
        env_builder = _build_pogema_topology_map_env
    if policy_factory is None:
        policy_factory = LacamExpertPolicy

    env = env_builder(
        num_agents=num_agents,
        map_name=map_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        collision_system="soft",
    )
    policy = policy_factory(timeouts=resolved_timeouts)
    policy.reset_states(env)
    observations = env.env.unwrapped._obs()
    refresh_flags = initial_refresh_flags(len(observations))
    batches = []

    for _ in range(int(num_steps)):
        actions = np.asarray(policy.act(observations), dtype=np.uint16)
        batches.append(
            _build_step_data(
                observations,
                actions,
                env_id=0,
                reset_flag=refresh_flags,
            )
        )
        _, _, terminated, truncated, _ = env.step(actions.astype(np.int64, copy=False))
        if all(terminated) or all(truncated):
            env.reset()
            policy.reset_states(env)
            observations = env.env.unwrapped._obs()
            refresh_flags = initial_refresh_flags(len(observations))
        else:
            observations = env.env.unwrapped._obs()
            refresh_flags = no_refresh_flags(len(observations))

    raw_batches = np.ascontiguousarray(np.stack(batches), dtype=np.uint16)
    metadata = {
        "map_name": map_name,
        "maps_path": maps_path,
        "seed": int(seed),
        "num_steps": int(num_steps),
        "num_agents": int(num_agents),
        "max_episode_steps": int(max_episode_steps),
        "expert_timeouts": [float(value) for value in resolved_timeouts],
        "collision_system": "soft",
        "comparison_mode": "same-state expert-trajectory argmax accuracy",
    }
    digest = save_frozen_trajectory(output, raw_batches, metadata)
    return {
        "path": str(Path(output).expanduser().resolve()),
        "sha256": digest,
        "metadata": metadata,
    }


def create_paired_runtimes(
    *,
    model_seed: int,
    runtime_factory=None,
    **runtime_kwargs,
):
    if runtime_factory is None:
        from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter

        runtime_factory = MAGATRuntimeAdapter

    seed_training_rngs(model_seed)
    runtime_a = runtime_factory(
        train_on_arrived_agents=True,
        **runtime_kwargs,
    )
    post_a_initialization = capture_torch_rng_state()
    runtime_b = runtime_factory(
        train_on_arrived_agents=False,
        **runtime_kwargs,
    )
    runtime_b.model.load_state_dict(copy.deepcopy(runtime_a.model.state_dict()), strict=True)
    runtime_b.optimizer.load_state_dict(copy.deepcopy(runtime_a.optimizer.state_dict()))
    if runtime_a.scheduler is not None:
        if runtime_b.scheduler is None:
            raise ValueError("paired runtime B is missing scheduler configured on runtime A")
        runtime_b.scheduler.load_state_dict(copy.deepcopy(runtime_a.scheduler.state_dict()))
    elif runtime_b.scheduler is not None:
        raise ValueError("paired runtime scheduler configuration mismatch")
    restore_torch_rng_state(post_a_initialization)
    return runtime_a, runtime_b


def train_paired_on_batch(runtime_a, runtime_b, materialized_batch):
    return run_paired_rng(
        lambda: runtime_a.train_step_from_batch(materialized_batch),
        lambda: runtime_b.train_step_from_batch(materialized_batch),
    )


def prepare_new_output_directories(path_a: str | Path, path_b: str | Path) -> tuple[Path, Path]:
    resolved_a = Path(path_a).expanduser().resolve()
    resolved_b = Path(path_b).expanduser().resolve()
    if resolved_a == resolved_b:
        raise ValueError("paired checkpoint directories must be distinct")
    for path in (resolved_a, resolved_b):
        if path.exists():
            raise FileExistsError(f"paired checkpoint directory must not already exist: {path}")
    resolved_a.parent.mkdir(parents=True, exist_ok=True)
    resolved_b.parent.mkdir(parents=True, exist_ok=True)
    resolved_a.mkdir()
    resolved_b.mkdir()
    return resolved_a, resolved_b


def enforce_pilot_gate(diagnostic: dict, stop_workers: Callable[[], None]) -> None:
    required = {
        "worker_states",
        "worker_exit_codes",
        "reserve_ptr",
        "dma_ptr",
        "compute_ptr",
        "gpu_utilization",
        "gpu_memory_mib",
        "elapsed_s",
        "rows_processed",
        "rows_s",
        "updates_a",
        "updates_b",
    }
    missing = sorted(required - set(diagnostic))
    if missing:
        raise ValueError(f"pilot diagnostic is missing fields: {missing}")
    passed = (
        float(diagnostic["rows_s"]) >= 20_000.0
        and int(diagnostic["updates_a"]) == 1000
        and int(diagnostic["updates_b"]) == 1000
    )
    if passed:
        return
    stop_workers()
    raise RuntimeError(
        "paired pilot gate failed: "
        + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
    )


def process_shared_frontier(
    *,
    runtime_a,
    runtime_b,
    simulator,
    raw_batch: torch.Tensor,
    supervision_stats: dict,
) -> tuple[float, float]:
    materialized_batch = runtime_a.build_batch(
        simulator,
        raw_batch,
        materialize_edges=True,
    )
    arrived_rows = int(materialized_batch.arrived.sum().item())
    total_rows = int(materialized_batch.arrived.numel())
    effective_a = total_rows
    effective_b = total_rows - arrived_rows
    supervision_stats["arrived_rows"] += arrived_rows
    supervision_stats["effective_rows_a"] += effective_a
    supervision_stats["effective_rows_b"] += effective_b
    supervision_stats["zero_effective_batches_a"] += int(effective_a == 0)
    supervision_stats["zero_effective_batches_b"] += int(effective_b == 0)
    loss_a, loss_b = train_paired_on_batch(runtime_a, runtime_b, materialized_batch)
    return float(loss_a.item()), float(loss_b.item())


def audit_checkpoint_directory(
    checkpoint_dir: str | Path,
    *,
    expected_steps,
    expected_meta: dict,
) -> list[dict]:
    directory = Path(checkpoint_dir).expanduser().resolve()
    paths = sorted(directory.glob("*.pt"))
    payloads = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        step = int(payload.get("optimizer_step", -1))
        metadata = payload.get("meta", {})
        for key, expected_value in expected_meta.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise ValueError(
                    f"checkpoint metadata mismatch for {key} at step {step}: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )
        payloads.append(
            {
                "path": str(path),
                "optimizer_step": step,
                "loss": float(payload["loss"]),
                "meta": metadata,
            }
        )
    actual_steps = [row["optimizer_step"] for row in payloads]
    expected_steps = [int(step) for step in expected_steps]
    if actual_steps != expected_steps:
        raise ValueError(
            f"checkpoint steps mismatch: expected {expected_steps}, got {actual_steps}"
        )
    if len(set(actual_steps)) != len(actual_steps):
        raise ValueError(f"checkpoint steps must be unique, got {actual_steps}")
    return payloads


def compose_training_summary(
    *,
    base: dict,
    runtime_a,
    runtime_b,
    checkpoints_a: list[dict],
    checkpoints_b: list[dict],
    pilot_diagnostic: dict | None,
) -> dict:
    config_a = runtime_a.training_config()
    config_b = runtime_b.training_config()
    summary = {
        **dict(base),
        "pilot_diagnostic": None if pilot_diagnostic is None else dict(pilot_diagnostic),
        "branches": {
            "with_arrived": {
                "branch": "with_arrived",
                "train_on_arrived_agents": True,
                "training_config": config_a,
                "checkpoints": list(checkpoints_a),
            },
            "without_arrived": {
                "branch": "without_arrived",
                "train_on_arrived_agents": False,
                "training_config": config_b,
                "checkpoints": list(checkpoints_b),
            },
        },
        "runtime_with_arrived": config_a,
        "runtime_without_arrived": config_b,
        "checkpoints_with_arrived": list(checkpoints_a),
        "checkpoints_without_arrived": list(checkpoints_b),
    }
    return summary


def _runtime_checkpoint_metadata(
    *,
    runtime,
    model_seed: int,
    branch: str,
    paired_run_id: str,
    samples_processed: int,
    total_samples: int,
    num_agents: int,
    num_experts: int,
    train_batch_size: int,
    map_names,
    maps_path: str,
    max_episode_steps: int,
    expert_timeouts,
    supervision_stats: dict,
) -> dict:
    return {
        "samples_processed": int(samples_processed),
        "total_samples": int(total_samples),
        "num_agents": int(num_agents),
        "num_experts": int(num_experts),
        "train_batch_size": int(train_batch_size),
        "map_names": list(map_names),
        "maps_path": maps_path,
        "max_episode_steps": int(max_episode_steps),
        "expert_timeouts": [float(value) for value in expert_timeouts],
        "model_seed": int(model_seed),
        "branch": branch,
        "paired_run_id": paired_run_id,
        **dict(supervision_stats),
        **runtime.training_config(),
    }


def _stop_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)


def _pilot_diagnostic(
    *,
    pipeline,
    processes,
    elapsed_s: float,
    rows_processed: int,
    updates_a: int,
    updates_b: int,
) -> dict:
    pipeline_stats = pipeline.get_stats()
    utilization_fn = getattr(torch.cuda, "utilization", None)
    try:
        gpu_utilization = int(utilization_fn()) if utilization_fn is not None else -1
    except Exception:
        gpu_utilization = -1
    return {
        "worker_states": ["alive" if process.is_alive() else "finished" for process in processes],
        "worker_exit_codes": [process.exitcode for process in processes],
        "reserve_ptr": int(pipeline_stats["reserve_ptr"]),
        "dma_ptr": int(pipeline_stats["dma_read_ptr"]),
        "compute_ptr": int(pipeline_stats["compute_ptr"]),
        "gpu_utilization": gpu_utilization,
        "gpu_memory_mib": int(torch.cuda.memory_allocated() // (1024 * 1024)),
        "elapsed_s": float(elapsed_s),
        "rows_processed": int(rows_processed),
        "rows_s": float(rows_processed / elapsed_s),
        "updates_a": int(updates_a),
        "updates_b": int(updates_b),
    }


def run_controlled_paired_training(
    *,
    num_steps: int,
    num_agents: int,
    num_experts: int,
    batch_threshold: int,
    train_batch_size: int,
    seed: int,
    model_seed: int,
    paired_run_id: str,
    maps_path: str,
    map_names,
    max_episode_steps: int,
    expert_timeouts,
    pyg_builder_mode: str,
    pyg_local_gather_impl: str,
    lr_start: float,
    lr_end: float,
    lr_scheduler: str | None,
    scheduler_total_steps: int | None,
    grad_clip_norm: float | None,
    checkpoint_interval: int,
    checkpoint_dir_with_arrived: str | Path,
    checkpoint_dir_without_arrived: str | Path,
    summary_json: str | Path,
) -> dict:
    from mapf_cuda.training.topology_async import (
        _apply_simulator_builder_policy,
        _build_stateless_simulator_from_grids,
        _default_async_ring_capacity,
        _select_topology_training_maps,
        _topology_async_training_expert_worker_loop,
    )
    from expert.checkpoint_manager import TopKCheckpointManager
    from expert.expert_running import ExtremeMAPFPipeline, resolve_expert_timeouts

    if int(batch_threshold) != int(num_agents) * int(num_experts):
        raise ValueError("controlled paired training requires one full four-environment frontier")
    if int(train_batch_size) != int(batch_threshold):
        raise ValueError("controlled paired training requires train_batch_size == batch_threshold")
    checkpoint_a, checkpoint_b = prepare_new_output_directories(
        checkpoint_dir_with_arrived,
        checkpoint_dir_without_arrived,
    )
    summary_path = Path(summary_json).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_timeouts = resolve_expert_timeouts(expert_timeouts)
    selected_maps = _select_topology_training_maps(
        maps_path=maps_path,
        num_experts=num_experts,
        map_names=map_names,
        map_families=None,
    )
    selected_map_names = [name for name, _ in selected_maps]
    simulator = _build_stateless_simulator_from_grids(
        [grid for _, grid in selected_maps],
        num_agents=num_agents,
        device="cuda:0",
    )
    resolved_builder = _apply_simulator_builder_policy(
        simulator,
        pyg_builder_mode=pyg_builder_mode,
        pyg_local_gather_impl=pyg_local_gather_impl,
    )
    full_batch = int(num_agents) * int(num_experts)
    pipeline = ExtremeMAPFPipeline(
        simulator,
        capacity=_default_async_ring_capacity(full_batch),
        batch_threshold=full_batch,
        device="cuda:0",
    )
    pipeline.initialize()
    runtime_a, runtime_b = create_paired_runtimes(
        model_seed=model_seed,
        device="cuda:0",
        lr=lr_start,
        lr_end=lr_end,
        lr_scheduler=lr_scheduler,
        scheduler_total_steps=scheduler_total_steps if lr_scheduler else None,
        grad_clip_norm=grad_clip_norm,
    )
    manager_a = TopKCheckpointManager(
        checkpoint_a,
        top_k=num_steps // checkpoint_interval,
        save_interval_steps=checkpoint_interval,
        mode_label="controlled_arrived_ab_with_arrived",
        logger=print,
    )
    manager_b = TopKCheckpointManager(
        checkpoint_b,
        top_k=num_steps // checkpoint_interval,
        save_interval_steps=checkpoint_interval,
        mode_label="controlled_arrived_ab_without_arrived",
        logger=print,
    )

    losses_a = []
    losses_b = []
    supervision_stats = {
        "arrived_rows": 0,
        "effective_rows_a": 0,
        "effective_rows_b": 0,
        "zero_effective_batches_a": 0,
        "zero_effective_batches_b": 0,
    }
    total_samples = int(num_steps) * full_batch
    samples_processed = 0
    processes = []
    worker_timing_rows = []
    pilot_diagnostic = None
    manager = mp.Manager()
    worker_timings = manager.list()
    deadline = time.perf_counter() + max(60.0, num_steps * num_experts * 2.0)
    total_t0 = time.perf_counter()

    pipeline.dma_worker.start()
    try:
        for expert_id, map_name in enumerate(selected_map_names):
            process = mp.Process(
                target=_topology_async_training_expert_worker_loop,
                args=(expert_id, pipeline.ring_buffer, worker_timings),
                kwargs={
                    "maps_path": maps_path,
                    "map_name": map_name,
                    "num_agents": num_agents,
                    "num_steps": num_steps,
                    "seed": seed + expert_id,
                    "max_episode_steps": max_episode_steps,
                    "expert_timeouts": resolved_timeouts,
                },
                daemon=True,
            )
            process.start()
            processes.append(process)

        print(
            "[start] controlled_arrived_ab "
            f"paired_run_id={paired_run_id} steps={num_steps} total_samples={total_samples}"
        )
        while len(losses_a) < int(num_steps):
            failed_workers = [
                (idx, process.exitcode)
                for idx, process in enumerate(processes)
                if process.exitcode not in (None, 0)
            ]
            if failed_workers:
                raise RuntimeError(f"controlled expert workers failed: {failed_workers}")
            if time.perf_counter() > deadline:
                raise TimeoutError("controlled paired training timed out")
            pipeline.dma_read_ptr = pipeline.dma_worker.dma_read_ptr
            if not pipeline.has_env_aligned_batch(num_experts):
                time.sleep(0.001)
                continue
            pipeline.dma_event.synchronize()
            raw_batch = pipeline.extract_env_aligned_batch(
                full_batch,
                agents_per_env=num_agents,
                num_envs=num_experts,
            )
            if raw_batch is None:
                time.sleep(0.001)
                continue
            agents_to_update = pipeline.gpu_handler.filter_reset_agents(raw_batch)
            pipeline.gpu_handler.process_batch(raw_batch, agents_to_update)
            loss_a, loss_b = process_shared_frontier(
                runtime_a=runtime_a,
                runtime_b=runtime_b,
                simulator=simulator,
                raw_batch=raw_batch,
                supervision_stats=supervision_stats,
            )
            losses_a.append(loss_a)
            losses_b.append(loss_b)
            samples_processed += full_batch
            step = len(losses_a)

            common_meta = dict(
                model_seed=model_seed,
                paired_run_id=paired_run_id,
                samples_processed=samples_processed,
                total_samples=total_samples,
                num_agents=num_agents,
                num_experts=num_experts,
                train_batch_size=train_batch_size,
                map_names=selected_map_names,
                maps_path=maps_path,
                max_episode_steps=max_episode_steps,
                expert_timeouts=resolved_timeouts,
                supervision_stats=supervision_stats,
            )
            manager_a.maybe_save(
                loss=loss_a,
                runtime=runtime_a,
                optimizer_step=step,
                extra_meta=_runtime_checkpoint_metadata(
                    runtime=runtime_a,
                    branch="with_arrived",
                    **common_meta,
                ),
            )
            manager_b.maybe_save(
                loss=loss_b,
                runtime=runtime_b,
                optimizer_step=step,
                extra_meta=_runtime_checkpoint_metadata(
                    runtime=runtime_b,
                    branch="without_arrived",
                    **common_meta,
                ),
            )

            if step == 1000:
                elapsed = time.perf_counter() - total_t0
                diagnostic = _pilot_diagnostic(
                    pipeline=pipeline,
                    processes=processes,
                    elapsed_s=elapsed,
                    rows_processed=samples_processed,
                    updates_a=step,
                    updates_b=len(losses_b),
                )
                pilot_diagnostic = dict(diagnostic)
                print("[pilot] " + json.dumps(diagnostic, sort_keys=True))
                enforce_pilot_gate(
                    diagnostic,
                    stop_workers=lambda: _stop_processes(processes),
                )
            if step % 1000 == 0:
                elapsed = time.perf_counter() - total_t0
                print(
                    f"[progress] step={step}/{num_steps} rows={samples_processed}/{total_samples} "
                    f"loss_a={loss_a:.6f} loss_b={loss_b:.6f} "
                    f"rows_s={samples_processed / elapsed:.3f}"
                )

        torch.cuda.synchronize()
        total_wall_s = time.perf_counter() - total_t0
    finally:
        pipeline.dma_worker.stop()
        for process in processes:
            process.join(timeout=15)
        _stop_processes(processes)
        worker_timing_rows = sorted(list(worker_timings), key=lambda item: item["expert_id"])
        manager.shutdown()

    expected_steps = list(range(checkpoint_interval, num_steps + 1, checkpoint_interval))
    expected_common = {
        "model_seed": int(model_seed),
        "paired_run_id": paired_run_id,
        "map_names": selected_map_names,
        "scheduler_total_steps": int(scheduler_total_steps) if lr_scheduler else None,
        "max_episode_steps": int(max_episode_steps),
        "expert_timeouts": [float(value) for value in resolved_timeouts],
    }
    audited_a = audit_checkpoint_directory(
        checkpoint_a,
        expected_steps=expected_steps,
        expected_meta={
            **expected_common,
            "branch": "with_arrived",
            "train_on_arrived_agents": True,
        },
    )
    audited_b = audit_checkpoint_directory(
        checkpoint_b,
        expected_steps=expected_steps,
        expected_meta={
            **expected_common,
            "branch": "without_arrived",
            "train_on_arrived_agents": False,
        },
    )
    summary = compose_training_summary(
        base={
            "mode": "controlled_arrived_ab",
            "paired_run_id": paired_run_id,
            "model_seed": int(model_seed),
            "seed": int(seed),
            "num_steps": int(num_steps),
            "num_agents": int(num_agents),
            "num_experts": int(num_experts),
            "batch_threshold": int(batch_threshold),
            "train_batch_size": int(train_batch_size),
            "samples_processed": int(samples_processed),
            "total_samples": int(total_samples),
            "total_wall_s": float(total_wall_s),
            "samples_s": float(total_samples / total_wall_s),
            "updates_s_per_branch": float(num_steps / total_wall_s),
            "map_names": selected_map_names,
            "maps_path": maps_path,
            "max_episode_steps": int(max_episode_steps),
            "expert_timeouts": [float(value) for value in resolved_timeouts],
            "resolved_pyg_builder_impl": resolved_builder,
            "supervision": dict(supervision_stats),
            "losses_with_arrived": losses_a,
            "losses_without_arrived": losses_b,
            "worker_timings": worker_timing_rows,
        },
        runtime_a=runtime_a,
        runtime_b=runtime_b,
        checkpoints_a=audited_a,
        checkpoints_b=audited_b,
        pilot_diagnostic=pilot_diagnostic,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[done] controlled_arrived_ab steps={num_steps} samples={samples_processed} "
        f"wall_s={total_wall_s:.3f} rows_s={total_samples / total_wall_s:.3f}"
    )
    return summary


def replay_frozen_batches(
    *,
    raw_batches: np.ndarray,
    simulator,
    device: str,
    predict_actions: Callable,
) -> dict:
    from collections import Counter
    from expert.expert_running import RESET_FLAG_COL

    overall_correct = 0
    overall_total = 0
    nonstay_correct = 0
    nonstay_total = 0
    expert_counter = Counter()
    model_counter = Counter()
    for raw_batch_np in raw_batches:
        raw_batch = torch.from_numpy(
            np.ascontiguousarray(raw_batch_np).astype(np.int16, copy=False)
        ).to(device)
        refresh_rows = raw_batch[raw_batch[:, RESET_FLAG_COL] != 0]
        if refresh_rows.shape[0] > 0:
            simulator.update_derived_state(refresh_rows, int(refresh_rows.shape[0]))
        simulator.refresh_compact_state_from_raw_batch(raw_batch)
        model_actions = np.asarray(predict_actions(simulator, raw_batch), dtype=np.int64)
        expert_actions = raw_batch_np[:, 6].astype(np.int64, copy=False)
        matches = model_actions == expert_actions
        nonstay = expert_actions != 0
        overall_correct += int(matches.sum())
        overall_total += int(matches.size)
        nonstay_correct += int(matches[nonstay].sum())
        nonstay_total += int(nonstay.sum())
        expert_counter.update(expert_actions.tolist())
        model_counter.update(model_actions.tolist())
    return {
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "overall_accuracy": overall_correct / overall_total if overall_total else 0.0,
        "nonstay_correct": nonstay_correct,
        "nonstay_total": nonstay_total,
        "nonstay_accuracy": nonstay_correct / nonstay_total if nonstay_total else 0.0,
        "expert_action_counts": dict(sorted(expert_counter.items())),
        "model_action_counts": dict(sorted(model_counter.items())),
    }


def evaluate_checkpoint_on_frozen_trajectory(
    *,
    checkpoint: str | Path,
    trajectory: str | Path,
    device: str = "cuda:0",
) -> dict:
    from mapf_cuda.models.checkpoints import load_magat_checkpoint
    from mapf_cuda.simulation.grids import extract_single_env_grid
    from mapf_cuda.training.topology_async import _build_pogema_topology_map_env
    from expert.expert_running import put_maps_into_registry
    from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter
    import grid_world_cpp as ext

    frozen = load_frozen_trajectory(trajectory, expected={})
    metadata = frozen.metadata
    put_maps_into_registry(metadata["maps_path"])
    env = _build_pogema_topology_map_env(
        num_agents=metadata["num_agents"],
        map_name=metadata["map_name"],
        seed=metadata["seed"],
        max_episode_steps=metadata["max_episode_steps"],
        collision_system="soft",
    )
    simulator = ext.StatelessGridWorldSimulator(
        extract_single_env_grid(env, device=device),
        metadata["num_agents"],
        3,
    )
    runtime = MAGATRuntimeAdapter(device=device)
    checkpoint_path = str(Path(checkpoint).expanduser().resolve())
    load_magat_checkpoint(runtime, checkpoint_path, device)

    def predict_actions(sim, raw_batch):
        batch = runtime.build_batch(sim, raw_batch)
        with torch.no_grad():
            logits = runtime.model(batch.x, batch)
        return logits.argmax(dim=-1).to(torch.int64).cpu().numpy()

    metrics = replay_frozen_batches(
        raw_batches=frozen.raw_batches,
        simulator=simulator,
        device=device,
        predict_actions=predict_actions,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return {
        "checkpoint": checkpoint_path,
        "optimizer_step": int(payload["optimizer_step"]),
        "loss": float(payload["loss"]),
        "branch": payload["meta"]["branch"],
        "train_on_arrived_agents": bool(payload["meta"]["train_on_arrived_agents"]),
        "paired_run_id": payload["meta"]["paired_run_id"],
        "trajectory_sha256": frozen.digest,
        **metrics,
    }


def evaluate_paired_checkpoint_directories(
    *,
    trajectory: str | Path,
    checkpoint_dir_with_arrived: str | Path,
    checkpoint_dir_without_arrived: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    device: str = "cuda:0",
) -> dict:
    dirs = {
        "with_arrived": Path(checkpoint_dir_with_arrived).expanduser().resolve(),
        "without_arrived": Path(checkpoint_dir_without_arrived).expanduser().resolve(),
    }
    first_payloads = {}
    for branch, directory in dirs.items():
        paths = sorted(directory.glob("*.pt"))
        if len(paths) != 10:
            raise ValueError(f"expected exactly 10 checkpoints for {branch}, got {len(paths)}")
        first_payloads[branch] = torch.load(paths[0], map_location="cpu", weights_only=False)
    common_keys = (
        "model_seed",
        "paired_run_id",
        "map_names",
        "scheduler_total_steps",
        "max_episode_steps",
        "expert_timeouts",
    )
    common_meta = {
        key: first_payloads["with_arrived"]["meta"][key]
        for key in common_keys
    }
    for key in common_keys:
        other = first_payloads["without_arrived"]["meta"][key]
        if other != common_meta[key]:
            raise ValueError(f"paired checkpoint metadata differs for {key}")
    steps = list(range(1000, 10001, 1000))
    audited = {
        "with_arrived": audit_checkpoint_directory(
            dirs["with_arrived"],
            expected_steps=steps,
            expected_meta={
                **common_meta,
                "branch": "with_arrived",
                "train_on_arrived_agents": True,
            },
        ),
        "without_arrived": audit_checkpoint_directory(
            dirs["without_arrived"],
            expected_steps=steps,
            expected_meta={
                **common_meta,
                "branch": "without_arrived",
                "train_on_arrived_agents": False,
            },
        ),
    }
    rows = []
    for branch in ("with_arrived", "without_arrived"):
        for checkpoint_row in audited[branch]:
            result = evaluate_checkpoint_on_frozen_trajectory(
                checkpoint=checkpoint_row["path"],
                trajectory=trajectory,
                device=device,
            )
            rows.append(result)
            print(
                f"[accuracy] branch={branch} step={result['optimizer_step']} "
                f"overall={result['overall_accuracy']:.6f} "
                f"nonstay={result['nonstay_accuracy']:.6f}"
            )
    digests = {row["trajectory_sha256"] for row in rows}
    if len(digests) != 1:
        raise ValueError(f"checkpoint evaluations used different trajectory digests: {digests}")
    report = {
        "trajectory": str(Path(trajectory).expanduser().resolve()),
        "trajectory_sha256": next(iter(digests)),
        "paired_run_id": common_meta["paired_run_id"],
        "model_seed": common_meta["model_seed"],
        "results": rows,
    }
    json_path = Path(output_json).expanduser().resolve()
    csv_path = Path(output_csv).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "branch",
            "train_on_arrived_agents",
            "optimizer_step",
            "loss",
            "overall_correct",
            "overall_total",
            "overall_accuracy",
            "nonstay_correct",
            "nonstay_total",
            "nonstay_accuracy",
            "trajectory_sha256",
            "checkpoint",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    return report


def _parse_timeouts(raw: str):
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled arrived-agent A/B experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    for name in ("num_steps", "num_agents", "num_experts", "batch_threshold", "train_batch_size", "seed", "model_seed", "max_episode_steps", "scheduler_total_steps", "checkpoint_interval"):
        train.add_argument(f"--{name}", type=int, required=True)
    for name in ("lr_start", "lr_end", "grad_clip_norm"):
        train.add_argument(f"--{name}", type=float, required=True)
    for name in (
        "paired_run_id",
        "maps_path",
        "map_names",
        "expert_timeouts",
        "pyg_builder_mode",
        "pyg_local_gather_impl",
        "lr_scheduler",
        "checkpoint_dir_with_arrived",
        "checkpoint_dir_without_arrived",
        "summary_json",
    ):
        train.add_argument(f"--{name}", required=True)

    freeze = subparsers.add_parser("freeze-validation")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--maps_path", required=True)
    freeze.add_argument("--map_name", required=True)
    freeze.add_argument("--num_agents", type=int, required=True)
    freeze.add_argument("--num_steps", type=int, required=True)
    freeze.add_argument("--seed", type=int, required=True)
    freeze.add_argument("--max_episode_steps", type=int, required=True)
    freeze.add_argument("--expert_timeouts", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--trajectory", required=True)
    evaluate.add_argument("--checkpoint_dir_with_arrived", required=True)
    evaluate.add_argument("--checkpoint_dir_without_arrived", required=True)
    evaluate.add_argument("--output_json", required=True)
    evaluate.add_argument("--output_csv", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "train":
        run_controlled_paired_training(
            num_steps=args.num_steps,
            num_agents=args.num_agents,
            num_experts=args.num_experts,
            batch_threshold=args.batch_threshold,
            train_batch_size=args.train_batch_size,
            seed=args.seed,
            model_seed=args.model_seed,
            paired_run_id=args.paired_run_id,
            maps_path=args.maps_path,
            map_names=[part.strip() for part in args.map_names.split(",") if part.strip()],
            max_episode_steps=args.max_episode_steps,
            expert_timeouts=_parse_timeouts(args.expert_timeouts),
            pyg_builder_mode=args.pyg_builder_mode,
            pyg_local_gather_impl=args.pyg_local_gather_impl,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            lr_scheduler=args.lr_scheduler or None,
            scheduler_total_steps=args.scheduler_total_steps,
            grad_clip_norm=args.grad_clip_norm,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir_with_arrived=args.checkpoint_dir_with_arrived,
            checkpoint_dir_without_arrived=args.checkpoint_dir_without_arrived,
            summary_json=args.summary_json,
        )
    elif args.command == "freeze-validation":
        result = collect_frozen_validation_trajectory(
            output=args.output,
            maps_path=args.maps_path,
            map_name=args.map_name,
            num_agents=args.num_agents,
            num_steps=args.num_steps,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            expert_timeouts=_parse_timeouts(args.expert_timeouts),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        evaluate_paired_checkpoint_directories(
            trajectory=args.trajectory,
            checkpoint_dir_with_arrived=args.checkpoint_dir_with_arrived,
            checkpoint_dir_without_arrived=args.checkpoint_dir_without_arrived,
            output_json=args.output_json,
            output_csv=args.output_csv,
            device=args.device,
        )


if __name__ == "__main__":
    main()
