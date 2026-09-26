"""Concrete B2 reference, strong-online, and compact-sync training engines."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


def _resolve_training_wall_budget(kwargs: dict[str, Any]) -> float | None:
    value = kwargs.get("max_training_wall_s")
    if value is None:
        return None
    budget = float(value)
    if budget <= 0:
        raise ValueError("max_training_wall_s must be positive")
    return budget


def _training_wall_reached(started: float, budget: float | None) -> bool:
    return budget is not None and time.perf_counter() - started >= budget


def _resolve_checkpoint_wall_interval(kwargs: dict[str, Any]) -> float | None:
    value = kwargs.get("checkpoint_interval_s")
    if value is None:
        return None
    interval = float(value)
    if interval <= 0:
        raise ValueError("checkpoint_interval_s must be positive")
    return interval


def _advance_wall_checkpoint_deadline(
    current_deadline_s: float, interval_s: float, elapsed_s: float
) -> float:
    while current_deadline_s <= elapsed_s:
        current_deadline_s += interval_s
    return current_deadline_s


class _StageSink:
    def __init__(self) -> None:
        self.blocks: list[np.ndarray] = []

    def reserve_and_write(self, rows: np.ndarray):
        self.blocks.append(np.array(rows, dtype=np.uint16, copy=True))
        return (0, int(rows.shape[0]))


class _HostPipelineState:
    def __init__(self, ring_buffer, stage_rows: int) -> None:
        self.ring_buffer = ring_buffer
        self.capacity = int(ring_buffer.capacity)
        self.compute_ptr = 0
        self.dma_worker = SimpleNamespace(dma_read_ptr=0)
        self.stage_rows = int(stage_rows)

    def get_stats(self) -> dict[str, int]:
        return {
            "reserve_ptr": int(self.ring_buffer.shared_reserve_ptr.value),
            "dma_read_ptr": int(self.dma_worker.dma_read_ptr),
            "compute_ptr": int(self.compute_ptr),
            "capacity": self.capacity,
            "stage_rows": self.stage_rows,
        }


def _selected_maps(maps_path: str, map_names) -> tuple[list[tuple[str, Any]], np.ndarray]:
    from mapf_cuda.training.topology_async import (
        _select_topology_training_maps,
    )

    selected = _select_topology_training_maps(
        maps_path=maps_path,
        num_experts=len(map_names),
        map_names=map_names,
    )
    grids = np.stack(
        [np.asarray(grid, dtype=np.uint8) for _, grid in selected], axis=0
    )
    return selected, grids


def _generate_sequential_stages(
    backend: str,
    *,
    maps_path: str,
    selected,
    num_agents: int,
    num_steps: int,
    seed: int,
    max_episode_steps: int,
) -> list[np.ndarray]:
    sinks = []
    for env_id, (map_name, _) in enumerate(selected):
        sink = _StageSink()
        sinks.append(sink)
        kwargs = {
            "maps_path": maps_path,
            "map_name": map_name,
            "num_agents": int(num_agents),
            "num_steps": int(num_steps),
            "seed": int(seed) + env_id,
            "max_episode_steps": int(max_episode_steps),
            "health_handle": None,
        }
        if backend == "magat":
            from mapf_cuda.training.topology_async import (
                _topology_async_training_expert_worker_loop,
            )

            _topology_async_training_expert_worker_loop(env_id, sink, **kwargs)
        else:
            from expert.mapf_gpt_online import mapf_gpt_expert_worker_loop

            mapf_gpt_expert_worker_loop(env_id, sink, **kwargs)
    if any(len(sink.blocks) != int(num_steps) for sink in sinks):
        raise RuntimeError("sequential expert generation returned an incomplete stage set")
    return [
        np.concatenate([sink.blocks[step] for sink in sinks], axis=0)
        for step in range(int(num_steps))
    ]


def _pin_to_device(array: np.ndarray, *, device: str, dtype=None):
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.pin_memory().to(device, non_blocking=True)


def _build_host_magat_batch(grids: np.ndarray, rows: np.ndarray, *, device: str):
    import torch

    from expert.fixed_magat_plus_runtime import RuntimePyGBatch
    from expert.magat_reference_oracle import build_magat_reference_batch

    num_envs = int(grids.shape[0])
    if rows.shape[0] % num_envs != 0:
        raise ValueError("MAGAT host stage does not divide evenly across environments")
    num_agents = rows.shape[0] // num_envs
    references = []
    for env_id in range(num_envs):
        start = env_id * num_agents
        end = start + num_agents
        references.append(build_magat_reference_batch(grids[env_id], rows[start:end]))
    x = np.concatenate([item.x for item in references], axis=0)
    labels = np.concatenate([item.labels for item in references], axis=0)
    edge_indices = []
    for env_id, item in enumerate(references):
        edge_indices.append(item.edge_index + env_id * num_agents)
    edge_index = np.concatenate(edge_indices, axis=1)
    edge_attr = np.concatenate([item.edge_attr for item in references], axis=0)
    arrived = np.all(rows[:, 2:4] == rows[:, 4:6], axis=1)
    return RuntimePyGBatch(
        x=_pin_to_device(x.astype(np.float32, copy=False), device=device),
        edge_index=_pin_to_device(
            edge_index.astype(np.int64, copy=False), device=device, dtype=torch.int64
        ),
        edge_attr=_pin_to_device(
            edge_attr.astype(np.float32, copy=False), device=device
        ),
        batch=_pin_to_device(
            np.repeat(np.arange(num_envs, dtype=np.int64), num_agents),
            device=device,
            dtype=torch.int64,
        ),
        ptr=_pin_to_device(
            np.arange(0, rows.shape[0] + 1, num_agents, dtype=np.int64),
            device=device,
            dtype=torch.int64,
        ),
        y=_pin_to_device(labels.astype(np.int64, copy=False), device=device, dtype=torch.int64),
        terminated=torch.zeros(rows.shape[0], dtype=torch.bool, device=device),
        arrived=_pin_to_device(arrived.astype(np.bool_), device=device, dtype=torch.bool),
    )


class _HostMapfGPTBuilder:
    def __init__(self, grids: np.ndarray, *, num_agents: int, device: str) -> None:
        self.grids = np.array(grids, dtype=np.uint8, copy=True)
        self.num_agents = int(num_agents)
        self.device = device
        self.tokens = self.labels = self.active_mask = self.diagnostics = None

    def build_tokens(self, raw_stage) -> None:
        import torch

        from expert.mapf_gpt_schema import tokenize_stage_reference

        rows = raw_stage.detach().cpu().numpy().astype(np.uint16, copy=False)
        tokens, labels = tokenize_stage_reference(
            self.grids, rows, num_agents=self.num_agents
        )
        active = np.any(rows[:, 2:4] != rows[:, 4:6], axis=1)
        self.tokens = _pin_to_device(
            tokens.astype(np.int32, copy=False), device=self.device, dtype=torch.int32
        )
        self.labels = _pin_to_device(
            labels.astype(np.int64, copy=False), device=self.device, dtype=torch.long
        )
        self.active_mask = _pin_to_device(
            active.astype(np.bool_), device=self.device, dtype=torch.bool
        )
        self.diagnostics = torch.zeros(4, dtype=torch.int64, device=self.device)


def _gpu_snapshot() -> dict[str, Any]:
    from mapf_cuda.observability.pipeline import query_gpu_snapshot

    return query_gpu_snapshot(device_index=0)


def _no_worker_health_start(stage_rows: int):
    from expert.benchmark_health import HealthCollector

    state = {"rows": 0, "optimizer_steps": 0, "start": time.monotonic()}

    def pipeline_provider():
        elapsed = max(time.monotonic() - state["start"], 1e-9)
        rows = int(state["rows"])
        return {
            "applicable": True,
            "producer_count": rows,
            "dma_count": rows,
            "consumer_count": rows,
            "optimizer_steps": int(state["optimizer_steps"]),
            "producer_backlog": False,
            "producer_capacity_available": True,
            "dma_backlog": False,
            "consumer_backlog": False,
            "optimizer_backlog": False,
            "reserve_ptr": rows,
            "dma_ptr": rows,
            "compute_ptr": rows,
            "ring_capacity": int(stage_rows),
            "block_size": int(stage_rows),
            "gpu_active": False,
            "gpu_progress_counter": int(state["optimizer_steps"]),
            "active_expert_calls": [],
            "producer_workers": [],
            "rolling_throughput": float(rows / elapsed),
        }

    collector = HealthCollector(
        process_provider=lambda: {"parent_pid": os.getpid(), "workers": []},
        gpu_provider=_gpu_snapshot,
        pipeline_provider=pipeline_provider,
        sample_interval_s=5.0,
    )
    collector.start()
    return state, collector


def _no_worker_health_finish(collector):
    from expert.benchmark_health import build_live_health_report

    collector.sample_once(reason="final")
    collector.stop()
    return build_live_health_report(
        collector.events,
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
        sample_interval_s=5.0,
    )


def _tensor_nbytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _train_host_stage_profiled(backend: str, runtime, grids, rows, *, device: str):
    import torch

    build_started = time.perf_counter()
    if backend == "magat":
        batch = _build_host_magat_batch(grids, rows, device=device)
        torch.cuda.synchronize(torch.device(device))
        build_s = time.perf_counter() - build_started
        h2d_bytes = sum(
            _tensor_nbytes(tensor)
            for tensor in (
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.ptr,
                batch.y,
                batch.arrived,
            )
        )
        train_started = time.perf_counter()
        loss = runtime.train_step_from_batch(batch)
        torch.cuda.synchronize(torch.device(device))
        train_s = time.perf_counter() - train_started
        return (
            float(loss.item()),
            rows.shape[0],
            {
                "host_builder_and_h2d_s": build_s,
                "train_step_s": train_s,
                "h2d_bytes": h2d_bytes,
                "d2h_bytes": 4,
            },
        )
    builder = _HostMapfGPTBuilder(
        grids, num_agents=rows.shape[0] // grids.shape[0], device=device
    )
    metrics = runtime.train_step(
        builder, torch.from_numpy(rows.astype(np.int16, copy=False))
    )
    torch.cuda.synchronize(torch.device(device))
    combined_s = time.perf_counter() - build_started
    h2d_bytes = sum(
        _tensor_nbytes(tensor)
        for tensor in (builder.tokens, builder.labels, builder.active_mask)
    )
    return (
        metrics["loss"],
        int(metrics["samples"]),
        {
            "host_builder_and_h2d_s": combined_s,
            "train_step_s": 0.0,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": 4,
        },
    )


def _train_host_stage(backend: str, runtime, grids, rows, *, device: str):
    loss, count, _ = _train_host_stage_profiled(
        backend, runtime, grids, rows, device=device
    )
    return loss, count


def _make_runtime(backend: str, *, device: str, seed: int, kwargs: dict[str, Any]):
    if backend == "magat":
        import torch

        from expert.fixed_magat_plus_runtime import MAGATRuntimeAdapter

        torch.manual_seed(int(seed))
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        return MAGATRuntimeAdapter(
            device=device,
            lr=float(kwargs.get("lr_start", 1e-3)),
            lr_end=float(kwargs.get("lr_end", 1e-6)),
            lr_scheduler=kwargs.get("lr_scheduler"),
            scheduler_total_steps=kwargs.get("scheduler_total_steps"),
            grad_clip_norm=kwargs.get("grad_clip_norm"),
            train_on_arrived_agents=bool(kwargs.get("train_on_arrived_agents", True)),
        )
    from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter

    return MapfGPTRuntimeAdapter(
        model_size=str(kwargs.get("model_size", "2M")),
        device=device,
        train_batch_size=int(kwargs["train_batch_size"]),
        microbatch_size=int(kwargs.get("microbatch_size", 256)),
        shuffle_capacity=int(kwargs.get("shuffle_capacity", 8192)),
        seed=int(seed),
        train_on_arrived_agents=bool(kwargs.get("train_on_arrived_agents", True)),
    )


def _make_checkpoint_manager(backend: str, kwargs: dict[str, Any]):
    from expert.checkpoint_manager import TopKCheckpointManager

    selection_mode = str(kwargs.get("checkpoint_selection_mode", "latest"))
    evaluator = None
    if selection_mode == "validation_accuracy":
        if backend == "magat":
            from mapf_cuda.training.topology_async import (
                _build_checkpoint_validation_evaluator,
            )

            evaluator, _ = _build_checkpoint_validation_evaluator(
                kwargs.get("validation_trajectories"), device=kwargs["device"]
            )
        else:
            from expert.mapf_gpt_validation import (
                build_frozen_mapf_gpt_validation_evaluator,
            )

            datasets = kwargs.get("validation_datasets") or []
            if datasets:
                evaluator = build_frozen_mapf_gpt_validation_evaluator(
                    datasets,
                    device=kwargs["device"],
                    batch_size=int(kwargs.get("validation_batch_size", 256)),
                )
        if evaluator is None:
            raise ValueError(
                f"{backend} validation artifacts are required for validation_accuracy selection"
            )
    return TopKCheckpointManager(
        kwargs.get("checkpoint_dir"),
        top_k=int(kwargs.get("top_k_checkpoints", 3)),
        save_interval_steps=int(kwargs.get("checkpoint_interval", 1000)),
        mode_label=f"{backend}_{kwargs.get('system_mode', 'b2')}",
        selection_mode=selection_mode,
        validation_evaluator=evaluator,
    )


def _record_checkpoint(
    manager,
    runtime,
    losses,
    *,
    samples: int,
    training_elapsed_s: float | None = None,
    force: bool = False,
):
    if not losses:
        return None
    extra_meta = {"samples_processed": int(samples)}
    if training_elapsed_s is not None:
        extra_meta["training_elapsed_s"] = float(training_elapsed_s)
    return manager.maybe_save(
        loss=float(losses[-1]),
        runtime=runtime,
        optimizer_step=len(losses),
        extra_meta=extra_meta,
        force=bool(force),
    )


def run_reference_training(*, backend: str, **kwargs) -> dict[str, Any]:
    import torch

    selected, grids = _selected_maps(kwargs["maps_path"], kwargs["map_names"])
    stage_rows = int(kwargs["num_agents"]) * len(selected)
    health_state, health_collector = _no_worker_health_start(stage_rows)
    losses = []
    start = time.perf_counter()
    try:
        stages = _generate_sequential_stages(
            backend,
            maps_path=kwargs["maps_path"],
            selected=selected,
            num_agents=kwargs["num_agents"],
            num_steps=kwargs["num_steps"],
            seed=kwargs["seed"],
            max_episode_steps=kwargs["max_episode_steps"],
        )
        runtime = _make_runtime(
            backend, device=kwargs["device"], seed=kwargs["seed"], kwargs=kwargs
        )
        checkpoint_manager = _make_checkpoint_manager(backend, kwargs)
        samples = 0
        for stage_index, rows in enumerate(stages, start=1):
            loss, count = _train_host_stage(
                backend, runtime, grids, rows, device=kwargs["device"]
            )
            if loss is not None:
                losses.append(float(loss))
            samples += int(count)
            _record_checkpoint(checkpoint_manager, runtime, losses, samples=samples)
            health_state.update(rows=stage_index * stage_rows, optimizer_steps=len(losses))
            health_collector.sample_once(reason="optimizer_progress")
        torch.cuda.synchronize(torch.device(kwargs["device"]))
        _record_checkpoint(
            checkpoint_manager, runtime, losses, samples=samples, force=True
        )
        wall_s = time.perf_counter() - start
        health = _no_worker_health_finish(health_collector)
    except BaseException:
        if health_collector.running:
            health_collector.stop()
        raise
    return _common_summary(
        backend=backend,
        mode="host_reference",
        kwargs=kwargs,
        wall_s=wall_s,
        samples=samples,
        losses=losses,
        worker_pids=[],
        worker_exitcodes=[],
        health=health,
        checkpoint_manager=checkpoint_manager,
    )


def run_compact_sync_training(*, backend: str, **kwargs) -> dict[str, Any]:
    import torch
    import grid_world_cpp as ext

    selected, grids = _selected_maps(kwargs["maps_path"], kwargs["map_names"])
    stage_rows = int(kwargs["num_agents"]) * len(selected)
    health_state, health_collector = _no_worker_health_start(stage_rows)
    losses = []
    start = time.perf_counter()
    try:
        stages = _generate_sequential_stages(
            backend,
            maps_path=kwargs["maps_path"],
            selected=selected,
            num_agents=kwargs["num_agents"],
            num_steps=kwargs["num_steps"],
            seed=kwargs["seed"],
            max_episode_steps=kwargs["max_episode_steps"],
        )
        device = kwargs["device"]
        from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator

        grids_cuda, _ = stack_grids_for_compiled_simulator(
            [grid for _, grid in selected], device=device
        )
        runtime = _make_runtime(backend, device=device, seed=kwargs["seed"], kwargs=kwargs)
        checkpoint_manager = _make_checkpoint_manager(backend, kwargs)
        if backend == "magat":
            builder = ext.StatelessGridWorldSimulator(grids_cuda, kwargs["num_agents"], 3)
            builder.pyg_builder_mode = str(kwargs.get("pyg_builder_mode", "local_gather"))
        else:
            builder = ext.MapfGPTObservationBuilder(grids_cuda, kwargs["num_agents"])
        samples = 0
        for stage_index, rows in enumerate(stages, start=1):
            raw = _pin_to_device(
                rows.astype(np.int16, copy=False), device=device, dtype=torch.int16
            )
            if backend == "magat":
                reset_rows = raw[raw[:, 7] != 0]
                if reset_rows.shape[0]:
                    builder.update_energy_maps(reset_rows, reset_rows.shape[0])
                builder.refresh_compact_state_from_raw_batch(raw)
                loss = runtime.train_step(builder, raw)
                losses.append(float(loss.item()))
                count = rows.shape[0]
            else:
                metrics = runtime.train_step(builder, raw)
                if metrics["loss"] is not None:
                    losses.append(float(metrics["loss"]))
                count = int(metrics["samples"])
            samples += count
            _record_checkpoint(checkpoint_manager, runtime, losses, samples=samples)
            health_state.update(rows=stage_index * stage_rows, optimizer_steps=len(losses))
            health_collector.sample_once(reason="optimizer_progress")
        torch.cuda.synchronize(torch.device(device))
        _record_checkpoint(
            checkpoint_manager, runtime, losses, samples=samples, force=True
        )
        wall_s = time.perf_counter() - start
        health = _no_worker_health_finish(health_collector)
    except BaseException:
        if health_collector.running:
            health_collector.stop()
        raise
    return _common_summary(
        backend=backend,
        mode="compact_cuda_sync",
        kwargs=kwargs,
        wall_s=wall_s,
        samples=samples,
        losses=losses,
        worker_pids=[],
        worker_exitcodes=[],
        health=health,
        checkpoint_manager=checkpoint_manager,
    )


def run_magat_compact_sync_matched_training(**kwargs) -> dict[str, Any]:
    """Run the compact MAGAT path with parallel experts and bounded backpressure.

    Four experts still produce one timestep in parallel.  The current stage is
    released only after synchronous H2D, CUDA reconstruction, and the optimizer
    step complete.  ``ring_buffer_steps=1`` is strict one-stage backpressure;
    larger values allow future stages to be produced concurrently without
    changing the consumer data path.
    """

    import torch
    import grid_world_cpp as ext

    from mapf_cuda.training.topology_async import (
        _topology_async_training_expert_worker_loop,
    )
    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator
    from expert.benchmark_training_health import TrainingHealthLifecycle
    from expert.expert_running import RingBuffer, resolve_expert_timeouts

    selected, _ = _selected_maps(kwargs["maps_path"], kwargs["map_names"])
    num_envs = len(selected)
    num_agents = int(kwargs["num_agents"])
    stage_rows = num_envs * num_agents
    ring_buffer_steps = int(kwargs.get("ring_buffer_steps", 1))
    if ring_buffer_steps <= 0:
        raise ValueError("ring_buffer_steps must be positive")
    ring = RingBuffer(
        stage_rows * ring_buffer_steps,
        8,
        num_envs=num_envs,
        agents_per_env=num_agents,
    )
    pipeline = _HostPipelineState(ring, stage_rows)
    optimizer_steps = 0
    samples = 0
    health = TrainingHealthLifecycle(
        expected_workers=num_envs,
        pipeline=pipeline,
        block_size=stage_rows,
        optimizer_steps_provider=lambda: optimizer_steps,
        samples_processed_provider=lambda: samples,
        gpu_device_index=0,
        expert_timeout_s=float(sum(resolve_expert_timeouts())),
        sample_interval_s=float(kwargs.get("health_sample_interval_s", 5.0)),
    )
    timing_manager = mp.Manager()
    worker_timings = timing_manager.list()
    processes = []
    for env_id, (map_name, _) in enumerate(selected):
        processes.append(
            mp.Process(
                target=_topology_async_training_expert_worker_loop,
                args=(env_id, ring, worker_timings),
                kwargs={
                    "maps_path": kwargs["maps_path"],
                    "map_name": map_name,
                    "num_agents": num_agents,
                    "num_steps": int(kwargs["num_steps"]),
                    "seed": int(kwargs["seed"]) + env_id,
                    "max_episode_steps": int(kwargs["max_episode_steps"]),
                    "health_handle": health.worker_handles[env_id],
                },
                daemon=True,
            )
        )
    health.register_processes(processes)
    health.start()
    losses = []
    consumer_wait_s = 0.0
    consumer_dma_sync_s = 0.0
    consumer_gpu_builder_s = 0.0
    consumer_train_step_s = 0.0
    h2d_bytes = 0
    d2h_bytes = 0
    stage_sha256s = []
    wall_budget = _resolve_training_wall_budget(kwargs)
    wall_budget_reached = False
    try:
        for process in processes:
            process.start()
        health.wait_for_workers_ready(timeout_s=120.0)
        device = kwargs["device"]
        grids_cuda, _ = stack_grids_for_compiled_simulator(
            [grid for _, grid in selected], device=device
        )
        runtime = _make_runtime(
            "magat", device=device, seed=kwargs["seed"], kwargs=kwargs
        )
        checkpoint_manager = _make_checkpoint_manager("magat", kwargs)
        builder = ext.StatelessGridWorldSimulator(grids_cuda, num_agents, 3)
        builder.pyg_builder_mode = str(kwargs.get("pyg_builder_mode", "local_gather"))
        builder.pyg_local_gather_impl = str(
            kwargs.get("pyg_local_gather_impl", "auto")
        )
        start = time.perf_counter()
        for stage_seq in range(int(kwargs["num_steps"])):
            wait_started = time.perf_counter()
            deadline = time.monotonic() + 120.0
            while not ring.is_stage_ready(stage_seq):
                failed = [
                    process.exitcode
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(f"compact-sync expert failure: {failed}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for a complete compact-sync stage"
                    )
                time.sleep(0.0005)
            consumer_wait_s += time.perf_counter() - wait_started
            begin, end = ring._stage_row_bounds(stage_seq)
            if kwargs.get("capture_stage_hashes", False):
                stage_sha256s.append(
                    hashlib.sha256(
                        ring.cpu_buffer[begin:end].contiguous().numpy().tobytes()
                    ).hexdigest()
                )

            stage_started = time.perf_counter()
            raw = ring.cpu_buffer[begin:end].to(device, non_blocking=False)
            torch.cuda.synchronize(torch.device(device))
            consumer_dma_sync_s += time.perf_counter() - stage_started
            h2d_bytes += _tensor_nbytes(raw)
            pipeline.dma_worker.dma_read_ptr = (stage_seq + 1) * stage_rows

            stage_started = time.perf_counter()
            reset_rows = raw[raw[:, 7] != 0]
            if reset_rows.shape[0]:
                builder.update_energy_maps(reset_rows, reset_rows.shape[0])
            builder.refresh_compact_state_from_raw_batch(raw)
            batch = runtime.build_batch(builder, raw, materialize_edges=False)
            torch.cuda.synchronize(torch.device(device))
            consumer_gpu_builder_s += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            loss = runtime.train_step_from_batch(batch)
            torch.cuda.synchronize(torch.device(device))
            consumer_train_step_s += time.perf_counter() - stage_started
            losses.append(float(loss.item()))
            d2h_bytes += 4
            optimizer_steps += 1
            samples += stage_rows
            _record_checkpoint(
                checkpoint_manager, runtime, losses, samples=samples
            )

            ring.release_stage(stage_seq)
            ring.shared_compute_ptr.value = (stage_seq + 1) * stage_rows
            pipeline.compute_ptr = (stage_seq + 1) * stage_rows
            if kwargs.get("inline_health_sampling", True):
                health.maybe_sample_progress(
                    force=optimizer_steps == 1 or optimizer_steps % 1000 == 0
                )
            if _training_wall_reached(start, wall_budget):
                wall_budget_reached = True
                break

        torch.cuda.synchronize(torch.device(device))
        wall_s = time.perf_counter() - start
        _record_checkpoint(
            checkpoint_manager, runtime, losses, samples=samples, force=True
        )
        health.release_workers()
        if wall_budget_reached:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
        else:
            for process in processes:
                process.join(timeout=15.0)
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError(
                    f"compact-sync worker exit failure: {[p.exitcode for p in processes]}"
                )
        health_report = health.finalize(reference_throughput=None)
    finally:
        health.release_workers()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        if health.running:
            health.abort(reason="training_aborted", reference_throughput=None)
        worker_timing_rows = sorted(
            list(worker_timings), key=lambda item: item["expert_id"]
        )
        timing_manager.shutdown()

    summary = _common_summary(
        backend="magat",
        mode="compact_cuda_sync",
        kwargs=kwargs,
        wall_s=wall_s,
        samples=samples,
        losses=losses,
        worker_pids=[int(process.pid) for process in processes],
        worker_exitcodes=[process.exitcode for process in processes],
        health=health_report,
        checkpoint_manager=checkpoint_manager,
    )
    summary.update(
        {
            "consumer_wait_s": consumer_wait_s,
            "consumer_dma_sync_s": consumer_dma_sync_s,
            "consumer_gpu_builder_s": consumer_gpu_builder_s,
            "consumer_train_step_s": consumer_train_step_s,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "worker_timings": worker_timing_rows,
            "ring_buffer_steps": ring_buffer_steps,
            "stage_sha256s": stage_sha256s,
            "max_training_wall_s": wall_budget,
            "wall_budget_reached": wall_budget_reached,
        }
    )
    return summary


def run_mapf_gpt_compact_sync_matched_training(**kwargs) -> dict[str, Any]:
    """Run MAPF-GPT compact-state training with one-stage backpressure."""

    import grid_world_cpp as ext
    import torch

    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator
    from expert.benchmark_training_health import TrainingHealthLifecycle
    from expert.expert_running import RingBuffer, resolve_expert_timeouts
    from expert.mapf_gpt_online import mapf_gpt_expert_worker_loop
    from expert.mapf_gpt_schema import MAPFGPT_FEATURE_DIM

    selected, _ = _selected_maps(kwargs["maps_path"], kwargs["map_names"])
    num_envs = len(selected)
    num_agents = int(kwargs["num_agents"])
    stage_rows = num_envs * num_agents
    ring = RingBuffer(
        stage_rows,
        MAPFGPT_FEATURE_DIM,
        num_envs=num_envs,
        agents_per_env=num_agents,
    )
    pipeline = _HostPipelineState(ring, stage_rows)
    optimizer_steps = 0
    samples = 0
    health = TrainingHealthLifecycle(
        expected_workers=num_envs,
        pipeline=pipeline,
        block_size=stage_rows,
        optimizer_steps_provider=lambda: optimizer_steps,
        samples_processed_provider=lambda: samples,
        gpu_device_index=0,
        expert_timeout_s=float(sum(resolve_expert_timeouts())),
        sample_interval_s=float(kwargs.get("health_sample_interval_s", 5.0)),
    )
    processes = []
    for env_id, (map_name, _) in enumerate(selected):
        processes.append(
            mp.Process(
                target=mapf_gpt_expert_worker_loop,
                args=(env_id, ring),
                kwargs={
                    "maps_path": kwargs["maps_path"],
                    "map_name": map_name,
                    "num_agents": num_agents,
                    "num_steps": int(kwargs["num_steps"]),
                    "seed": int(kwargs["seed"]) + env_id,
                    "max_episode_steps": int(kwargs["max_episode_steps"]),
                    "health_handle": health.worker_handles[env_id],
                },
                daemon=True,
            )
        )
    health.register_processes(processes)
    health.start()
    losses = []
    consumer_wait_s = 0.0
    consumer_dma_sync_s = 0.0
    consumer_pipeline_s = 0.0
    h2d_bytes = 0
    d2h_bytes = 0
    stage_sha256s = []
    try:
        for process in processes:
            process.start()
        health.wait_for_workers_ready(timeout_s=120.0)
        device = kwargs["device"]
        grids_cuda, _ = stack_grids_for_compiled_simulator(
            [grid for _, grid in selected], device=device
        )
        runtime = _make_runtime(
            "mapf_gpt", device=device, seed=kwargs["seed"], kwargs=kwargs
        )
        checkpoint_manager = _make_checkpoint_manager("mapf_gpt", kwargs)
        builder = ext.MapfGPTObservationBuilder(grids_cuda, num_agents)
        start = time.perf_counter()
        for stage_seq in range(int(kwargs["num_steps"])):
            wait_started = time.perf_counter()
            deadline = time.monotonic() + 120.0
            while not ring.is_stage_ready(stage_seq):
                failed = [
                    process.exitcode
                    for process in processes
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError(
                        f"MAPF-GPT compact-sync expert failure: {failed}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out waiting for a complete MAPF-GPT compact-sync stage"
                    )
                time.sleep(0.0005)
            consumer_wait_s += time.perf_counter() - wait_started
            begin, end = ring._stage_row_bounds(stage_seq)
            if kwargs.get("capture_stage_hashes", False):
                stage_sha256s.append(
                    hashlib.sha256(
                        ring.cpu_buffer[begin:end].contiguous().numpy().tobytes()
                    ).hexdigest()
                )

            stage_started = time.perf_counter()
            raw = ring.cpu_buffer[begin:end].to(device, non_blocking=False)
            torch.cuda.synchronize(torch.device(device))
            consumer_dma_sync_s += time.perf_counter() - stage_started
            h2d_bytes += _tensor_nbytes(raw)
            pipeline.dma_worker.dma_read_ptr = (stage_seq + 1) * stage_rows

            stage_started = time.perf_counter()
            metrics = runtime.train_step(builder, raw)
            torch.cuda.synchronize(torch.device(device))
            consumer_pipeline_s += time.perf_counter() - stage_started
            if metrics["loss"] is not None:
                losses.append(float(metrics["loss"]))
                optimizer_steps += 1
            samples += int(metrics["samples"])
            _record_checkpoint(
                checkpoint_manager, runtime, losses, samples=samples
            )

            ring.release_stage(stage_seq)
            ring.shared_compute_ptr.value = (stage_seq + 1) * stage_rows
            pipeline.compute_ptr = (stage_seq + 1) * stage_rows
            if kwargs.get("inline_health_sampling", True):
                health.maybe_sample_progress(
                    force=optimizer_steps == 1 or optimizer_steps % 1000 == 0
                )

        torch.cuda.synchronize(torch.device(device))
        wall_s = time.perf_counter() - start
        _record_checkpoint(
            checkpoint_manager, runtime, losses, samples=samples, force=True
        )
        health.release_workers()
        for process in processes:
            process.join(timeout=15.0)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(
                "MAPF-GPT compact-sync worker exit failure: "
                f"{[process.exitcode for process in processes]}"
            )
        health_report = health.finalize(reference_throughput=None)
    finally:
        health.release_workers()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        if health.running:
            health.abort(reason="training_aborted", reference_throughput=None)

    summary = _common_summary(
        backend="mapf_gpt",
        mode="compact_cuda_sync",
        kwargs=kwargs,
        wall_s=wall_s,
        samples=samples,
        losses=losses,
        worker_pids=[int(process.pid) for process in processes],
        worker_exitcodes=[process.exitcode for process in processes],
        health=health_report,
        checkpoint_manager=checkpoint_manager,
    )
    summary.update(
        {
            "consumer_wait_s": consumer_wait_s,
            "consumer_dma_sync_s": consumer_dma_sync_s,
            "consumer_pipeline_s": consumer_pipeline_s,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "transfer_accounting": "raw_payload_only_control_excluded",
            "sync_ring_buffer_steps": 1,
            "stage_sha256s": stage_sha256s,
        }
    )
    return summary


def run_strong_online_training(*, backend: str, **kwargs) -> dict[str, Any]:
    import torch

    from expert.benchmark_training_health import TrainingHealthLifecycle
    from expert.expert_running import RingBuffer, resolve_expert_timeouts
    from expert.mapf_gpt_schema import MAPFGPT_FEATURE_DIM

    selected, grids = _selected_maps(kwargs["maps_path"], kwargs["map_names"])
    num_envs = len(selected)
    num_agents = int(kwargs["num_agents"])
    stage_rows = num_envs * num_agents
    feature_dim = 8 if backend == "magat" else MAPFGPT_FEATURE_DIM
    ring_buffer_steps = int(kwargs.get("ring_buffer_steps", 4))
    if ring_buffer_steps <= 0:
        raise ValueError("ring_buffer_steps must be positive")
    ring = RingBuffer(
        stage_rows * ring_buffer_steps,
        feature_dim,
        num_envs=num_envs,
        agents_per_env=num_agents,
    )
    pipeline = _HostPipelineState(ring, stage_rows)
    optimizer_steps = 0
    samples = 0
    health = TrainingHealthLifecycle(
        expected_workers=num_envs,
        pipeline=pipeline,
        block_size=stage_rows,
        optimizer_steps_provider=lambda: optimizer_steps,
        samples_processed_provider=lambda: samples,
        gpu_device_index=0,
        expert_timeout_s=float(sum(resolve_expert_timeouts())),
        sample_interval_s=float(kwargs.get("health_sample_interval_s", 5.0)),
    )
    processes = []
    timing_manager = mp.Manager()
    worker_timings = timing_manager.list()
    for env_id, (map_name, _) in enumerate(selected):
        if backend == "magat":
            from mapf_cuda.training.topology_async import (
                _topology_async_training_expert_worker_loop,
            )

            target = _topology_async_training_expert_worker_loop
        else:
            from expert.mapf_gpt_online import mapf_gpt_expert_worker_loop

            target = mapf_gpt_expert_worker_loop
        processes.append(
            mp.Process(
                target=target,
                args=(
                    (env_id, ring, worker_timings)
                    if backend == "magat"
                    else (env_id, ring)
                ),
                kwargs={
                    "maps_path": kwargs["maps_path"],
                    "map_name": map_name,
                    "num_agents": num_agents,
                    "num_steps": int(kwargs["num_steps"]),
                    "seed": int(kwargs["seed"]) + env_id,
                    "max_episode_steps": int(kwargs["max_episode_steps"]),
                    "health_handle": health.worker_handles[env_id],
                },
                daemon=True,
            )
        )
    health.register_processes(processes)
    health.start()
    losses = []
    consumer_wait_s = 0.0
    consumer_host_builder_and_h2d_s = 0.0
    consumer_train_step_s = 0.0
    h2d_bytes = 0
    d2h_bytes = 0
    stage_sha256s = []
    wall_budget = _resolve_training_wall_budget(kwargs)
    checkpoint_wall_interval_s = _resolve_checkpoint_wall_interval(kwargs)
    next_checkpoint_wall_s = checkpoint_wall_interval_s
    wall_budget_reached = False
    try:
        for process in processes:
            process.start()
        health.wait_for_workers_ready(timeout_s=120.0)
        runtime = _make_runtime(
            backend, device=kwargs["device"], seed=kwargs["seed"], kwargs=kwargs
        )
        from expert.checkpoint_manager import model_state_sha256

        initial_model_state_sha256 = model_state_sha256(runtime.model)
        checkpoint_manager = _make_checkpoint_manager(backend, kwargs)
        start = time.perf_counter()
        for stage_seq in range(int(kwargs["num_steps"])):
            wait_started = time.perf_counter()
            deadline = time.monotonic() + 120.0
            while not ring.is_stage_ready(stage_seq):
                failed = [p.exitcode for p in processes if p.exitcode not in (None, 0)]
                if failed:
                    raise RuntimeError(f"strong-online expert failure: {failed}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for a complete strong-online stage")
                time.sleep(0.0005)
            consumer_wait_s += time.perf_counter() - wait_started
            begin, end = ring._stage_row_bounds(stage_seq)
            rows = ring.cpu_buffer[begin:end].numpy().astype(np.uint16, copy=True)
            if kwargs.get("capture_stage_hashes", False):
                stage_sha256s.append(hashlib.sha256(rows.tobytes()).hexdigest())
            pipeline.dma_worker.dma_read_ptr = (stage_seq + 1) * stage_rows
            loss, count, stage_metrics = _train_host_stage_profiled(
                backend, runtime, grids, rows, device=kwargs["device"]
            )
            consumer_host_builder_and_h2d_s += stage_metrics[
                "host_builder_and_h2d_s"
            ]
            consumer_train_step_s += stage_metrics["train_step_s"]
            h2d_bytes += int(stage_metrics["h2d_bytes"])
            d2h_bytes += int(stage_metrics["d2h_bytes"])
            if loss is not None:
                losses.append(float(loss))
                optimizer_steps += 1
            samples += int(count)
            training_elapsed_s = time.perf_counter() - start
            if checkpoint_wall_interval_s is None:
                _record_checkpoint(
                    checkpoint_manager,
                    runtime,
                    losses,
                    samples=samples,
                    training_elapsed_s=training_elapsed_s,
                )
            elif training_elapsed_s >= next_checkpoint_wall_s:
                _record_checkpoint(
                    checkpoint_manager,
                    runtime,
                    losses,
                    samples=samples,
                    training_elapsed_s=training_elapsed_s,
                    force=True,
                )
                next_checkpoint_wall_s = _advance_wall_checkpoint_deadline(
                    next_checkpoint_wall_s,
                    checkpoint_wall_interval_s,
                    time.perf_counter() - start,
                )
            ring.release_stage(stage_seq)
            ring.shared_compute_ptr.value = (stage_seq + 1) * stage_rows
            pipeline.compute_ptr = (stage_seq + 1) * stage_rows
            if kwargs.get("inline_health_sampling", True):
                health.maybe_sample_progress(
                    force=optimizer_steps == 1 or optimizer_steps % 1000 == 0
                )
            if _training_wall_reached(start, wall_budget):
                wall_budget_reached = True
                break
        torch.cuda.synchronize(torch.device(kwargs["device"]))
        wall_s = time.perf_counter() - start
        _record_checkpoint(
            checkpoint_manager,
            runtime,
            losses,
            samples=samples,
            training_elapsed_s=wall_s,
            force=True,
        )
        health.release_workers()
        if wall_budget_reached:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
        else:
            for process in processes:
                process.join(timeout=15.0)
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError(
                    f"strong-online worker exit failure: {[p.exitcode for p in processes]}"
                )
        health_report = health.finalize(reference_throughput=None)
    finally:
        health.release_workers()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        if health.running:
            health.abort(reason="training_aborted", reference_throughput=None)
        worker_timing_rows = sorted(
            list(worker_timings), key=lambda item: item["expert_id"]
        )
        timing_manager.shutdown()
    summary = _common_summary(
        backend=backend,
        mode="host_strong_online",
        kwargs=kwargs,
        wall_s=wall_s,
        samples=samples,
        losses=losses,
        worker_pids=[int(process.pid) for process in processes],
        worker_exitcodes=[process.exitcode for process in processes],
        health=health_report,
        checkpoint_manager=checkpoint_manager,
    )
    summary.update(
        {
            "consumer_wait_s": consumer_wait_s,
            "consumer_host_builder_and_h2d_s": consumer_host_builder_and_h2d_s,
            "consumer_train_step_s": consumer_train_step_s,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "checkpoint_history": checkpoint_manager.checkpoint_history(),
            "stage_sha256s": stage_sha256s,
            "worker_timings": worker_timing_rows,
            "ring_buffer_steps": ring_buffer_steps,
            "max_training_wall_s": wall_budget,
            "wall_budget_reached": wall_budget_reached,
            "checkpoint_interval_s": checkpoint_wall_interval_s,
            "initial_model_state_sha256": initial_model_state_sha256,
        }
    )
    return summary


def _common_summary(
    *,
    backend,
    mode,
    kwargs,
    wall_s,
    samples,
    losses,
    worker_pids,
    worker_exitcodes,
    health,
    checkpoint_manager,
):
    return {
        "accepted": bool(health.get("validation", {}).get("valid"))
        or bool(kwargs.get("max_training_wall_s") and losses),
        "backend": backend,
        "mode": mode,
        "num_agents": int(kwargs["num_agents"]),
        "num_experts": len(kwargs["map_names"]),
        "num_steps": int(kwargs["num_steps"]),
        "num_optimizer_steps": len(losses),
        "samples_processed": int(samples),
        "total_wall_s": float(wall_s),
        "samples_s": float(samples / wall_s),
        "mean_loss": None if not losses else float(np.mean(losses)),
        "final_loss": None if not losses else float(losses[-1]),
        "loss_history": [float(value) for value in losses],
        "worker_pids": list(worker_pids),
        "worker_exitcodes": list(worker_exitcodes),
        "health_report": health,
        "checkpoint_selection_mode": checkpoint_manager.selection_mode,
        "saved_checkpoints": checkpoint_manager.snapshot(),
        "checkpoint_history": checkpoint_manager.checkpoint_history(),
        "validation_metrics": checkpoint_manager.last_validation_metrics(),
        "max_training_wall_s": (
            None
            if kwargs.get("max_training_wall_s") is None
            else float(kwargs["max_training_wall_s"])
        ),
    }


def run_magat_reference_engine(**kwargs):
    return run_reference_training(backend="magat", **kwargs)


def run_magat_strong_online_engine(**kwargs):
    return run_strong_online_training(backend="magat", **kwargs)


def run_magat_compact_sync_engine(**kwargs):
    return run_compact_sync_training(backend="magat", **kwargs)


def run_magat_compact_sync_matched_engine(**kwargs):
    return run_magat_compact_sync_matched_training(**kwargs)


def run_mapf_gpt_reference_engine(**kwargs):
    return run_reference_training(backend="mapf_gpt", **kwargs)


def run_mapf_gpt_strong_online_engine(**kwargs):
    return run_strong_online_training(backend="mapf_gpt", **kwargs)


def run_mapf_gpt_compact_sync_engine(**kwargs):
    return run_compact_sync_training(backend="mapf_gpt", **kwargs)


def run_mapf_gpt_compact_sync_matched_engine(**kwargs):
    return run_mapf_gpt_compact_sync_matched_training(**kwargs)


__all__ = [
    "run_magat_compact_sync_engine",
    "run_magat_compact_sync_matched_engine",
    "run_magat_reference_engine",
    "run_magat_strong_online_engine",
    "run_mapf_gpt_compact_sync_engine",
    "run_mapf_gpt_compact_sync_matched_engine",
    "run_mapf_gpt_reference_engine",
    "run_mapf_gpt_strong_online_engine",
]
