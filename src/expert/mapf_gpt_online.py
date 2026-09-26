"""Opt-in online LaCAM -> CUDA -> MAPF-GPT training pipeline.

This module deliberately does not alter the default MAGAT benchmark entrypoint.
Its producers use a dedicated 13-column row contract while reusing the existing
stage-aligned ring buffer and pinned-memory DMA implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from expert.expert_running import (
    DMAWorker,
    ExtremeMAPFPipeline,
    LacamExpertPolicy,
    RingBuffer,
    configure_cpu_thread_limits,
    initial_refresh_flags,
    no_refresh_flags,
    resolve_expert_timeouts,
)
from expert.mapf_gpt_runtime import MapfGPTRuntimeAdapter
from expert.mapf_gpt_schema import (
    MAPFGPT_FEATURE_DIM,
    ActionHistory,
    build_mapf_gpt_rows,
)
from expert.checkpoint_manager import TopKCheckpointManager
from expert.mapf_gpt_validation import build_frozen_mapf_gpt_validation_evaluator
from expert.benchmark_training_health import TrainingHealthLifecycle
from mapf_cuda.observability.pipeline import (
    device_index as _device_index_from_string,
    pipeline_snapshot as _pipeline_health_snapshot,
)


DEFAULT_MAP_NAMES = (
    "mazes-s0_wc8_od55",
    "mazes-s1_wc3_od70",
    "mazes-s27_wc7_od60",
    "mazes-s28_wc2_od35",
)


def _counter_delta(current: int, initial: int, *, name: str) -> int:
    delta = int(current) - int(initial)
    if delta < 0:
        raise RuntimeError(
            f"{name} regressed across resumed run: initial={initial}, current={current}"
        )
    return delta


class MapfGPTStepBuilder:
    """Own one environment's five-action history and materialize raw rows."""

    def __init__(
        self,
        *,
        env_id: int,
        num_agents: int,
        coordinate_offset: int = 5,
    ):
        self.env_id = int(env_id)
        self.num_agents = int(num_agents)
        self.coordinate_offset = int(coordinate_offset)
        self.history = ActionHistory(self.num_agents)

    def build_rows(self, observations, actions, *, refresh_flags) -> np.ndarray:
        if len(observations) != self.num_agents:
            raise ValueError(
                f"expected {self.num_agents} observations, got {len(observations)}"
            )
        return build_mapf_gpt_rows(
            observations,
            actions,
            env_id=self.env_id,
            refresh_flags=refresh_flags,
            history=self.history.snapshot(),
            coordinate_offset=self.coordinate_offset,
        )

    def record_actions(self, actions) -> None:
        self.history.append(actions)

    def reset(self) -> None:
        self.history.reset()


class MapfGPTOnlinePipeline(ExtremeMAPFPipeline):
    """Stage-aligned consumer that bypasses all MAGAT/PyG compute paths."""

    def __init__(
        self,
        builder,
        runtime: MapfGPTRuntimeAdapter,
        *,
        capacity: int,
        batch_threshold: int,
        device: str = "cuda:0",
    ):
        super().__init__(
            builder,
            capacity=capacity,
            batch_threshold=batch_threshold,
            feature_dim=MAPFGPT_FEATURE_DIM,
            device=device,
        )
        self.runtime = runtime

    def initialize(self):
        """Initialize only the generic ring/DMA state used by MAPF-GPT."""

        self.agents_per_env, self.num_envs = self._infer_stage_layout()
        if self.agents_per_env is None or self.num_envs is None:
            raise ValueError("MAPF-GPT builder must expose a non-empty pyg_ptr stage layout")
        self.stage_mode = True
        stage_rows = self.agents_per_env * self.num_envs
        if self.batch_threshold != stage_rows:
            raise ValueError(
                "MAPF-GPT online pipeline requires one complete frontier per batch: "
                f"expected {stage_rows}, got {self.batch_threshold}"
            )
        self.ring_buffer = RingBuffer(
            self.capacity,
            MAPFGPT_FEATURE_DIM,
            num_envs=self.num_envs,
            agents_per_env=self.agents_per_env,
        )
        self.gpu_buffer = torch.zeros(
            (self.capacity, MAPFGPT_FEATURE_DIM),
            dtype=torch.int16,
            device=self.device,
        )
        self.stage_batch_buffer = torch.empty(
            (stage_rows, MAPFGPT_FEATURE_DIM),
            dtype=torch.int16,
            device=self.device,
        )
        self.gpu_handler = None
        self.dma_worker = DMAWorker(
            self.ring_buffer,
            self.gpu_buffer,
            self.batch_threshold,
            str(self.device),
        )
        self.dma_event = self.dma_worker.dma_event
        self.compute_stream = torch.cuda.Stream(device=self.device, priority=-1)
        self.graph_rows = int(self.cuda_simulator.tokens.shape[0])
        return self

    def consume_once(self, *, capture_stage_hash: bool = False):
        if self.dma_worker is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        if not self.has_env_aligned_batch(self.num_envs):
            return None
        stage_rows = self.agents_per_env * self.num_envs
        if self.dma_worker.dma_read_ptr - self.compute_ptr < stage_rows:
            return None
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_event(self.dma_event)
            raw_stage = self.extract_env_aligned_batch(
                stage_rows,
                agents_per_env=self.agents_per_env,
                num_envs=self.num_envs,
            )
            if raw_stage is None:
                return None
            stage_sha256 = None
            if capture_stage_hash:
                stage_sha256 = hashlib.sha256(
                    raw_stage.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
            metrics = self.runtime.train_step(self.cuda_simulator, raw_stage)
            if stage_sha256 is not None:
                metrics["stage_sha256"] = stage_sha256
            return metrics


def mapf_gpt_expert_worker_loop(
    env_id: int,
    ring_buffer,
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
    """Generate standard-MAPF labels with history ordered before current action."""

    from mapf_cuda.training.topology_async import _build_pogema_topology_map_env
    from expert.expert_running import put_maps_into_registry

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
    refresh_flags = initial_refresh_flags(num_agents)
    row_builder = MapfGPTStepBuilder(env_id=env_id, num_agents=num_agents)
    resolved_expert_timeouts = resolve_expert_timeouts(expert_timeouts)
    if health_handle is not None:
        health_handle.mark_ready()

    for _ in range(int(num_steps)):
        if health_handle is not None:
            health_handle.begin_expert_call(
                timeout_s=float(sum(resolved_expert_timeouts))
            )
        try:
            actions = policy.act(observations)
        finally:
            if health_handle is not None:
                health_handle.finish_expert_call()
        rows = row_builder.build_rows(
            observations, actions, refresh_flags=refresh_flags
        )
        ring_buffer.reserve_and_write(rows)
        if health_handle is not None:
            health_handle.mark_published()
        row_builder.record_actions(actions)

        _, _, terminated, truncated, _ = env.step(
            actions.astype(np.int64, copy=False)
        )
        episode_done = bool(
            np.logical_or(
                np.asarray(terminated, dtype=bool),
                np.asarray(truncated, dtype=bool),
            ).all()
        )
        if episode_done:
            env.reset()
            observations = env.env.unwrapped._obs()
            policy.reset_states(env)
            row_builder.reset()
            refresh_flags = initial_refresh_flags(num_agents)
        else:
            observations = env.env.unwrapped._obs()
            refresh_flags = no_refresh_flags(num_agents)
    if health_handle is not None:
        health_handle.mark_complete_and_wait()


def run_topology_mapf_gpt_online(
    *,
    num_steps: int,
    num_agents: int,
    maps_path: str,
    map_names: tuple[str, ...] | list[str],
    model_size: str = "2M",
    train_batch_size: int | None = None,
    microbatch_size: int | None = None,
    shuffle_capacity: int = 8192,
    ring_capacity_steps: int = 4,
    max_episode_steps: int = 256,
    seed: int = 42,
    device: str = "cuda:0",
    checkpoint: str | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_interval: int = 1000,
    top_k_checkpoints: int = 3,
    checkpoint_selection_mode: str = "latest",
    validation_datasets: tuple[str, ...] | list[str] | None = None,
    validation_batch_size: int = 256,
    resume_checkpoint: str | None = None,
    stall_timeout_s: float = 120.0,
    train_on_arrived_agents: bool = True,
    capture_stage_hashes: bool = False,
) -> dict[str, object]:
    """Run the explicit MAPF-GPT topology training path and return evidence."""

    import grid_world_cpp as ext
    from mapf_cuda.training.topology_async import (
        _select_topology_training_maps,
    )
    from mapf_cuda.simulation.grids import stack_grids_for_compiled_simulator

    if num_steps <= 0 or num_agents <= 0:
        raise ValueError("num_steps and num_agents must be positive")
    selected = _select_topology_training_maps(
        maps_path=maps_path,
        num_experts=len(map_names),
        map_names=map_names,
    )
    grids_cuda, _ = stack_grids_for_compiled_simulator(
        [grid for _, grid in selected], device=device
    )
    builder = ext.MapfGPTObservationBuilder(grids_cuda, num_agents)
    stage_rows = num_agents * len(selected)
    if train_batch_size is None:
        train_batch_size = stage_rows
    runtime = MapfGPTRuntimeAdapter(
        model_size=model_size,
        device=device,
        train_batch_size=train_batch_size,
        microbatch_size=microbatch_size,
        shuffle_capacity=max(shuffle_capacity, train_batch_size),
        seed=seed,
        train_on_arrived_agents=train_on_arrived_agents,
    )
    resume_payload = None
    if resume_checkpoint is not None:
        resume_path = str(Path(resume_checkpoint).expanduser().resolve())
        resume_payload = runtime.load_checkpoint(resume_path)
    else:
        resume_path = None

    resolved_validation_datasets = [
        str(Path(path).expanduser().resolve()) for path in (validation_datasets or [])
    ]
    selection_mode = str(checkpoint_selection_mode)
    validation_evaluator = None
    if selection_mode == "validation_accuracy":
        if not resolved_validation_datasets:
            raise ValueError(
                "validation_datasets are required when "
                "checkpoint_selection_mode='validation_accuracy'"
            )
        validation_evaluator = build_frozen_mapf_gpt_validation_evaluator(
            resolved_validation_datasets,
            device=device,
            batch_size=validation_batch_size,
        )
    checkpoint_manager = TopKCheckpointManager(
        checkpoint_dir,
        top_k=top_k_checkpoints,
        save_interval_steps=checkpoint_interval,
        mode_label="mapf_gpt_online",
        selection_mode=selection_mode,
        validation_evaluator=validation_evaluator,
    )
    initial_optimizer_steps = runtime.optimizer_steps
    initial_samples_processed = runtime.samples_processed
    initial_supervised_samples_processed = runtime.supervised_samples_processed
    initial_stage_samples_seen = runtime.stage_samples_seen
    pipeline = MapfGPTOnlinePipeline(
        builder,
        runtime,
        capacity=stage_rows * max(1, int(ring_capacity_steps)),
        batch_threshold=stage_rows,
        device=device,
    ).initialize()

    processes = []
    losses = []
    stages_consumed = 0
    consumer_wait_s = 0.0
    consumer_pipeline_s = 0.0
    stage_sha256s = []
    dma_thread = None
    health_report = None
    start = time.perf_counter()
    deadline = start + max(60.0, float(num_steps * len(selected) * 60))
    last_progress = start
    health_lifecycle = TrainingHealthLifecycle(
        expected_workers=len(selected),
        pipeline=pipeline,
        block_size=stage_rows,
        optimizer_steps_provider=lambda: runtime.optimizer_steps,
        samples_processed_provider=lambda: runtime.samples_processed,
        gpu_device_index=_device_index_from_string(device),
        expert_timeout_s=float(sum(resolve_expert_timeouts())),
    )
    try:
        for env_id, (map_name, _) in enumerate(selected):
            process = mp.Process(
                target=mapf_gpt_expert_worker_loop,
                args=(env_id, pipeline.ring_buffer),
                kwargs={
                    "maps_path": maps_path,
                    "map_name": map_name,
                    "num_agents": num_agents,
                    "num_steps": num_steps,
                    "seed": seed + env_id,
                    "max_episode_steps": max_episode_steps,
                    "health_handle": health_lifecycle.worker_handles[env_id],
                },
                daemon=True,
            )
            processes.append(process)

        health_lifecycle.register_processes(processes)
        health_lifecycle.start()
        dma_thread = pipeline.dma_worker.start()
        for process in processes:
            process.start()
        health_lifecycle.wait_for_workers_ready(timeout_s=120.0)

        worker_pids = [int(process.pid) for process in processes]
        workers_alive_after_start = sum(process.is_alive() for process in processes)

        while stages_consumed < num_steps:
            now = time.perf_counter()
            if now > deadline:
                raise TimeoutError("timed out waiting for MAPF-GPT online stages")
            if now - last_progress > float(stall_timeout_s):
                stats = pipeline.get_stats()
                raise TimeoutError(
                    "MAPF-GPT pipeline made no optimizer progress for "
                    f"{stall_timeout_s}s; worker_exitcodes="
                    f"{[process.exitcode for process in processes]}, stats={stats}"
                )
            failed = [
                process.exitcode
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"MAPF-GPT expert worker failed: exitcodes={failed}")
            consume_started = time.perf_counter()
            metrics = (
                pipeline.consume_once(capture_stage_hash=True)
                if capture_stage_hashes
                else pipeline.consume_once()
            )
            if metrics is None:
                consumer_wait_s += time.perf_counter() - consume_started
                if all(not process.is_alive() for process in processes):
                    raise RuntimeError(
                        "all MAPF-GPT workers exited before a complete stage was consumable"
                    )
                time.sleep(0.0005)
                continue
            consumer_pipeline_s += time.perf_counter() - consume_started
            stages_consumed += 1
            if capture_stage_hashes:
                stage_sha256s.append(str(metrics.pop("stage_sha256")))
            if not metrics["skipped"]:
                current_loss = float(metrics["loss"])
                losses.append(current_loss)
                checkpoint_manager.maybe_save(
                    loss=current_loss,
                    runtime=runtime,
                    optimizer_step=runtime.optimizer_steps,
                    extra_meta={
                        "map_names": [name for name, _ in selected],
                        "producer_resume_semantics": "new_expert_stream",
                    },
                )
            last_progress = time.perf_counter()
            health_lifecycle.maybe_sample_progress(
                force=runtime.optimizer_steps == initial_optimizer_steps + 1
                or runtime.optimizer_steps % 1000 == 0
            )

        health_lifecycle.release_workers()
        for process in processes:
            process.join(timeout=5.0)
        bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
        if bad_exits:
            raise RuntimeError(f"MAPF-GPT expert worker exit failure: {bad_exits}")

        if (
            checkpoint_manager.enabled
            and runtime.optimizer_steps > initial_optimizer_steps
            and losses
        ):
            checkpoint_manager.maybe_save(
                loss=losses[-1],
                runtime=runtime,
                optimizer_step=runtime.optimizer_steps,
                extra_meta={
                    "map_names": [name for name, _ in selected],
                    "producer_resume_semantics": "new_expert_stream",
                },
                force=True,
            )
        metadata = runtime.checkpoint_metadata()
        pipeline.dma_read_ptr = pipeline.dma_worker.dma_read_ptr
        pipeline_stats = pipeline.get_stats()
        wall_s = time.perf_counter() - start
        health_snapshot = _pipeline_health_snapshot(
            pipeline,
            fallback_block_size=stage_rows,
        )
        worker_exitcodes = [process.exitcode for process in processes]
        health_report = health_lifecycle.finalize(reference_throughput=None)
        validation_metrics = checkpoint_manager.last_validation_metrics()
        run_optimizer_steps = _counter_delta(
            runtime.optimizer_steps,
            initial_optimizer_steps,
            name="optimizer_steps",
        )
        run_samples_processed = _counter_delta(
            runtime.samples_processed,
            initial_samples_processed,
            name="samples_processed",
        )
        run_supervised_samples_processed = _counter_delta(
            runtime.supervised_samples_processed,
            initial_supervised_samples_processed,
            name="supervised_samples_processed",
        )
        run_stage_samples_seen = _counter_delta(
            runtime.stage_samples_seen,
            initial_stage_samples_seen,
            name="stage_samples_seen",
        )
        cuda_diagnostics = [
            int(value) for value in builder.diagnostics.detach().cpu().tolist()
        ]
        summary = {
            **metadata,
            "num_steps": int(num_steps),
            "num_agents": int(num_agents),
            "num_envs": len(selected),
            "map_names": [name for name, _ in selected],
            "stages_consumed": stages_consumed,
            "run_optimizer_steps": run_optimizer_steps,
            "run_samples_processed": run_samples_processed,
            "run_supervised_samples_processed": run_supervised_samples_processed,
            "run_stage_samples_seen": run_stage_samples_seen,
            "mean_loss": float(np.mean(losses)) if losses else None,
            "final_loss": float(losses[-1]) if losses else None,
            "loss_history": [float(loss) for loss in losses],
            "wall_s": wall_s,
            "total_wall_s": wall_s,
            "samples_s": float(run_samples_processed / wall_s),
            "num_optimizer_steps": run_optimizer_steps,
            "samples_processed": run_samples_processed,
            "worker_pids": worker_pids,
            "workers_started": len(processes),
            "workers_alive_after_start": workers_alive_after_start,
            "worker_exitcodes": worker_exitcodes,
            "ring_reserve_ptr": int(health_snapshot["reserve_ptr"]),
            "dma_read_ptr": int(health_snapshot["dma_ptr"]),
            "compute_ptr": int(health_snapshot["compute_ptr"]),
            "cuda_diagnostics": cuda_diagnostics,
            "checkpoint_selection_mode": selection_mode,
            "checkpoint_interval": int(checkpoint_interval),
            "checkpoint_inventory": checkpoint_manager.snapshot(),
            "checkpoint_history": (
                checkpoint_manager.checkpoint_history()
                if hasattr(checkpoint_manager, "checkpoint_history")
                else []
            ),
            "validation_datasets": resolved_validation_datasets,
            "validation_metrics": validation_metrics,
            "consumer_wait_s": consumer_wait_s,
            "consumer_pipeline_s": consumer_pipeline_s,
            "h2d_bytes": int(run_stage_samples_seen)
            * MAPFGPT_FEATURE_DIM
            * 2,
            "d2h_bytes": 0,
            "transfer_accounting": "raw_payload_only_control_excluded",
            "stage_sha256s": stage_sha256s,
            "accepted": bool(
                health_report.get("validation", {}).get("valid")
            ),
            "health_report": health_report,
            "resume_checkpoint": resume_path,
            "consumer_training_state_restored": bool(
                resume_payload is not None and resume_payload.get("runtime_state") is not None
            ),
            "producer_resume_semantics": "new_expert_stream",
        }
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint).expanduser().resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": runtime.model.state_dict(),
                    "optimizer": runtime.optimizer.state_dict(),
                    "runtime_state": runtime.checkpoint_state(),
                    "metadata": summary,
                },
                checkpoint_path,
            )
            summary["checkpoint"] = str(checkpoint_path)
        return summary
    finally:
        pipeline.shutdown()
        health_lifecycle.release_workers()
        if dma_thread is not None:
            dma_thread.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
        if health_lifecycle.running:
            health_lifecycle.abort(
                reason="training_aborted", reference_throughput=None
            )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Opt-in online standard-MAPF training for MAPF-GPT"
    )
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=16)
    parser.add_argument("--maps-path", default="maps/maps.yaml")
    parser.add_argument("--maps", nargs="+", default=list(DEFAULT_MAP_NAMES[:1]))
    parser.add_argument("--model-size", choices=("2M", "6M", "85M"), default="2M")
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--shuffle-capacity", type=int, default=8192)
    parser.add_argument("--ring-capacity-steps", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--top-k-checkpoints", type=int, default=3)
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=("latest", "validation_accuracy"),
        default="latest",
    )
    parser.add_argument("--validation-datasets", default="")
    parser.add_argument("--validation-batch-size", type=int, default=256)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--stall-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--train-on-arrived-agents",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    summary = run_topology_mapf_gpt_online(
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        maps_path=args.maps_path,
        map_names=args.maps,
        model_size=args.model_size,
        train_batch_size=args.train_batch_size,
        microbatch_size=args.microbatch_size,
        shuffle_capacity=args.shuffle_capacity,
        ring_capacity_steps=args.ring_capacity_steps,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        top_k_checkpoints=args.top_k_checkpoints,
        checkpoint_selection_mode=args.checkpoint_selection_mode,
        validation_datasets=[
            path for path in args.validation_datasets.split(",") if path
        ],
        validation_batch_size=args.validation_batch_size,
        resume_checkpoint=args.resume_checkpoint,
        stall_timeout_s=args.stall_timeout_s,
        train_on_arrived_agents=args.train_on_arrived_agents,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAP_NAMES",
    "MapfGPTOnlinePipeline",
    "MapfGPTStepBuilder",
    "mapf_gpt_expert_worker_loop",
    "run_topology_mapf_gpt_online",
]
