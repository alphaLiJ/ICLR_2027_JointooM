"""Live runtime-health lifecycle shared by asynchronous training backends."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from expert.benchmark_health import HealthCollector, build_live_health_report
from mapf_cuda.observability.pipeline import (
    pipeline_snapshot,
    query_gpu_snapshot,
)


@dataclass(frozen=True)
class WorkerHealthHandle:
    """Pickle-safe worker-side view of shared health counters."""

    worker_id: int
    ready: Any
    published_count: Any
    active_call_id: Any
    active_call_started_s: Any
    active_call_timeout_s: Any
    call_sequence: Any
    complete: Any
    release_event: Any

    def mark_ready(self) -> None:
        self.ready.value = 1

    def begin_expert_call(
        self, *, call_id: int | None = None, timeout_s: float
    ) -> int:
        if call_id is None:
            with self.call_sequence.get_lock():
                self.call_sequence.value += 1
                call_id = int(self.call_sequence.value)
        self.active_call_id.value = int(call_id)
        self.active_call_timeout_s.value = float(timeout_s)
        self.active_call_started_s.value = float(time.monotonic())
        return int(call_id)

    def finish_expert_call(self) -> None:
        self.active_call_started_s.value = -1.0
        self.active_call_timeout_s.value = 0.0
        self.active_call_id.value = -1

    def mark_published(self) -> None:
        with self.published_count.get_lock():
            self.published_count.value += 1

    def mark_complete_and_wait(self) -> None:
        self.complete.value = 1
        self.release_event.wait()


def _new_worker_handle(worker_id: int, release_event: Any) -> WorkerHealthHandle:
    return WorkerHealthHandle(
        worker_id=int(worker_id),
        ready=mp.Value("B", 0),
        published_count=mp.Value("Q", 0),
        active_call_id=mp.Value("q", -1),
        active_call_started_s=mp.Value("d", -1.0),
        active_call_timeout_s=mp.Value("d", 0.0),
        call_sequence=mp.Value("Q", 0),
        complete=mp.Value("B", 0),
        release_event=release_event,
    )


class TrainingHealthLifecycle:
    """Own health sampling from pre-launch through captured worker exits."""

    def __init__(
        self,
        *,
        expected_workers: int,
        pipeline: Any,
        block_size: int,
        optimizer_steps_provider: Callable[[], int],
        samples_processed_provider: Callable[[], int],
        gpu_device_index: int = 0,
        gpu_provider: Callable[[], dict[str, Any]] | None = None,
        collector_factory: Callable[..., Any] = HealthCollector,
        expert_timeout_s: float | None,
        expert_grace_s: float = 5.0,
        sample_interval_s: float = 5.0,
    ) -> None:
        if expected_workers <= 0:
            raise ValueError("expected_workers must be positive for training health")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.expected_workers = int(expected_workers)
        self.pipeline = pipeline
        self.block_size = int(block_size)
        self.optimizer_steps_provider = optimizer_steps_provider
        self.samples_processed_provider = samples_processed_provider
        self.expert_timeout_s = expert_timeout_s
        self.expert_grace_s = float(expert_grace_s)
        self.sample_interval_s = float(sample_interval_s)
        self._worker_release_event = mp.Event()
        self.worker_handles = [
            _new_worker_handle(worker_id, self._worker_release_event)
            for worker_id in range(self.expected_workers)
        ]
        self._processes: list[Any] = []
        self._start_time_s = time.monotonic()
        self._initial_samples = int(samples_processed_provider())
        self._finalized_report: dict[str, Any] | None = None
        self._workers_released = False
        self._last_progress_sample_s = self._start_time_s
        if gpu_provider is None:
            gpu_provider = lambda: query_gpu_snapshot(
                device_index=int(gpu_device_index)
            )
        self.collector = collector_factory(
            process_provider=self._process_snapshot,
            gpu_provider=gpu_provider,
            pipeline_provider=self._pipeline_snapshot,
            sample_interval_s=self.sample_interval_s,
        )

    @property
    def running(self) -> bool:
        return bool(self.collector.running)

    def register_processes(self, processes: Sequence[Any]) -> None:
        if len(processes) != self.expected_workers:
            raise ValueError(
                "training health process count mismatch: "
                f"expected {self.expected_workers}, got {len(processes)}"
            )
        self._processes = list(processes)

    def _process_snapshot(self) -> dict[str, Any]:
        workers = []
        for worker_id, process in enumerate(self._processes):
            handle = self.worker_handles[worker_id]
            pid = getattr(process, "pid", None)
            if pid is None:
                continue
            workers.append(
                {
                    "worker_id": worker_id,
                    "pid": -1 if pid is None else int(pid),
                    "parent_pid": os.getpid(),
                    "ready": bool(handle.ready.value),
                    "alive": bool(process.is_alive()),
                    "exit_code": getattr(process, "exitcode", None),
                    "complete": bool(handle.complete.value),
                    "released": bool(self._workers_released),
                }
            )
        return {"parent_pid": os.getpid(), "workers": workers}

    def _active_expert_calls(self) -> list[dict[str, Any]]:
        calls = []
        for handle in self.worker_handles:
            started = float(handle.active_call_started_s.value)
            if started < 0.0:
                continue
            calls.append(
                {
                    "worker_id": int(handle.worker_id),
                    "call_id": int(handle.active_call_id.value),
                    "start_time_s": started,
                    "timeout_s": float(handle.active_call_timeout_s.value),
                }
            )
        return calls

    def _pipeline_snapshot(self) -> dict[str, Any]:
        snapshot = pipeline_snapshot(
            self.pipeline, fallback_block_size=self.block_size
        )
        block_size = max(1, int(snapshot["block_size"]))

        def align(value: int) -> int:
            return max(0, int(value)) // block_size * block_size

        reserve_ptr = align(snapshot["reserve_ptr"])
        dma_ptr = align(snapshot["dma_ptr"])
        compute_ptr = align(snapshot["compute_ptr"])
        capacity = int(snapshot["ring_capacity"])
        optimizer_steps = int(self.optimizer_steps_provider())
        samples_processed = int(self.samples_processed_provider())
        run_samples_processed = max(0, samples_processed - self._initial_samples)
        processes_alive = [
            process.is_alive() for process in self._processes
        ]
        ready_alive = any(
            bool(handle.ready.value) and alive and not bool(handle.complete.value)
            for handle, alive in zip(self.worker_handles, processes_alive)
        )
        active_calls = self._active_expert_calls()
        free_capacity = reserve_ptr - compute_ptr < capacity
        elapsed_s = max(time.monotonic() - self._start_time_s, 1e-9)
        sample_delta = max(0, samples_processed - self._initial_samples)
        return {
            "applicable": True,
            "producer_count": reserve_ptr,
            "dma_count": dma_ptr,
            "consumer_count": compute_ptr,
            "optimizer_steps": optimizer_steps,
            "producer_backlog": bool(ready_alive and free_capacity and not active_calls),
            "producer_capacity_available": bool(free_capacity),
            "dma_backlog": reserve_ptr > dma_ptr,
            "consumer_backlog": dma_ptr > compute_ptr,
            "optimizer_backlog": compute_ptr > run_samples_processed,
            "reserve_ptr": reserve_ptr,
            "dma_ptr": dma_ptr,
            "compute_ptr": compute_ptr,
            "ring_capacity": capacity,
            "block_size": block_size,
            "gpu_active": bool(
                dma_ptr > compute_ptr or compute_ptr > run_samples_processed
            ),
            "gpu_progress_counter": optimizer_steps,
            "active_expert_calls": active_calls,
            "producer_workers": [
                {
                    "worker_id": int(handle.worker_id),
                    "ready": bool(handle.ready.value),
                    "alive": bool(alive),
                    "published_count": int(handle.published_count.value),
                    "complete": bool(handle.complete.value),
                }
                for handle, alive in zip(self.worker_handles, processes_alive)
            ],
            "rolling_throughput": float(sample_delta / elapsed_s),
        }

    def start(self) -> None:
        self._start_time_s = time.monotonic()
        self._last_progress_sample_s = self._start_time_s
        self._initial_samples = int(self.samples_processed_provider())
        self.collector.start()

    def sample_once(self, *, reason: str) -> dict[str, Any]:
        return self.collector.sample_once(reason=reason)

    def maybe_sample_progress(self, *, force: bool = False) -> dict[str, Any] | None:
        now = time.monotonic()
        if not force and now - self._last_progress_sample_s < self.sample_interval_s:
            return None
        self._last_progress_sample_s = now
        return self.sample_once(reason="optimizer_progress")

    def all_workers_ready(self) -> bool:
        return bool(self._processes) and all(
            bool(handle.ready.value) for handle in self.worker_handles
        )

    def release_workers(self) -> None:
        if self._workers_released:
            return
        self._workers_released = True
        if self.running:
            self.sample_once(reason="workers_release")
        self._worker_release_event.set()

    def wait_for_workers_ready(
        self, *, timeout_s: float = 120.0, poll_interval_s: float = 0.01
    ) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while not self.all_workers_ready():
            failed = [
                process.exitcode
                for process in self._processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(
                    f"training worker failed before readiness: exitcodes={failed}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"training workers were not ready within {float(timeout_s):.3f}s"
                )
            time.sleep(float(poll_interval_s))
        self.sample_once(reason="workers_ready")

    def finalize(
        self, *, reference_throughput: float | None, reason: str = "final"
    ) -> dict[str, Any]:
        if self._finalized_report is not None:
            return self._finalized_report
        self.sample_once(reason=reason)
        self.collector.stop()
        self._finalized_report = build_live_health_report(
            self.collector.events,
            expected_workers=self.expected_workers,
            reference_throughput=reference_throughput,
            expert_timeout_s=self.expert_timeout_s,
            expert_grace_s=self.expert_grace_s,
            sample_interval_s=self.sample_interval_s,
        )
        return self._finalized_report

    def abort(
        self, *, reason: str, reference_throughput: float | None
    ) -> dict[str, Any]:
        return self.finalize(reference_throughput=reference_throughput, reason=reason)


__all__ = ["TrainingHealthLifecycle", "WorkerHealthHandle"]
