"""In-memory runtime-health collection and validation for benchmark runs.

The collector deliberately knows nothing about archive publication.  Callers
provide lightweight process, GPU, and pipeline snapshots, while the validator
turns the resulting event stream into an auditable pass/fail report.
"""

from __future__ import annotations

import copy
import numbers
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


Provider = Callable[[], Mapping[str, Any]]
Sleeper = Callable[[float, threading.Event], bool]


def _default_sleeper(seconds: float, stop_event: threading.Event) -> bool:
    return stop_event.wait(seconds)


def _positive_finite(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a positive finite number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return normalized


def _nonnegative_finite(value: Any, *, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{label} must be a non-negative finite number")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return normalized


class HealthCollector:
    """Periodically collect provider snapshots without publishing side effects.

    ``sleeper`` receives ``(interval_seconds, stop_event)`` and should return
    truthy when the wait ended because stopping was requested.  The injected
    form makes periodic behavior testable without wall-clock sleeps.
    """

    def __init__(
        self,
        *,
        process_provider: Provider,
        gpu_provider: Provider,
        pipeline_provider: Provider,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Sleeper = _default_sleeper,
        sample_interval_s: float = 5.0,
    ) -> None:
        for label, provider in (
            ("process_provider", process_provider),
            ("gpu_provider", gpu_provider),
            ("pipeline_provider", pipeline_provider),
            ("monotonic", monotonic),
            ("sleeper", sleeper),
        ):
            if not callable(provider):
                raise ValueError(f"{label} must be callable")
        self.sample_interval_s = _positive_finite(
            sample_interval_s, label="sample_interval_s"
        )
        self._providers = {
            "process": process_provider,
            "gpu": gpu_provider,
            "pipeline": pipeline_provider,
        }
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._events: list[dict[str, Any]] = []
        self._events_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._worker_pids: dict[Any, int] = {}
        self._worker_respawns: dict[Any, int] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_error: str | None = None

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._events_lock:
            return copy.deepcopy(self._events)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def thread_error(self) -> str | None:
        return self._thread_error

    def _normalize_process_snapshot(
        self, snapshot: Mapping[str, Any]
    ) -> dict[str, Any]:
        normalized = copy.deepcopy(dict(snapshot))
        workers_raw = normalized.get("workers", [])
        if isinstance(workers_raw, (str, bytes)) or not isinstance(
            workers_raw, Sequence
        ):
            raise ValueError("process provider workers must be a sequence")
        workers = []
        seen_ids = set()
        for raw in workers_raw:
            if not isinstance(raw, Mapping):
                raise ValueError("process provider workers must contain mappings")
            worker = copy.deepcopy(dict(raw))
            if "worker_id" not in worker or "pid" not in worker:
                raise ValueError("worker snapshot requires worker_id and pid")
            worker_id = worker["worker_id"]
            try:
                duplicate = worker_id in seen_ids
            except TypeError as exc:
                raise ValueError("worker_id must be hashable") from exc
            if duplicate:
                raise ValueError(f"duplicate worker_id {worker_id!r}")
            seen_ids.add(worker_id)
            pid = int(worker["pid"])
            previous = self._worker_pids.get(worker_id)
            respawned = previous is not None and previous != pid
            if respawned:
                self._worker_respawns[worker_id] = (
                    self._worker_respawns.get(worker_id, 0) + 1
                )
            self._worker_pids[worker_id] = pid
            worker["pid"] = pid
            worker["respawned"] = respawned
            worker["respawn_count"] = self._worker_respawns.get(worker_id, 0)
            workers.append(worker)
        normalized["workers"] = workers
        return normalized

    def sample_once(self, *, reason: str = "periodic") -> dict[str, Any]:
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a non-empty string")
        with self._sample_lock:
            timestamp = float(self._monotonic())
            if not np.isfinite(timestamp):
                raise RuntimeError("monotonic provider returned a non-finite timestamp")
            event: dict[str, Any] = {
                "timestamp_s": timestamp,
                "reason": reason,
                "process": None,
                "gpu": None,
                "pipeline": None,
                "collector_errors": [],
            }
            for name, provider in self._providers.items():
                try:
                    snapshot = provider()
                    if not isinstance(snapshot, Mapping):
                        raise ValueError(f"{name} provider must return a mapping")
                    if name == "process":
                        snapshot = self._normalize_process_snapshot(snapshot)
                    else:
                        snapshot = copy.deepcopy(dict(snapshot))
                    event[name] = snapshot
                except BaseException as exc:  # monitoring must survive provider failures
                    event["collector_errors"].append(
                        {
                            "provider": name,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            with self._events_lock:
                self._events.append(event)
            return copy.deepcopy(event)

    def _record_thread_error(self, exc: BaseException) -> None:
        self._thread_error = f"{type(exc).__name__}: {exc}"
        event = {
            "timestamp_s": float(self._monotonic()),
            "reason": "collector_error",
            "process": None,
            "gpu": None,
            "pipeline": None,
            "collector_errors": [
                {
                    "provider": "collector_thread",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ],
        }
        with self._events_lock:
            self._events.append(event)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._sleeper(self.sample_interval_s, self._stop_event):
                    break
                self.sample_once(reason="periodic")
        except BaseException as exc:
            self._record_thread_error(exc)
            self._stop_event.set()

    def start(self) -> None:
        if self.running:
            raise RuntimeError("health collector is already running")
        self._stop_event.clear()
        self._thread_error = None
        self.sample_once(reason="start")
        self._thread = threading.Thread(
            target=self._run,
            name="benchmark-health-collector",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout_s: float = 5.0) -> None:
        timeout = _positive_finite(timeout_s, label="timeout_s")
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise RuntimeError(
                f"health collector did not stop within {timeout:.3f} seconds"
            )


def _add_violation(
    violations: list[dict[str, Any]], code: str, message: str, **details: Any
) -> None:
    if any(item["code"] == code and item.get("details") == details for item in violations):
        return
    item: dict[str, Any] = {"code": code, "message": message}
    if details:
        item["details"] = details
    violations.append(item)


def _event_workers(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    process = event.get("process")
    if not isinstance(process, Mapping):
        return []
    workers = process.get("workers", [])
    if isinstance(workers, Sequence) and not isinstance(workers, (str, bytes)):
        return [item for item in workers if isinstance(item, Mapping)]
    return []


def validate_health_events(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_workers: int,
    reference_throughput: float | None,
    readiness_timeout_s: float = 120.0,
    sample_interval_s: float = 5.0,
    progress_timeout_s: float = 60.0,
    gpu_progress_window_s: float = 30.0,
    expert_timeout_s: float | None = None,
    expert_grace_s: float = 0.0,
) -> dict[str, Any]:
    """Validate a collected health stream without changing or dropping events."""

    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise ValueError("events must be a non-empty sequence")
    if (
        isinstance(expected_workers, (bool, np.bool_))
        or not isinstance(expected_workers, numbers.Integral)
        or expected_workers < 0
    ):
        raise ValueError("expected_workers must be a non-negative integer")
    readiness_timeout = _positive_finite(
        readiness_timeout_s, label="readiness_timeout_s"
    )
    sample_interval = _positive_finite(
        sample_interval_s, label="sample_interval_s"
    )
    progress_timeout = _positive_finite(
        progress_timeout_s, label="progress_timeout_s"
    )
    gpu_window = _positive_finite(
        gpu_progress_window_s, label="gpu_progress_window_s"
    )
    grace = _nonnegative_finite(expert_grace_s, label="expert_grace_s")
    timeout = (
        None
        if expert_timeout_s is None
        else _positive_finite(expert_timeout_s, label="expert_timeout_s")
    )
    reference = (
        None
        if reference_throughput is None
        else _positive_finite(reference_throughput, label="reference_throughput")
    )
    materialized = copy.deepcopy([dict(event) for event in events])
    if any(not isinstance(event, Mapping) for event in events):
        raise ValueError("events must contain only mappings")
    timestamps = []
    for index, event in enumerate(materialized):
        value = event.get("timestamp_s")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"event {index} timestamp_s must be finite")
        timestamp = float(value)
        if not np.isfinite(timestamp):
            raise ValueError(f"event {index} timestamp_s must be finite")
        timestamps.append(timestamp)

    violations: list[dict[str, Any]] = []
    start = timestamps[0]
    for previous, current in zip(timestamps, timestamps[1:]):
        if current < previous:
            _add_violation(
                violations, "timestamp_order", "health timestamps moved backwards"
            )
        # A periodic monitoring thread cannot wake at an exact nanosecond, and
        # each wake performs a synchronous nvidia-smi query.  Allow bounded
        # scheduler/provider jitter while still rejecting a genuinely missed
        # sample.
        allowed_gap_s = sample_interval * 1.5
        if current - previous > allowed_gap_s:
            _add_violation(
                violations,
                "sample_gap",
                "health sampling gap exceeded the configured interval",
                observed_gap_s=current - previous,
                allowed_gap_s=allowed_gap_s,
            )

    for event in materialized:
        for error in event.get("collector_errors", []) or []:
            _add_violation(
                violations,
                "collector_error",
                "health collector recorded a provider or thread error",
                provider=(error.get("provider") if isinstance(error, Mapping) else None),
            )

    max_workers = 0
    ready_at = None if expected_workers else start
    ready_worker_ids: set[Any] = set()
    known_pids: dict[Any, int] = {}
    for event_index, (timestamp, event) in enumerate(zip(timestamps, materialized)):
        workers = _event_workers(event)
        max_workers = max(max_workers, len(workers))
        process = event.get("process")
        parent_pid = process.get("parent_pid") if isinstance(process, Mapping) else None
        ids = []
        for worker in workers:
            worker_id = worker.get("worker_id")
            ids.append(worker_id)
            pid = worker.get("pid")
            if worker.get("parent_pid") != parent_pid:
                _add_violation(
                    violations, "pid_tree", "worker parent PID does not match collector parent"
                )
            if worker.get("exit_code") not in (None, 0):
                _add_violation(
                    violations,
                    "worker_nonzero_exit",
                    "worker exited with a nonzero code",
                    worker_id=worker_id,
                    exit_code=worker.get("exit_code"),
                )
            lifecycle_handshake = (
                "complete" in worker or "released" in worker
            )
            clean_released_exit = (
                lifecycle_handshake
                and bool(worker.get("complete"))
                and bool(worker.get("released"))
                and worker.get("exit_code") == 0
            )
            if (
                bool(worker.get("ready"))
                and not bool(worker.get("alive"))
                and (
                    (lifecycle_handshake and not clean_released_exit)
                    or (
                        not lifecycle_handshake
                        and event_index < len(materialized) - 1
                    )
                )
            ):
                _add_violation(
                    violations,
                    "worker_unexpected_death",
                    "a ready worker exited outside the complete-and-released lifecycle",
                    worker_id=worker_id,
                )
            previous_pid = known_pids.get(worker_id)
            if previous_pid is not None and previous_pid != pid:
                _add_violation(
                    violations,
                    "worker_respawn",
                    "worker PID changed during the run",
                    worker_id=worker_id,
                )
            known_pids[worker_id] = pid
            if worker.get("respawned") or int(worker.get("respawn_count", 0) or 0) > 0:
                _add_violation(
                    violations,
                    "worker_respawn",
                    "worker provider reported a respawn",
                    worker_id=worker_id,
                )
        if len(set(ids)) != len(ids):
            _add_violation(
                violations, "worker_count", "worker IDs are duplicated in one sample"
            )
        if (
            len(workers) == expected_workers
            and all(bool(worker.get("ready")) for worker in workers)
            and ready_at is None
        ):
            ready_at = timestamp
            ready_worker_ids = set(ids)
        elif ready_at is not None and expected_workers and set(ids) != ready_worker_ids:
            _add_violation(
                violations,
                "worker_count",
                "worker roster changed after readiness",
                expected=expected_workers,
                observed=len(workers),
            )
    if max_workers != expected_workers:
        _add_violation(
            violations,
            "worker_count",
            "observed worker count differs from expected fan-out",
            expected=expected_workers,
            observed=max_workers,
        )
    if ready_at is None or ready_at - start > readiness_timeout:
        _add_violation(
            violations,
            "readiness_timeout",
            "workers did not become ready before the readiness deadline",
            timeout_s=readiness_timeout,
        )
    final_workers = _event_workers(materialized[-1])
    if expected_workers and (
        len(final_workers) != expected_workers
        or any(worker.get("exit_code") is None for worker in final_workers)
    ):
        _add_violation(
            violations,
            "worker_exit_missing",
            "final health sample did not capture every worker exit code",
            expected=expected_workers,
            observed=len(final_workers),
        )
    if any(
        "complete" in worker or "released" in worker
        for event in materialized
        for worker in _event_workers(event)
    ) and any(not bool(worker.get("complete")) for worker in final_workers):
        _add_violation(
            violations,
            "worker_incomplete",
            "final training health sample contains an incomplete producer",
        )

    applicable_values = []
    na_reasons = []
    layer_state: dict[str, dict[str, Any]] = {}
    producer_worker_state: dict[Any, dict[str, Any]] = {}
    gpu_state: dict[str, Any] = {}
    gpu_utilization_window: dict[str, Any] = {}
    previous_ring_pointers: dict[str, int] | None = None
    consecutive_low = 0
    layer_fields = {
        "producer": ("producer_count", "producer_backlog"),
        "dma": ("dma_count", "dma_backlog"),
        "consumer": ("consumer_count", "consumer_backlog"),
        "optimizer": ("optimizer_steps", "optimizer_backlog"),
    }
    for timestamp, event in zip(timestamps, materialized):
        gpu = event.get("gpu")
        if not isinstance(gpu, Mapping):
            _add_violation(
                violations, "gpu_metrics_missing", "GPU 0 health snapshot is missing"
            )
        else:
            if gpu.get("device_index") != 0:
                _add_violation(
                    violations,
                    "gpu_device",
                    "health snapshot must describe physical GPU 0",
                    observed=gpu.get("device_index"),
                )
            try:
                utilization = float(gpu["utilization_pct"])
                used = float(gpu["memory_used_bytes"])
                total = float(gpu["memory_total_bytes"])
                if (
                    not all(np.isfinite(item) for item in (utilization, used, total))
                    or not 0.0 <= utilization <= 100.0
                    or used < 0.0
                    or total <= 0.0
                    or used > total
                ):
                    raise ValueError("GPU metrics outside valid bounds")
            except (KeyError, TypeError, ValueError):
                _add_violation(
                    violations,
                    "gpu_metrics_missing",
                    "GPU utilization or memory metadata is missing or invalid",
                )
        pipeline = event.get("pipeline")
        if not isinstance(pipeline, Mapping):
            continue
        applicable = bool(pipeline.get("applicable", True))
        applicable_values.append(applicable)
        if not applicable:
            reason = pipeline.get("not_applicable_reason")
            if isinstance(reason, str) and reason.strip():
                na_reasons.append(reason)
            else:
                _add_violation(
                    violations,
                    "pipeline_na_reason_missing",
                    "non-applicable pipeline requires an explicit reason",
                )
            continue

        active_calls_value = pipeline.get("active_expert_calls", [])
        active_call_workers = {
            call.get("worker_id")
            for call in active_calls_value
            if isinstance(call, Mapping) and call.get("worker_id") is not None
        } if isinstance(active_calls_value, Sequence) and not isinstance(
            active_calls_value, (str, bytes)
        ) else set()
        producer_workers = pipeline.get("producer_workers", [])
        capacity_available = bool(
            pipeline.get("producer_capacity_available", False)
        )
        if isinstance(producer_workers, Sequence) and not isinstance(
            producer_workers, (str, bytes)
        ):
            for producer in producer_workers:
                if not isinstance(producer, Mapping):
                    continue
                worker_id = producer.get("worker_id")
                published = producer.get("published_count")
                if (
                    worker_id is None
                    or isinstance(published, bool)
                    or not isinstance(published, numbers.Real)
                ):
                    _add_violation(
                        violations,
                        "producer_worker_invalid",
                        "producer worker progress metadata is missing or invalid",
                    )
                    continue
                state = producer_worker_state.setdefault(
                    worker_id,
                    {
                        "published_count": published,
                        "last_activity": timestamp,
                        "monitored": False,
                    },
                )
                if published != state["published_count"]:
                    state["published_count"] = published
                    state["last_activity"] = timestamp
                needs_progress = (
                    capacity_available
                    and bool(producer.get("ready"))
                    and bool(producer.get("alive"))
                    and not bool(producer.get("complete"))
                )
                monitored = needs_progress and worker_id not in active_call_workers
                if monitored and not state["monitored"]:
                    state["last_activity"] = timestamp
                state["monitored"] = monitored
                if (
                    monitored
                    and timestamp - state["last_activity"] > progress_timeout
                ):
                    _add_violation(
                        violations,
                        "producer_worker_stall",
                        "one live producer made no progress while ring capacity was free",
                        worker_id=worker_id,
                        timeout_s=progress_timeout,
                    )

        for layer, (counter_key, backlog_key) in layer_fields.items():
            counter = pipeline.get(counter_key)
            backlog = bool(pipeline.get(backlog_key, False))
            if not isinstance(counter, numbers.Real) or isinstance(counter, bool):
                _add_violation(
                    violations,
                    f"{layer}_counter_invalid",
                    f"{layer} progress counter is missing or invalid",
                )
                continue
            state = layer_state.setdefault(
                layer, {"counter": counter, "last_progress": timestamp, "backlog": False}
            )
            if counter != state["counter"]:
                state["counter"] = counter
                state["last_progress"] = timestamp
            if backlog and not state["backlog"]:
                state["last_progress"] = timestamp
            state["backlog"] = backlog
            if backlog and timestamp - state["last_progress"] > progress_timeout:
                _add_violation(
                    violations,
                    f"{layer}_stall",
                    f"{layer} made no progress while backlog was present",
                    timeout_s=progress_timeout,
                )

        pointer_names = ("reserve_ptr", "dma_ptr", "compute_ptr")
        try:
            reserve, dma, compute = (int(pipeline[name]) for name in pointer_names)
            capacity = int(pipeline["ring_capacity"])
            block = int(pipeline["block_size"])
            if block <= 0 or capacity <= 0 or capacity % block != 0 or any(
                pointer % block != 0 for pointer in (reserve, dma, compute)
            ):
                _add_violation(
                    violations,
                    "ring_alignment",
                    "ring capacity and logical pointers must be block aligned",
                )
            if (
                min(reserve, dma, compute) < 0
                or not compute <= dma <= reserve
                or reserve - compute > capacity
            ):
                _add_violation(
                    violations,
                    "ring_bounds",
                    "ring logical pointers are out of order or exceed capacity",
                )
            current_ring_pointers = {
                "reserve_ptr": reserve,
                "dma_ptr": dma,
                "compute_ptr": compute,
            }
            if previous_ring_pointers is not None:
                regressed = [
                    name
                    for name, value in current_ring_pointers.items()
                    if value < previous_ring_pointers[name]
                ]
                if regressed:
                    _add_violation(
                        violations,
                        "ring_pointer_regression",
                        "ring logical pointers must be monotonic over time",
                        pointers=regressed,
                    )
            previous_ring_pointers = current_ring_pointers
        except (KeyError, TypeError, ValueError):
            _add_violation(
                violations,
                "ring_bounds",
                "ring pointer metadata is missing or invalid",
            )

        gpu_active = bool(pipeline.get("gpu_active", False))
        progress_value = pipeline.get("gpu_progress_counter")
        if progress_value is None and isinstance(event.get("gpu"), Mapping):
            progress_value = event["gpu"].get("progress_counter")
        if gpu_active:
            utilization = None
            if isinstance(gpu, Mapping):
                try:
                    utilization = float(gpu.get("utilization_pct"))
                except (TypeError, ValueError):
                    utilization = None
            if not gpu_utilization_window:
                gpu_utilization_window = {
                    "start": timestamp,
                    "saw_nonzero": bool(utilization is not None and utilization > 0.0),
                    "start_progress": progress_value,
                    "saw_progress": False,
                }
            else:
                if utilization is not None and utilization > 0.0:
                    gpu_utilization_window["saw_nonzero"] = True
                if progress_value != gpu_utilization_window["start_progress"]:
                    gpu_utilization_window["saw_progress"] = True
                if timestamp - gpu_utilization_window["start"] >= gpu_window:
                    # nvidia-smi reports an instantaneous utilization sample.
                    # A zero sample is not evidence of a stalled training run
                    # when the optimizer/GPU progress counter advanced during
                    # the same window.
                    if (
                        not gpu_utilization_window["saw_nonzero"]
                        and not gpu_utilization_window["saw_progress"]
                    ):
                        _add_violation(
                            violations,
                            "gpu_utilization_zero",
                            "an active GPU window contained neither nonzero utilization nor progress",
                            window_s=gpu_window,
                        )
                    gpu_utilization_window = {
                        "start": timestamp,
                        "saw_nonzero": bool(
                            utilization is not None and utilization > 0.0
                        ),
                        "start_progress": progress_value,
                        "saw_progress": False,
                    }
            if progress_value is None:
                _add_violation(
                    violations,
                    "gpu_progress_missing",
                    "active GPU sample is missing a progress counter",
                )
            if not gpu_state:
                gpu_state = {"counter": progress_value, "last_progress": timestamp}
            elif progress_value != gpu_state["counter"]:
                gpu_state = {"counter": progress_value, "last_progress": timestamp}
            elif timestamp - gpu_state["last_progress"] > gpu_window:
                _add_violation(
                    violations,
                    "gpu_stall",
                    "GPU made no progress during an active window",
                    window_s=gpu_window,
                )
        else:
            gpu_state = {}
            gpu_utilization_window = {}

        if reference is not None:
            throughput = pipeline.get("rolling_throughput")
            if isinstance(throughput, numbers.Real) and np.isfinite(float(throughput)):
                consecutive_low = (
                    consecutive_low + 1
                    if float(throughput) < 0.5 * reference
                    else 0
                )
                if consecutive_low >= 3:
                    _add_violation(
                        violations,
                        "throughput_collapse",
                        "three consecutive throughput windows fell below half reference",
                        reference=reference,
                    )
            else:
                _add_violation(
                    violations,
                    "throughput_missing",
                    "rolling throughput is missing or non-finite",
                )

        active_calls = active_calls_value
        if timeout is not None and isinstance(active_calls, Sequence):
            for call in active_calls:
                if not isinstance(call, Mapping):
                    continue
                started = call.get("start_time_s")
                call_timeout = call.get("timeout_s", timeout)
                if (
                    isinstance(started, numbers.Real)
                    and isinstance(call_timeout, numbers.Real)
                    and timestamp - float(started) > float(call_timeout) + grace
                ):
                    _add_violation(
                        violations,
                        "expert_timeout",
                        "expert call exceeded frozen timeout plus grace",
                        call_id=call.get("call_id"),
                    )

    if applicable_values and any(applicable_values) and not all(applicable_values):
        _add_violation(
            violations,
            "pipeline_applicability_mixed",
            "pipeline applicability changed during one run",
        )

    return {
        "valid": not violations,
        "sample_count": len(materialized),
        "start_timestamp_s": timestamps[0],
        "end_timestamp_s": timestamps[-1],
        "expected_workers": int(expected_workers),
        "ready_timestamp_s": ready_at,
        "pipeline_not_applicable_reason": (
            na_reasons[0] if na_reasons and not any(applicable_values) else None
        ),
        "violations": violations,
    }


def build_live_health_report(
    events: Sequence[Mapping[str, Any]],
    *,
    expected_workers: int,
    reference_throughput: float | None,
    expert_timeout_s: float | None,
    expert_grace_s: float,
    **validation_options: Any,
) -> dict[str, Any]:
    """Validate and package events produced during the measured run."""

    materialized = copy.deepcopy([dict(event) for event in events])
    validation = validate_health_events(
        materialized,
        expected_workers=expected_workers,
        reference_throughput=reference_throughput,
        expert_timeout_s=expert_timeout_s,
        expert_grace_s=expert_grace_s,
        **validation_options,
    )
    return {
        "schema_version": 1,
        "status": "pass" if validation["valid"] else "fail",
        "collector": "benchmark_health_live",
        "live_sampling": True,
        "paper_eligible": bool(validation["valid"]),
        "expected_workers": int(expected_workers),
        "sample_count": len(materialized),
        "events": materialized,
        "validation": validation,
    }


__all__ = ["HealthCollector", "build_live_health_report", "validate_health_events"]
