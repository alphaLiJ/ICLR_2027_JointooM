"""Runtime-health contracts for A0a benchmark collection and validation."""

from __future__ import annotations

import copy

import pytest

from expert.benchmark_health import (
    HealthCollector,
    build_live_health_report,
    validate_health_events,
)


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


def _workers(*, ready=True, exit_code=None, pid_offset=0):
    return {
        "parent_pid": 100,
        "workers": [
            {
                "worker_id": index,
                "pid": 200 + pid_offset + index,
                "parent_pid": 100,
                "ready": ready,
                "alive": exit_code is None,
                "exit_code": exit_code,
            }
            for index in range(2)
        ],
    }


def _gpu(progress=0):
    return {
        "device_index": 0,
        "utilization_pct": 75.0,
        "memory_used_bytes": 1024,
        "memory_total_bytes": 8192,
        "progress_counter": progress,
    }


def _pipeline(counter=0, *, throughput=100.0, backlog=True):
    return {
        "applicable": True,
        "producer_count": counter,
        "dma_count": counter,
        "consumer_count": counter,
        "optimizer_steps": counter,
        "producer_backlog": backlog,
        "dma_backlog": backlog,
        "consumer_backlog": backlog,
        "optimizer_backlog": backlog,
        "reserve_ptr": counter * 8,
        "dma_ptr": counter * 8,
        "compute_ptr": counter * 8,
        "ring_capacity": 64,
        "block_size": 8,
        "gpu_active": backlog,
        "gpu_progress_counter": counter,
        "active_expert_calls": [],
        "rolling_throughput": throughput,
    }


def _event(timestamp, *, workers=None, gpu=None, pipeline=None, errors=None):
    return {
        "timestamp_s": float(timestamp),
        "reason": "periodic",
        "process": workers if workers is not None else _workers(),
        "gpu": gpu if gpu is not None else _gpu(int(timestamp)),
        "pipeline": pipeline if pipeline is not None else _pipeline(int(timestamp)),
        "collector_errors": list(errors or []),
    }


def _codes(report):
    return {item["code"] for item in report["violations"]}


def test_collector_samples_all_providers_and_derives_worker_respawn():
    clock = FakeClock(10.0)
    process_states = iter([_workers(), _workers(pid_offset=1000)])
    collector = HealthCollector(
        process_provider=lambda: next(process_states),
        gpu_provider=lambda: _gpu(3),
        pipeline_provider=lambda: _pipeline(3),
        monotonic=clock,
        sleeper=lambda _seconds, _stop: False,
    )

    first = collector.sample_once(reason="ready")
    clock.advance(5)
    second = collector.sample_once(reason="progress")

    assert first["timestamp_s"] == 10.0
    assert first["reason"] == "ready"
    assert first["process"]["parent_pid"] == 100
    assert first["gpu"]["device_index"] == 0
    assert first["pipeline"]["reserve_ptr"] == 24
    assert first["collector_errors"] == []
    assert second["process"]["workers"][0]["respawned"] is True
    assert second["process"]["workers"][0]["respawn_count"] == 1
    assert collector.events == [first, second]


def test_collector_records_provider_errors_and_monitoring_can_continue():
    clock = FakeClock()
    attempts = 0

    def flaky_gpu():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary NVML failure")
        return _gpu(2)

    collector = HealthCollector(
        process_provider=_workers,
        gpu_provider=flaky_gpu,
        pipeline_provider=lambda: _pipeline(1),
        monotonic=clock,
        sleeper=lambda _seconds, _stop: False,
    )
    failed = collector.sample_once()
    clock.advance(5)
    recovered = collector.sample_once()

    assert failed["gpu"] is None
    assert failed["collector_errors"][0]["provider"] == "gpu"
    assert "temporary NVML failure" in failed["collector_errors"][0]["error"]
    assert recovered["gpu"]["progress_counter"] == 2
    assert recovered["collector_errors"] == []
    assert len(collector.events) == 2


def test_collector_start_and_stop_use_bounded_injected_periodic_wait():
    clock = FakeClock()
    waits = []

    def one_period_then_stop(seconds, stop_event):
        waits.append(seconds)
        clock.advance(seconds)
        stop_event.set()
        return True

    collector = HealthCollector(
        process_provider=_workers,
        gpu_provider=lambda: _gpu(0),
        pipeline_provider=lambda: _pipeline(0),
        monotonic=clock,
        sleeper=one_period_then_stop,
        sample_interval_s=5.0,
    )
    collector.start()
    collector.stop(timeout_s=0.5)

    assert waits == [5.0]
    assert len(collector.events) == 1
    assert collector.running is False
    assert collector.thread_error is None


def test_live_health_report_preserves_events_and_marks_live_sampling():
    events = [
        _event(0, workers={"parent_pid": 100, "workers": []}, pipeline={
            "applicable": False,
            "not_applicable_reason": "transition-only run has no ring or DMA pipeline",
        }),
        _event(1, workers={"parent_pid": 100, "workers": []}, pipeline={
            "applicable": False,
            "not_applicable_reason": "transition-only run has no ring or DMA pipeline",
        }),
    ]

    report = build_live_health_report(
        events,
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
    )

    assert report["status"] == "pass"
    assert report["collector"] == "benchmark_health_live"
    assert report["live_sampling"] is True
    assert report["paper_eligible"] is True
    assert report["events"] == events
    assert report["validation"]["valid"] is True


def test_validation_allows_small_periodic_sampler_jitter():
    events = [
        _event(0.0, workers=[], pipeline=_pipeline(0)),
        _event(5.85, workers=[], pipeline=_pipeline(1)),
    ]

    validation = validate_health_events(
        events,
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
        sample_interval_s=5.0,
    )

    assert not any(
        violation["code"] == "sample_gap"
        for violation in validation["violations"]
    )


def test_validation_rejects_gap_beyond_bounded_sampler_jitter():
    events = [
        _event(0.0, workers=[], pipeline=_pipeline(0)),
        _event(7.51, workers=[], pipeline=_pipeline(1)),
    ]

    validation = validate_health_events(
        events,
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
        sample_interval_s=5.0,
    )

    sample_gap = next(
        violation
        for violation in validation["violations"]
        if violation["code"] == "sample_gap"
    )
    assert sample_gap["details"]["allowed_gap_s"] == 7.5


def test_validator_rejects_worker_that_disappears_before_exit_is_captured():
    workers_alive = _workers(ready=True, exit_code=None)
    workers_dead = _workers(ready=True, exit_code=0)
    workers_dead["workers"][0]["alive"] = False
    events = [
        _event(0, workers=workers_alive, pipeline=_pipeline(0, backlog=False)),
        _event(5, workers=workers_dead, pipeline=_pipeline(1, backlog=False)),
        _event(10, workers=_workers(ready=True, exit_code=0), pipeline=_pipeline(2, backlog=False)),
    ]

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=30.0,
        expert_grace_s=0.0,
    )

    assert "worker_unexpected_death" in _codes(validation)


def test_training_worker_exit_requires_complete_and_explicit_parent_release():
    alive = _workers(ready=True)
    for worker in alive["workers"]:
        worker.update(complete=True, released=False)
    released = copy.deepcopy(alive)
    for worker in released["workers"]:
        worker.update(alive=False, exit_code=0, released=True)
    final = copy.deepcopy(released)
    events = [
        _event(0, workers=alive, pipeline=_pipeline(0, backlog=False)),
        _event(5, workers=released, pipeline=_pipeline(1, backlog=False)),
        _event(10, workers=final, pipeline=_pipeline(2, backlog=False)),
    ]

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=30.0,
        expert_grace_s=0.0,
    )

    assert validation["valid"] is True


def test_training_worker_clean_exit_without_complete_is_rejected_at_final():
    workers = _workers(ready=True, exit_code=0)
    for worker in workers["workers"]:
        worker.update(complete=False, released=True)

    validation = validate_health_events(
        [_event(0, workers=workers, pipeline=_pipeline(0, backlog=False))],
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=30.0,
        expert_grace_s=0.0,
    )

    assert {"worker_unexpected_death", "worker_incomplete"} <= _codes(validation)


def test_validator_rejects_temporal_ring_pointer_regression():
    first = _pipeline(2, backlog=False)
    second = _pipeline(1, backlog=False)
    events = [
        _event(0, pipeline=first),
        _event(5, pipeline=second),
        _event(10, workers=_workers(exit_code=0), pipeline=second),
    ]

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=30.0,
        expert_grace_s=0.0,
    )

    assert "ring_pointer_regression" in _codes(validation)


def test_validator_accepts_zero_utilization_samples_when_gpu_progresses():
    events = []
    for timestamp in range(0, 31, 5):
        gpu = _gpu(timestamp)
        gpu["utilization_pct"] = 0.0
        workers = _workers(exit_code=0) if timestamp == 30 else _workers()
        events.append(
            _event(
                timestamp,
                workers=workers,
                gpu=gpu,
                pipeline=_pipeline(timestamp, backlog=True),
            )
        )

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=60.0,
        expert_grace_s=0.0,
    )

    assert "gpu_utilization_zero" not in _codes(validation)
    assert "gpu_stall" not in _codes(validation)


def test_validator_rejects_zero_utilization_when_gpu_does_not_progress():
    events = []
    for timestamp in range(0, 31, 5):
        gpu = _gpu(0)
        gpu["utilization_pct"] = 0.0
        pipeline = _pipeline(0, backlog=True)
        workers = _workers(exit_code=0) if timestamp == 30 else _workers()
        events.append(
            _event(
                timestamp,
                workers=workers,
                gpu=gpu,
                pipeline=pipeline,
            )
        )

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=60.0,
        expert_grace_s=0.0,
    )

    assert "gpu_utilization_zero" in _codes(validation)


def test_validator_detects_one_stalled_producer_hidden_by_aggregate_progress():
    events = []
    for timestamp, counts in ((0, (0, 0)), (30, (1, 0)), (65, (2, 0))):
        pipeline = _pipeline(timestamp, backlog=False)
        pipeline["producer_capacity_available"] = True
        pipeline["producer_workers"] = [
            {
                "worker_id": worker_id,
                "ready": True,
                "alive": timestamp < 65,
                "complete": timestamp == 65,
                "published_count": counts[worker_id],
            }
            for worker_id in range(2)
        ]
        if timestamp == 65:
            # Worker 0 completed normally; worker 1 is still live and has never
            # published, while worker 0's progress keeps the aggregate moving.
            pipeline["producer_workers"][0].update(alive=False, complete=True)
            pipeline["producer_workers"][1].update(alive=True, complete=False)
        workers = _workers(exit_code=0) if timestamp == 65 else _workers()
        if timestamp == 65:
            workers["workers"][1].update(alive=True, exit_code=None)
        events.append(
            _event(timestamp, workers=workers, pipeline=pipeline)
        )

    validation = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=None,
        expert_timeout_s=60.0,
        expert_grace_s=0.0,
        sample_interval_s=40.0,
    )

    assert "producer_worker_stall" in _codes(validation)


def test_producer_stall_timer_starts_when_capacity_becomes_available():
    events = []
    for timestamp, capacity_available in ((0, False), (120, False), (125, True)):
        pipeline = _pipeline(0, backlog=False)
        pipeline["producer_capacity_available"] = capacity_available
        pipeline["producer_workers"] = [
            {
                "worker_id": 0,
                "ready": True,
                "alive": True,
                "complete": False,
                "published_count": 0,
            }
        ]
        workers = {"parent_pid": 100, "workers": [_workers()["workers"][0]]}
        events.append(_event(timestamp, workers=workers, pipeline=pipeline))

    validation = validate_health_events(
        events,
        expected_workers=1,
        reference_throughput=None,
        expert_timeout_s=30.0,
        expert_grace_s=0.0,
        sample_interval_s=121.0,
    )

    assert "producer_worker_stall" not in _codes(validation)


def test_validator_accepts_healthy_pipeline_without_mutating_events():
    events = [
        _event(0, pipeline=_pipeline(0, backlog=False)),
        _event(5, pipeline=_pipeline(1)),
        _event(10, workers=_workers(exit_code=0), pipeline=_pipeline(2)),
    ]
    original = copy.deepcopy(events)
    report = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
    )

    assert report["valid"] is True
    assert report["violations"] == []
    assert report["sample_count"] == 3
    assert events == original


def test_validator_requires_final_zero_exit_codes_and_stable_worker_roster():
    events = [
        _event(0),
        _event(
            5,
            workers={"parent_pid": 100, "workers": _workers()["workers"][:1]},
        ),
    ]
    report = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
    )
    assert {"worker_count", "worker_exit_missing"} <= _codes(report)


def test_validator_requires_gpu0_metrics_progress_and_throughput_samples():
    gpu = _gpu(0)
    gpu["device_index"] = 1
    gpu.pop("memory_total_bytes")
    gpu.pop("progress_counter")
    pipeline = _pipeline(0)
    pipeline.pop("gpu_progress_counter")
    pipeline.pop("rolling_throughput")
    report = validate_health_events(
        [_event(0, workers=_workers(exit_code=0), gpu=gpu, pipeline=pipeline)],
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
    )
    assert {
        "gpu_device",
        "gpu_metrics_missing",
        "gpu_progress_missing",
        "throughput_missing",
    } <= _codes(report)


def test_validator_checks_readiness_cadence_exit_respawn_and_collector_errors():
    events = [
        _event(0, workers=_workers(ready=False)),
        _event(121, workers=_workers(ready=False), errors=[{"provider": "gpu", "error": "lost"}]),
        _event(126, workers=_workers(exit_code=7, pid_offset=1000)),
    ]
    report = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
    )

    assert {
        "readiness_timeout",
        "sample_gap",
        "worker_nonzero_exit",
        "worker_respawn",
        "collector_error",
    } <= _codes(report)


@pytest.mark.parametrize(
    "layer,counter_key,backlog_key,expected_code",
    [
        ("producer", "producer_count", "producer_backlog", "producer_stall"),
        ("dma", "dma_count", "dma_backlog", "dma_stall"),
        ("consumer", "consumer_count", "consumer_backlog", "consumer_stall"),
        ("optimizer", "optimizer_steps", "optimizer_backlog", "optimizer_stall"),
    ],
)
def test_validator_requires_each_backlogged_pipeline_layer_to_progress(
    layer, counter_key, backlog_key, expected_code
):
    del layer
    first = _pipeline(1, backlog=False)
    second = _pipeline(2, backlog=False)
    third = _pipeline(3, backlog=False)
    for item in (first, second, third):
        item[backlog_key] = True
        item[counter_key] = 9
    events = [_event(0, pipeline=first), _event(30, pipeline=second), _event(61, pipeline=third)]

    report = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
        sample_interval_s=31.0,
    )
    assert expected_code in _codes(report)


def test_validator_checks_ring_alignment_order_and_capacity_bounds():
    misaligned = _pipeline(1)
    misaligned["reserve_ptr"] = 10
    out_of_bounds = _pipeline(2)
    out_of_bounds.update(reserve_ptr=96, dma_ptr=64, compute_ptr=0, ring_capacity=64)
    report = validate_health_events(
        [_event(0, pipeline=misaligned), _event(5, pipeline=out_of_bounds)],
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
    )
    assert {"ring_alignment", "ring_bounds"} <= _codes(report)


def test_validator_detects_gpu_stall_and_three_consecutive_low_windows():
    events = []
    for timestamp, throughput in [(0, 80.0), (15, 49.0), (30, 40.0), (46, 30.0)]:
        pipeline = _pipeline(1, throughput=throughput)
        pipeline["gpu_progress_counter"] = 7
        events.append(_event(timestamp, pipeline=pipeline, gpu=_gpu(7)))
    report = validate_health_events(
        events,
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
        sample_interval_s=16.0,
    )
    assert {"gpu_stall", "throughput_collapse"} <= _codes(report)


def test_validator_detects_expert_timeout_plus_grace():
    pipeline = _pipeline(1)
    pipeline["active_expert_calls"] = [{"call_id": "expert-4", "start_time_s": 10.0}]
    report = validate_health_events(
        [_event(0), _event(46, pipeline=pipeline)],
        expected_workers=2,
        reference_throughput=100.0,
        expert_timeout_s=30.0,
        expert_grace_s=5.0,
        sample_interval_s=50.0,
    )
    assert "expert_timeout" in _codes(report)


def test_validator_accepts_transition_pipeline_with_explicit_na_reason():
    pipeline = {
        "applicable": False,
        "not_applicable_reason": "transition-only run has no ring or DMA pipeline",
    }
    report = validate_health_events(
        [_event(0, workers={"parent_pid": 100, "workers": []}, pipeline=pipeline)],
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
    )
    assert report["valid"] is True
    assert report["pipeline_not_applicable_reason"] == pipeline["not_applicable_reason"]


def test_validator_rejects_missing_na_reason_and_invalid_configuration():
    bad_pipeline = {"applicable": False, "not_applicable_reason": ""}
    report = validate_health_events(
        [_event(0, workers={"parent_pid": 100, "workers": []}, pipeline=bad_pipeline)],
        expected_workers=0,
        reference_throughput=None,
        expert_timeout_s=None,
        expert_grace_s=0.0,
    )
    assert "pipeline_na_reason_missing" in _codes(report)
    with pytest.raises(ValueError, match="events"):
        validate_health_events([], expected_workers=0, reference_throughput=None)
    with pytest.raises(ValueError, match="expected_workers"):
        validate_health_events([_event(0)], expected_workers=-1, reference_throughput=None)
