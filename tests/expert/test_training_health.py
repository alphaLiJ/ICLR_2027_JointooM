import types

from expert.benchmark_training_health import TrainingHealthLifecycle


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.exitcode = None
        self._alive = True

    def is_alive(self):
        return self._alive


class _FakePipeline:
    def __init__(self):
        self.capacity = 16
        self.compute_ptr = 0
        self.ring_buffer = types.SimpleNamespace(
            shared_reserve_ptr=types.SimpleNamespace(value=0),
            stage_rows=4,
        )
        self.dma_worker = types.SimpleNamespace(dma_read_ptr=0)


class _FakeCollector:
    def __init__(self, **kwargs):
        self.process_provider = kwargs["process_provider"]
        self.gpu_provider = kwargs["gpu_provider"]
        self.pipeline_provider = kwargs["pipeline_provider"]
        self.events = []
        self.running = False

    def start(self):
        self.running = True
        self.sample_once(reason="start")

    def sample_once(self, *, reason):
        event = {
            "timestamp_s": float(len(self.events)),
            "reason": reason,
            "process": self.process_provider(),
            "gpu": self.gpu_provider(),
            "pipeline": self.pipeline_provider(),
            "collector_errors": [],
        }
        self.events.append(event)
        return event

    def stop(self, *, timeout_s=5.0):
        del timeout_s
        self.running = False


def _gpu_snapshot():
    return {
        "device_index": 0,
        "utilization_pct": 25.0,
        "memory_used_bytes": 1024,
        "memory_total_bytes": 4096,
    }


def test_training_health_lifecycle_captures_ready_progress_and_clean_exit():
    pipeline = _FakePipeline()
    optimizer = {"steps": 0, "samples": 0}
    lifecycle = TrainingHealthLifecycle(
        expected_workers=2,
        pipeline=pipeline,
        block_size=4,
        optimizer_steps_provider=lambda: optimizer["steps"],
        samples_processed_provider=lambda: optimizer["samples"],
        gpu_provider=_gpu_snapshot,
        collector_factory=_FakeCollector,
        expert_timeout_s=30.0,
    )
    processes = [_FakeProcess(101), _FakeProcess(102)]

    lifecycle.start()
    lifecycle.register_processes(processes)
    for handle in lifecycle.worker_handles:
        handle.mark_ready()
    lifecycle.sample_once(reason="workers_ready")

    lifecycle.worker_handles[0].begin_expert_call(call_id=7, timeout_s=30.0)
    lifecycle.sample_once(reason="expert_in_flight")
    lifecycle.worker_handles[0].finish_expert_call()
    lifecycle.worker_handles[0].mark_published()
    pipeline.ring_buffer.shared_reserve_ptr.value = 4
    pipeline.dma_worker.dma_read_ptr = 4
    pipeline.compute_ptr = 4
    optimizer.update(steps=1, samples=4)
    lifecycle.sample_once(reason="optimizer_progress")

    for handle in lifecycle.worker_handles:
        handle.complete.value = 1
    lifecycle.release_workers()
    for process in processes:
        process._alive = False
        process.exitcode = 0
    report = lifecycle.finalize(reference_throughput=None)

    assert report["collector"] == "benchmark_health_live"
    assert report["live_sampling"] is True
    assert report["paper_eligible"] is True
    assert report["validation"]["valid"] is True
    assert [event["reason"] for event in report["events"]] == [
        "start",
        "workers_ready",
        "expert_in_flight",
        "optimizer_progress",
        "workers_release",
        "final",
    ]
    in_flight = report["events"][2]["pipeline"]["active_expert_calls"]
    assert in_flight[0]["call_id"] == 7
    final = report["events"][-1]
    assert [worker["exit_code"] for worker in final["process"]["workers"]] == [0, 0]
    assert final["pipeline"]["reserve_ptr"] == 4
    assert final["pipeline"]["dma_ptr"] == 4
    assert final["pipeline"]["compute_ptr"] == 4
    assert final["pipeline"]["optimizer_steps"] == 1


def test_completed_workers_waiting_for_release_do_not_create_producer_backlog():
    lifecycle = TrainingHealthLifecycle(
        expected_workers=1,
        pipeline=_FakePipeline(),
        block_size=4,
        optimizer_steps_provider=lambda: 0,
        samples_processed_provider=lambda: 0,
        gpu_provider=_gpu_snapshot,
        collector_factory=_FakeCollector,
        expert_timeout_s=30.0,
    )
    lifecycle.register_processes([_FakeProcess(105)])
    lifecycle.worker_handles[0].mark_ready()
    lifecycle.worker_handles[0].complete.value = 1

    snapshot = lifecycle.collector.pipeline_provider()

    assert snapshot["producer_backlog"] is False


def test_training_health_abort_stops_collector_and_preserves_failure_evidence():
    lifecycle = TrainingHealthLifecycle(
        expected_workers=1,
        pipeline=_FakePipeline(),
        block_size=4,
        optimizer_steps_provider=lambda: 0,
        samples_processed_provider=lambda: 0,
        gpu_provider=_gpu_snapshot,
        collector_factory=_FakeCollector,
        expert_timeout_s=30.0,
    )
    process = _FakeProcess(103)
    lifecycle.start()
    lifecycle.register_processes([process])
    lifecycle.worker_handles[0].mark_ready()
    process._alive = False
    process.exitcode = 9

    report = lifecycle.abort(reason="worker_failure", reference_throughput=None)

    assert lifecycle.running is False
    assert report["status"] == "fail"
    assert report["paper_eligible"] is False
    assert report["events"][-1]["reason"] == "worker_failure"
    assert any(
        violation["code"] == "worker_nonzero_exit"
        for violation in report["validation"]["violations"]
    )


def test_resumed_training_uses_run_local_samples_for_optimizer_backlog():
    pipeline = _FakePipeline()
    counters = {"optimizer": 10, "samples": 100}
    lifecycle = TrainingHealthLifecycle(
        expected_workers=1,
        pipeline=pipeline,
        block_size=4,
        optimizer_steps_provider=lambda: counters["optimizer"],
        samples_processed_provider=lambda: counters["samples"],
        gpu_provider=_gpu_snapshot,
        collector_factory=_FakeCollector,
        expert_timeout_s=30.0,
    )
    pipeline.ring_buffer.shared_reserve_ptr.value = 4
    pipeline.dma_worker.dma_read_ptr = 4
    pipeline.compute_ptr = 4

    before_optimizer = lifecycle.collector.pipeline_provider()
    counters["samples"] = 104
    after_optimizer = lifecycle.collector.pipeline_provider()

    assert before_optimizer["optimizer_backlog"] is True
    assert before_optimizer["gpu_active"] is True
    assert after_optimizer["optimizer_backlog"] is False
