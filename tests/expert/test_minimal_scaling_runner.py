from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _FakeAdapter:
    def __init__(self):
        self.reset_count = 0
        self.step_count = 0

    def reset_device_state(self):
        self.reset_count += 1

    def prepare_actions(self, actions):
        return tuple(range(actions.shape[0]))

    def synchronize_transition(self):
        return None

    def step_transition(self, _handle):
        self.step_count += 1

    def materialize_snapshot(self):
        raise AssertionError("minimal runner must not materialize snapshots")

    @property
    def consumed_input_sha256(self):
        raise AssertionError("minimal runner must not verify consumed hashes")


def test_single_backend_counts_fixed_trajectory_without_correctness_hooks():
    from expert.minimal_scaling_runner import run_single_backend

    batch = SimpleNamespace(
        num_envs=4,
        num_agents=3,
        horizon=2,
        actions=np.zeros((2, 4, 3), dtype=np.uint8),
    )
    adapter = _FakeAdapter()
    observed = {}

    def factory(backend, supplied_batch, device, *, verify_consumption):
        observed.update(
            backend=backend,
            batch=supplied_batch,
            device=device,
            verify_consumption=verify_consumption,
        )
        return adapter

    clock = iter((1.0, 3.0, 10.0, 14.0))
    rows = run_single_backend(
        backend="pogema",
        batch=batch,
        warmup_trajectories=1,
        repetitions=1,
        adapter_factory=factory,
        perf_counter=lambda: next(clock),
    )

    assert observed == {
        "backend": "pogema",
        "batch": batch,
        "device": "cpu",
        "verify_consumption": False,
    }
    assert adapter.reset_count == 2
    assert adapter.step_count == 4
    assert rows == [
        {
            "repetition": 0,
            "measured_duration_s": 4.0,
            "nominal_env_steps": 8,
            "nominal_agent_steps": 24,
            "env_steps_s": 2.0,
            "agent_steps_s": 6.0,
        }
    ]


def test_fast_pogema_adapter_skips_roundtrip_but_default_remains_strict(monkeypatch):
    import expert.transition_adapters as module
    from expert.benchmark_contract import FrozenTransitionBatch

    batch = FrozenTransitionBatch(
        instance_ids=np.asarray([1], dtype=np.int64),
        grids=np.zeros((1, 128, 128), dtype=np.uint8),
        positions=np.asarray([[[2, 2]]], dtype=np.uint16),
        goals=np.asarray([[[2, 3]]], dtype=np.uint16),
        arrived=np.zeros((1, 1), dtype=np.bool_),
        actions=np.asarray([[[4]]], dtype=np.uint8),
        horizon=1,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("roundtrip verifier was called")

    monkeypatch.setattr(module, "_verify_backend_roundtrip", forbidden)
    fast = module.make_transition_adapter(
        "pogema", batch, "cpu", verify_consumption=False
    )
    fast.reset_device_state()

    strict = module.make_transition_adapter("pogema", batch, "cpu")
    try:
        strict.reset_device_state()
    except AssertionError as exc:
        assert "roundtrip verifier" in str(exc)
    else:
        raise AssertionError("strict adapter unexpectedly skipped verification")


def test_minimal_result_archive_contains_only_throughput_metadata(tmp_path):
    from expert.minimal_scaling_runner import _write_results

    rows = [
        {
            "repetition": 0,
            "measured_duration_s": 2.0,
            "nominal_env_steps": 8,
            "nominal_agent_steps": 24,
            "env_steps_s": 4.0,
            "agent_steps_s": 12.0,
        },
        {
            "repetition": 1,
            "measured_duration_s": 1.0,
            "nominal_env_steps": 8,
            "nominal_agent_steps": 24,
            "env_steps_s": 8.0,
            "agent_steps_s": 24.0,
        },
    ]
    result = _write_results(
        tmp_path / "row",
        config={
            "backend": "pogema",
            "num_envs": 4,
            "num_agents": 3,
            "horizon": 2,
        },
        rows=rows,
    )

    assert result["median_env_steps_s"] == 6.0
    assert result["correctness_validation"] is False
    assert {path.name for path in (tmp_path / "row").iterdir()} == {
        "config.json",
        "environment.json",
        "measurements.csv",
        "result.json",
        "stdout.log",
    }


def test_minimal_pogema_spawn_smoke_uses_parent_wall_timing():
    from expert.benchmark_contract import FrozenTransitionBatch
    from expert.minimal_scaling_runner import run_pogema_multiprocess

    batch = FrozenTransitionBatch(
        instance_ids=np.asarray([10, 11], dtype=np.int64),
        grids=np.zeros((2, 128, 128), dtype=np.uint8),
        positions=np.asarray([[[2, 2]], [[3, 3]]], dtype=np.uint16),
        goals=np.asarray([[[2, 3]], [[3, 4]]], dtype=np.uint16),
        arrived=np.zeros((2, 1), dtype=np.bool_),
        actions=np.asarray([[[4], [4]]], dtype=np.uint8),
        horizon=1,
    )
    rows = run_pogema_multiprocess(
        batch=batch,
        num_workers=2,
        warmup_trajectories=0,
        repetitions=1,
        worker_timeout_s=60.0,
    )

    assert len(rows) == 1
    assert rows[0]["nominal_env_steps"] == 2
    assert rows[0]["env_steps_s"] > 0


def test_single_backend_samples_memory_outside_timed_region():
    from expert.minimal_scaling_runner import run_single_backend

    batch = SimpleNamespace(
        num_envs=2,
        num_agents=3,
        horizon=1,
        actions=np.zeros((1, 2, 3), dtype=np.uint8),
    )
    adapter = _FakeAdapter()

    class Tracker:
        def __init__(self):
            self.phases = []

        def start(self):
            self.phases.append("start")

        def sample(self, phase):
            self.phases.append(phase)

    tracker = Tracker()
    clock = iter((1.0, 2.0, 10.0, 12.0))
    run_single_backend(
        backend="cuda",
        batch=batch,
        warmup_trajectories=1,
        repetitions=1,
        adapter_factory=lambda *_args, **_kwargs: adapter,
        perf_counter=lambda: next(clock),
        memory_tracker=tracker,
    )

    assert tracker.phases == [
        "start",
        "after_adapter",
        "after_warmup",
        "after_repetition_0",
    ]


def test_failure_archive_marks_cuda_oom_as_capacity_boundary(tmp_path):
    from expert.minimal_scaling_runner import _write_failure

    result = _write_failure(
        tmp_path / "oom",
        config={
            "backend": "cuda",
            "num_envs": 1024,
            "num_agents": 512,
            "horizon": 256,
        },
        error=RuntimeError("CUDA out of memory"),
        memory={"max_nvml_process_memory_bytes": 123},
    )

    assert result["status"] == "capacity_failure"
    assert (tmp_path / "oom" / "result.json").exists()
