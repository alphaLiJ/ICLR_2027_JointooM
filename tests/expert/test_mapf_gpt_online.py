import types
from pathlib import Path

import numpy as np
import pytest
import torch


def _observations(offset: int = 5):
    return [
        {
            "global_xy": (offset + 10, offset + 11 + agent_id),
            "global_target_xy": (offset + 20, offset + 21 + agent_id),
        }
        for agent_id in range(2)
    ]


def test_step_builder_snapshots_history_before_current_action_and_resets():
    from expert.mapf_gpt_online import MapfGPTStepBuilder

    builder = MapfGPTStepBuilder(env_id=1, num_agents=2)
    actions_t0 = np.array([1, 4], dtype=np.uint16)
    rows_t0 = builder.build_rows(
        _observations(), actions_t0, refresh_flags=np.ones(2, dtype=np.uint16)
    )

    assert rows_t0.shape == (2, 13)
    assert rows_t0.dtype == np.uint16
    assert np.all(rows_t0[:, 0] == 1)
    assert np.array_equal(rows_t0[:, 1], np.arange(2, dtype=np.uint16))
    assert np.all(rows_t0[:, 8:13] == 5)
    assert np.array_equal(rows_t0[:, 6], actions_t0)

    builder.record_actions(actions_t0)
    actions_t1 = np.array([2, 3], dtype=np.uint16)
    rows_t1 = builder.build_rows(
        _observations(), actions_t1, refresh_flags=np.zeros(2, dtype=np.uint16)
    )
    assert np.array_equal(rows_t1[:, 8:13], np.array([[5, 5, 5, 5, 1], [5, 5, 5, 5, 4]]))
    assert not np.any(rows_t1[:, 8:13] == actions_t1[:, None])

    builder.reset()
    rows_reset = builder.build_rows(
        _observations(), actions_t1, refresh_flags=np.ones(2, dtype=np.uint16)
    )
    assert np.all(rows_reset[:, 8:13] == 5)


def test_default_magat_feature_width_remains_isolated():
    from expert.expert_running import FEATURE_DIM, ExtremeMAPFPipeline
    from expert.mapf_gpt_schema import MAPFGPT_FEATURE_DIM

    assert FEATURE_DIM == 8
    assert MAPFGPT_FEATURE_DIM == 13
    assert ExtremeMAPFPipeline.__init__.__defaults__[2] == 8


def test_cli_arrived_agent_training_is_explicit_and_defaults_on():
    from expert.mapf_gpt_online import _parse_args

    assert _parse_args([]).train_on_arrived_agents is True
    assert (
        _parse_args(["--no-train-on-arrived-agents"]).train_on_arrived_agents
        is False
    )


def test_cli_exposes_optional_microbatch_size():
    from expert.mapf_gpt_online import _parse_args

    assert _parse_args([]).microbatch_size is None
    assert _parse_args(["--microbatch-size", "64"]).microbatch_size == 64


def test_cli_exposes_periodic_checkpoint_validation_and_resume_options():
    from expert.mapf_gpt_online import _parse_args

    args = _parse_args(
        [
            "--checkpoint-dir",
            "ckpts",
            "--checkpoint-interval",
            "1000",
            "--top-k-checkpoints",
            "3",
            "--checkpoint-selection-mode",
            "validation_accuracy",
            "--validation-datasets",
            "a.npz,b.npz",
            "--validation-batch-size",
            "128",
            "--resume-checkpoint",
            "resume.pt",
        ]
    )

    assert args.checkpoint_dir == "ckpts"
    assert args.checkpoint_interval == 1000
    assert args.top_k_checkpoints == 3
    assert args.checkpoint_selection_mode == "validation_accuracy"
    assert args.validation_datasets == "a.npz,b.npz"
    assert args.validation_batch_size == 128
    assert args.resume_checkpoint == "resume.pt"




def test_counter_delta_reports_only_current_resumed_run_progress():
    from expert.mapf_gpt_online import _counter_delta

    assert _counter_delta(96, 64, name="samples_processed") == 32
    with pytest.raises(RuntimeError, match="samples_processed regressed"):
        _counter_delta(63, 64, name="samples_processed")


def test_worker_resets_history_after_standard_mapf_episode_done(monkeypatch):
    import expert.expert_running as expert_running
    import expert.mapf_gpt_online as online
    import mapf_cuda.training.topology_async as topology_training

    observations = _observations()

    class FakeUnwrapped:
        def _obs(self):
            return observations

    class FakeEnv:
        def __init__(self):
            self.env = type("Inner", (), {"unwrapped": FakeUnwrapped()})()
            self.reset_count = 0

        def step(self, actions):
            return observations, None, [True, True], [False, False], None

        def reset(self):
            self.reset_count += 1
            return observations, {}

    class FakePolicy:
        def __init__(self, **kwargs):
            self.reset_count = 0

        def reset_states(self, env):
            self.reset_count += 1

        def act(self, observations):
            return np.array([1, 4], dtype=np.uint16)

    class CapturingRing:
        def __init__(self):
            self.rows = []

        def reserve_and_write(self, rows):
            self.rows.append(rows.copy())

    fake_env = FakeEnv()
    ring = CapturingRing()
    monkeypatch.setattr(
        topology_training,
        "_build_pogema_topology_map_env",
        lambda **kwargs: fake_env,
    )
    monkeypatch.setattr(expert_running, "put_maps_into_registry", lambda *args, **kwargs: None)
    monkeypatch.setattr(online, "LacamExpertPolicy", FakePolicy)

    online.mapf_gpt_expert_worker_loop(
        0,
        ring,
        maps_path="unused.yaml",
        map_name="unused",
        num_agents=2,
        num_steps=2,
        seed=1,
        max_episode_steps=256,
    )

    assert len(ring.rows) == 2
    assert np.all(ring.rows[0][:, 8:13] == 5)
    assert np.all(ring.rows[1][:, 8:13] == 5)
    assert np.all(ring.rows[1][:, 7] == 1)
    assert fake_env.reset_count == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_online_pipeline_consumes_only_complete_env_aligned_stage():
    import grid_world_cpp as ext
    from expert.mapf_gpt_online import MapfGPTOnlinePipeline
    from expert.mapf_gpt_schema import build_mapf_gpt_rows

    grids = torch.zeros(
        (2, ext.COMPILED_MAP_W, ext.COMPILED_MAP_H),
        dtype=torch.int32,
        device="cuda",
    )
    cuda_builder = ext.MapfGPTObservationBuilder(grids, 2)

    class CapturingRuntime:
        def __init__(self):
            self.calls = []
            self.streams = []

        def train_step(self, builder, raw_stage):
            self.streams.append(torch.cuda.current_stream())
            self.calls.append(raw_stage.detach().cpu().clone())
            builder.build_tokens(raw_stage)
            return {"loss": 1.0, "samples": int(raw_stage.shape[0])}

    runtime = CapturingRuntime()
    pipeline = MapfGPTOnlinePipeline(
        cuda_builder,
        runtime,
        capacity=8,
        batch_threshold=4,
        device="cuda:0",
    )
    pipeline.initialize()
    try:
        history = np.full((2, 5), 5, dtype=np.uint16)
        env0 = build_mapf_gpt_rows(
            _observations(),
            np.array([0, 1], dtype=np.uint16),
            env_id=0,
            refresh_flags=1,
            history=history,
        )
        env1 = build_mapf_gpt_rows(
            _observations(),
            np.array([2, 3], dtype=np.uint16),
            env_id=1,
            refresh_flags=1,
            history=history,
        )

        pipeline.reserve_and_write(env0)
        assert pipeline.dma_worker.run_once() is None
        assert pipeline.consume_once() is None
        assert runtime.calls == []

        pipeline.reserve_and_write(env1)
        assert pipeline.dma_worker.run_once() == (0, 4)
        metrics = pipeline.consume_once()
        assert metrics == {"loss": 1.0, "samples": 4}
        assert len(runtime.calls) == 1
        assert runtime.streams[0] == pipeline.compute_stream
        raw = runtime.calls[0].numpy().astype(np.uint16)
        assert np.array_equal(raw[:2], env0)
        assert np.array_equal(raw[2:], env1)
        assert pipeline.compute_ptr == 4
    finally:
        pipeline.shutdown()


def test_run_topology_mapf_gpt_online_includes_health_report(monkeypatch):
    import expert.mapf_gpt_online as online
    import mapf_cuda.simulation.grids as grid_helpers
    import mapf_cuda.training.topology_async as topology_training

    monkeypatch.setattr(
        topology_training,
        "_select_topology_training_maps",
        lambda **kwargs: [("maze-0", np.zeros((8, 8), dtype=np.int32))],
    )
    monkeypatch.setattr(
        grid_helpers,
        "stack_grids_for_compiled_simulator",
        lambda grids, device: (torch.zeros((1, 8, 8), dtype=torch.int32), None),
    )

    class FakeBuilder:
        def __init__(self, grids_cuda, num_agents):
            del grids_cuda, num_agents
            self.diagnostics = torch.tensor([1, 2, 3], dtype=torch.int32)

    monkeypatch.setitem(__import__("sys").modules, "grid_world_cpp", types.SimpleNamespace(MapfGPTObservationBuilder=FakeBuilder))

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.optimizer_steps = 0
            self.samples_processed = 0
            self.supervised_samples_processed = 0
            self.stage_samples_seen = 0
            self.model = types.SimpleNamespace(state_dict=lambda: {})
            self.optimizer = types.SimpleNamespace(state_dict=lambda: {})

        def train_step(self, builder, raw_stage):
            del builder
            self.optimizer_steps += 1
            self.samples_processed += int(raw_stage.shape[0])
            self.supervised_samples_processed += int(raw_stage.shape[0])
            self.stage_samples_seen += int(raw_stage.shape[0])
            return {"loss": 0.25, "skipped": False}

        def load_checkpoint(self, path):
            return None

        def checkpoint_metadata(self):
            return {"schema_version": 1}

        def checkpoint_state(self):
            return {}

    monkeypatch.setattr(online, "MapfGPTRuntimeAdapter", FakeRuntime)

    class FakeCheckpointManager:
        def __init__(self, *args, **kwargs):
            self.enabled = True

        def maybe_save(self, **kwargs):
            return None

        def snapshot(self):
            return []

        def last_validation_metrics(self):
            return None

    monkeypatch.setattr(online, "TopKCheckpointManager", FakeCheckpointManager)

    class FakePipeline:
        def __init__(self, builder, runtime, capacity, batch_threshold, device):
            del builder, device
            self.runtime = runtime
            self.capacity = capacity
            self.batch_threshold = batch_threshold
            self.compute_ptr = 0
            self.ring_buffer = object()
            self.dma_worker = types.SimpleNamespace(
                dma_read_ptr=batch_threshold,
                start=lambda: types.SimpleNamespace(join=lambda timeout=None: None),
            )

        def initialize(self):
            return self

        def consume_once(self):
            if self.compute_ptr:
                return None
            self.compute_ptr = self.batch_threshold
            raw_stage = torch.zeros((self.batch_threshold, 13), dtype=torch.int16)
            return self.runtime.train_step(None, raw_stage)

        def get_stats(self):
            return {
                "capacity": self.capacity,
                "reserve_ptr": self.batch_threshold,
                "dma_read_ptr": self.batch_threshold,
                "compute_ptr": self.compute_ptr,
                "stage_rows": self.batch_threshold,
            }

        def shutdown(self):
            return None

    monkeypatch.setattr(online, "MapfGPTOnlinePipeline", FakePipeline)

    class FakeProcess:
        created = []

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs.get("kwargs", {})
            self.pid = 2000 + len(type(self).created)
            self.exitcode = 0
            self._alive = False
            type(self).created.append(self)

        def start(self):
            self._alive = False
            self.exitcode = 0

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

        def terminate(self):
            self._alive = False
            self.exitcode = 0

    monkeypatch.setattr(online.mp, "Process", FakeProcess)

    class FakeWorkerHealthHandle:
        pass

    class FakeTrainingHealthLifecycle:
        created = []

        def __init__(self, *, expected_workers, **kwargs):
            del kwargs
            self.worker_handles = [FakeWorkerHealthHandle() for _ in range(expected_workers)]
            self.reasons = []
            self.running = False
            type(self).created.append(self)

        def start(self):
            self.running = True
            self.reasons.append("start")

        def register_processes(self, processes):
            self.processes = list(processes)

        def wait_for_workers_ready(self, **kwargs):
            del kwargs
            self.reasons.append("workers_ready")

        def sample_once(self, *, reason):
            self.reasons.append(reason)

        def maybe_sample_progress(self, *, force=False):
            if force:
                self.reasons.append("optimizer_progress")

        def finalize(self, *, reference_throughput, reason="final"):
            del reference_throughput
            self.running = False
            self.reasons.append(reason)
            return {
                "schema_version": 1,
                "status": "pass",
                "collector": "benchmark_health_live",
                "live_sampling": True,
                "paper_eligible": True,
                "expected_workers": len(self.worker_handles),
                "events": [{"reason": item} for item in self.reasons],
                "validation": {"valid": True, "violations": []},
            }

        def release_workers(self):
            return None

        def abort(self, *, reason, reference_throughput):
            return self.finalize(reference_throughput=reference_throughput, reason=reason)

    monkeypatch.setattr(online, "TrainingHealthLifecycle", FakeTrainingHealthLifecycle)

    summary = online.run_topology_mapf_gpt_online(
        num_steps=1,
        num_agents=2,
        maps_path="ignored.yaml",
        map_names=["maze-0"],
        device="cuda:0",
    )

    assert summary["health_report"]["status"] == "pass"
    assert summary["health_report"]["live_sampling"] is True
    assert summary["health_report"]["paper_eligible"] is True
    assert summary["health_report"]["expected_workers"] == 1
    assert summary["health_report"]["validation"]["valid"] is True
    assert summary["worker_pids"] == [2000]
    assert summary["worker_exitcodes"] == [0]
    assert summary["ring_reserve_ptr"] == 2
    assert summary["dma_read_ptr"] == 2
    assert summary["compute_ptr"] == 2
    assert summary["validation_metrics"] is None
    assert FakeTrainingHealthLifecycle.created[0].reasons == [
        "start",
        "workers_ready",
        "optimizer_progress",
        "final",
    ]
    assert summary["cuda_diagnostics"] == [1, 2, 3]
    assert summary["run_optimizer_steps"] == 1
    assert summary["run_samples_processed"] == 2
    assert summary["run_supervised_samples_processed"] == 2
    assert summary["run_stage_samples_seen"] == 2
    assert summary["producer_resume_semantics"] == "new_expert_stream"
    assert summary["checkpoint_inventory"] == []
    assert summary["workers_started"] == 1
    assert summary["workers_alive_after_start"] == 0
    assert summary["samples_s"] > 0.0
    assert summary["wall_s"] > 0.0
    assert summary["mean_loss"] == pytest.approx(0.25)
    assert summary["final_loss"] == pytest.approx(0.25)
    assert summary["resume_checkpoint"] is None
    assert summary["consumer_training_state_restored"] is False
    assert summary["checkpoint_selection_mode"] == "latest"
    assert summary["checkpoint_interval"] == 1000
    assert summary["validation_datasets"] == []
    assert summary["num_envs"] == 1
    assert summary["num_agents"] == 2
    assert summary["num_steps"] == 1
    assert summary["stages_consumed"] == 1
    assert summary["schema_version"] == 1
    assert summary["health_report"]["collector"] == "benchmark_health_live"
    assert "health_handle" in FakeProcess.created[0].kwargs
    assert summary["map_names"] == ["maze-0"]
    assert FakeProcess.created
    monkeypatch.delitem(__import__("sys").modules, "grid_world_cpp", raising=False)
