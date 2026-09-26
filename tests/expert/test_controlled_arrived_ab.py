import copy
import random

import numpy as np
import pytest
import torch

import expert.controlled_arrived_ab as controlled_module
from expert.controlled_arrived_ab import (
    audit_checkpoint_directory,
    build_arg_parser,
    compose_training_summary,
    capture_torch_rng_state,
    create_paired_runtimes,
    collect_frozen_validation_trajectory,
    enforce_pilot_gate,
    load_frozen_trajectory,
    prepare_new_output_directories,
    process_shared_frontier,
    replay_frozen_batches,
    restore_torch_rng_state,
    run_paired_rng,
    save_frozen_trajectory,
    seed_training_rngs,
    train_paired_on_batch,
)
from expert.expert_running import FEATURE_DIM


def test_seed_training_rngs_forwards_seed_to_every_rng(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setattr(np.random, "seed", lambda value: calls.append(("numpy", value)))
    monkeypatch.setattr(torch, "manual_seed", lambda value: calls.append(("torch", value)))
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda value: calls.append(("cuda", value)),
    )

    seed_training_rngs(42)

    assert calls == [
        ("python", 42),
        ("numpy", 42),
        ("torch", 42),
        ("cuda", 42),
    ]


def test_run_paired_rng_replays_pre_a_state_and_keeps_post_a_state():
    torch.manual_seed(123)
    pre_state = capture_torch_rng_state()

    restore_torch_rng_state(pre_state)
    expected_a = torch.rand(8)
    expected_after = torch.rand(8)

    restore_torch_rng_state(pre_state)
    observed_a, observed_b = run_paired_rng(
        lambda: torch.rand(8),
        lambda: torch.rand(8),
    )
    observed_after = torch.rand(8)

    assert torch.equal(observed_a, expected_a)
    assert torch.equal(observed_b, expected_a)
    assert torch.equal(observed_after, expected_after)


def test_run_paired_rng_restores_post_a_state_when_b_raises():
    torch.manual_seed(456)
    pre_state = capture_torch_rng_state()

    restore_torch_rng_state(pre_state)
    _ = torch.rand(4)
    expected_after = torch.rand(4)

    restore_torch_rng_state(pre_state)

    def fail_after_random_draw():
        _ = torch.rand(4)
        raise RuntimeError("branch-b failed")

    with pytest.raises(RuntimeError, match="branch-b failed"):
        run_paired_rng(lambda: torch.rand(4), fail_after_random_draw)

    assert torch.equal(torch.rand(4), expected_after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paired_cuda_forward_uses_identical_dropout_mask_and_batch_object():
    device = torch.device("cuda:0")
    model_a = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(8, 5),
    ).to(device)
    model_b = copy.deepcopy(model_a)
    model_a.train()
    model_b.train()
    materialized_batch = torch.randn(32, 8, device=device)
    seen_ids = []

    def forward(model):
        seen_ids.append(id(materialized_batch))
        return model(materialized_batch)

    logits_a, logits_b = run_paired_rng(
        lambda: forward(model_a),
        lambda: forward(model_b),
    )

    assert seen_ids == [id(materialized_batch), id(materialized_batch)]
    assert torch.equal(logits_a, logits_b)


def _trajectory_fixture():
    raw = np.zeros((3, 4, FEATURE_DIM), dtype=np.uint16)
    raw[:, :, 0] = 0
    raw[:, :, 1] = np.arange(4, dtype=np.uint16)
    raw[:, :, 2] = np.arange(3, dtype=np.uint16).reshape(3, 1)
    raw[:, :, 4] = 7
    raw[:, :, 6] = np.array([1, 2, 3, 4], dtype=np.uint16)
    raw[0, :, 7] = 1
    metadata = {
        "map_name": "test-map",
        "seed": 42,
        "num_steps": 3,
        "num_agents": 4,
        "max_episode_steps": 32,
    }
    return raw, metadata


def test_frozen_trajectory_round_trip_validates_contract(tmp_path):
    raw, metadata = _trajectory_fixture()
    path = tmp_path / "trajectory.npz"

    digest = save_frozen_trajectory(path, raw, metadata)
    loaded = load_frozen_trajectory(path, expected=metadata)

    assert loaded.digest == digest
    assert loaded.metadata == metadata
    assert np.array_equal(loaded.raw_batches, raw)
    assert loaded.raw_batches.dtype == np.uint16


def test_frozen_trajectory_rejects_modified_rows_with_stale_digest(tmp_path):
    raw, metadata = _trajectory_fixture()
    path = tmp_path / "trajectory.npz"
    save_frozen_trajectory(path, raw, metadata)
    with np.load(path, allow_pickle=False) as payload:
        stored_meta = payload["metadata_json"].copy()
        stored_digest = payload["sha256"].copy()
    raw[1, 2, 2] += 1
    np.savez_compressed(
        path,
        raw_batches=raw,
        metadata_json=stored_meta,
        sha256=stored_digest,
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_frozen_trajectory(path, expected=metadata)


def test_frozen_trajectory_rejects_metadata_and_refresh_flag_drift(tmp_path):
    raw, metadata = _trajectory_fixture()
    path = tmp_path / "trajectory.npz"
    save_frozen_trajectory(path, raw, metadata)

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_frozen_trajectory(path, expected={**metadata, "seed": 43})

    raw[2, 0, 7] = 2
    with pytest.raises(ValueError, match="refresh flags"):
        save_frozen_trajectory(tmp_path / "bad.npz", raw, metadata)


def test_collect_frozen_validation_trajectory_calls_expert_once_per_step(tmp_path):
    class FakeUnwrapped:
        def __init__(self):
            self.step_idx = 0

        def _obs(self):
            return [
                {"global_xy": (5 + self.step_idx, 5), "global_target_xy": (9, 9)},
                {"global_xy": (5, 6 + self.step_idx), "global_target_xy": (8, 8)},
            ]

    class FakeEnv:
        def __init__(self):
            self.env = type("Nested", (), {"unwrapped": FakeUnwrapped()})()

        def step(self, actions):
            self.env.unwrapped.step_idx += 1
            return None, None, [False, False], [False, False], {}

        def reset(self):
            self.env.unwrapped.step_idx = 0

    class FakePolicy:
        def __init__(self):
            self.act_calls = 0
            self.reset_calls = 0

        def reset_states(self, env):
            self.reset_calls += 1

        def act(self, observations):
            self.act_calls += 1
            return np.array([1, 4], dtype=np.uint8)

    env = FakeEnv()
    policy = FakePolicy()
    output = tmp_path / "frozen.npz"

    result = collect_frozen_validation_trajectory(
        output=output,
        maps_path="unused.yaml",
        map_name="test-map",
        num_agents=2,
        num_steps=3,
        seed=42,
        max_episode_steps=32,
        expert_timeouts=[1.0, 5.0],
        env_builder=lambda **kwargs: env,
        policy_factory=lambda **kwargs: policy,
    )

    loaded = load_frozen_trajectory(
        output,
        expected={"map_name": "test-map", "seed": 42, "num_steps": 3, "num_agents": 2},
    )
    assert result["sha256"] == loaded.digest
    assert policy.act_calls == 3
    assert policy.reset_calls == 1
    assert loaded.raw_batches[:, :, 6].tolist() == [[1, 4]] * 3
    assert loaded.raw_batches[0, :, 7].tolist() == [1, 1]
    assert loaded.raw_batches[1:, :, 7].tolist() == [[0, 0], [0, 0]]
    assert loaded.raw_batches[:, 0, 2].tolist() == [0, 1, 2]


class _TinyRuntime:
    def __init__(self, *, train_on_arrived_agents, **kwargs):
        self.train_on_arrived_agents = train_on_arrived_agents
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2)
        self.seen_batches = []

    def train_step_from_batch(self, batch):
        self.seen_batches.append(batch)
        logits = self.model(batch)
        loss = logits.square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.detach()


def test_create_paired_runtimes_clones_every_initial_state_and_only_changes_mask():
    runtime_a, runtime_b = create_paired_runtimes(
        model_seed=42,
        runtime_factory=_TinyRuntime,
    )

    assert runtime_a.train_on_arrived_agents is True
    assert runtime_b.train_on_arrived_agents is False
    for key, value_a in runtime_a.model.state_dict().items():
        assert torch.equal(value_a, runtime_b.model.state_dict()[key])
    assert runtime_a.optimizer.state_dict() == runtime_b.optimizer.state_dict()
    assert runtime_a.scheduler.state_dict() == runtime_b.scheduler.state_dict()

    other_a, _ = create_paired_runtimes(model_seed=43, runtime_factory=_TinyRuntime)
    assert any(
        not torch.equal(value, other_a.model.state_dict()[key])
        for key, value in runtime_a.model.state_dict().items()
    )


def test_train_paired_on_batch_reuses_object_and_preserves_input_tensor():
    runtime_a, runtime_b = create_paired_runtimes(
        model_seed=42,
        runtime_factory=_TinyRuntime,
    )
    batch = torch.randn(7, 3)
    original = batch.clone()

    loss_a, loss_b = train_paired_on_batch(runtime_a, runtime_b, batch)

    assert runtime_a.seen_batches == [batch]
    assert runtime_b.seen_batches == [batch]
    assert runtime_a.seen_batches[0] is runtime_b.seen_batches[0]
    assert torch.equal(batch, original)
    assert torch.equal(loss_a, loss_b)


def test_prepare_new_output_directories_rejects_any_existing_path(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    prepare_new_output_directories(path_a, path_b)
    assert path_a.is_dir() and path_b.is_dir()

    with pytest.raises(FileExistsError, match="must not already exist"):
        prepare_new_output_directories(path_a, tmp_path / "new-b")


@pytest.mark.parametrize(
    ("rows_s", "updates_a", "updates_b", "should_pass"),
    [
        (20_000.0, 1000, 1000, True),
        (19_999.9, 1000, 1000, False),
        (20_000.0, 999, 1000, False),
        (20_000.0, 1000, 999, False),
    ],
)
def test_pilot_gate_is_numeric_and_stops_with_complete_diagnostics(
    rows_s, updates_a, updates_b, should_pass
):
    stopped = []
    diagnostic = {
        "worker_states": ["alive"] * 4,
        "worker_exit_codes": [None] * 4,
        "reserve_ptr": 1024,
        "dma_ptr": 1024,
        "compute_ptr": 1024,
        "gpu_utilization": 75,
        "gpu_memory_mib": 1200,
        "elapsed_s": 51.2,
        "rows_processed": 1_024_000,
        "rows_s": rows_s,
        "updates_a": updates_a,
        "updates_b": updates_b,
    }

    if should_pass:
        assert enforce_pilot_gate(diagnostic, stop_workers=lambda: stopped.append(True)) is None
        assert stopped == []
    else:
        with pytest.raises(RuntimeError, match="paired pilot gate failed") as exc_info:
            enforce_pilot_gate(diagnostic, stop_workers=lambda: stopped.append(True))
        assert stopped == [True]
        for key in diagnostic:
            assert key in str(exc_info.value)


def test_process_shared_frontier_materializes_once_and_counts_effective_rows():
    class MaterializedBatch:
        def __init__(self):
            self.arrived = torch.tensor([True, False, True, False])
            self.payload = torch.arange(4)

    class FakeRuntime:
        def __init__(self, train_on_arrived_agents):
            self.train_on_arrived_agents = train_on_arrived_agents
            self.build_calls = 0
            self.train_batches = []

        def build_batch(self, simulator, raw_batch, materialize_edges):
            self.build_calls += 1
            assert materialize_edges is True
            return MaterializedBatch()

        def train_step_from_batch(self, batch):
            self.train_batches.append(batch)
            return torch.tensor(0.25 if self.train_on_arrived_agents else 0.5)

    runtime_a = FakeRuntime(True)
    runtime_b = FakeRuntime(False)
    stats = {
        "arrived_rows": 0,
        "effective_rows_a": 0,
        "effective_rows_b": 0,
        "zero_effective_batches_a": 0,
        "zero_effective_batches_b": 0,
    }

    result = process_shared_frontier(
        runtime_a=runtime_a,
        runtime_b=runtime_b,
        simulator=object(),
        raw_batch=torch.zeros((4, FEATURE_DIM), dtype=torch.int16),
        supervision_stats=stats,
    )

    assert runtime_a.build_calls == 1
    assert runtime_b.build_calls == 0
    assert runtime_a.train_batches[0] is runtime_b.train_batches[0]
    assert result == (0.25, 0.5)
    assert stats == {
        "arrived_rows": 2,
        "effective_rows_a": 4,
        "effective_rows_b": 2,
        "zero_effective_batches_a": 0,
        "zero_effective_batches_b": 0,
    }


def _write_checkpoint(path, step, metadata):
    torch.save(
        {
            "model": {},
            "optimizer": {},
            "optimizer_step": step,
            "loss": 0.5,
            "meta": metadata,
        },
        path / f"ckpt_step{step:08d}_loss0.500000.pt",
    )


def test_audit_checkpoint_directory_validates_steps_and_control_metadata(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    expected_meta = {
        "model_seed": 42,
        "branch": "with_arrived",
        "train_on_arrived_agents": True,
        "paired_run_id": "paired-test",
        "map_names": ["m0", "m1", "m2", "m3"],
        "scheduler_total_steps": 100000,
        "max_episode_steps": 256,
        "expert_timeouts": [1.0, 5.0, 10.0, 30.0],
    }
    _write_checkpoint(checkpoint_dir, 1000, expected_meta)
    _write_checkpoint(checkpoint_dir, 2000, expected_meta)

    audited = audit_checkpoint_directory(
        checkpoint_dir,
        expected_steps=[1000, 2000],
        expected_meta=expected_meta,
    )

    assert [row["optimizer_step"] for row in audited] == [1000, 2000]


def test_audit_checkpoint_directory_rejects_metadata_drift(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    _write_checkpoint(checkpoint_dir, 1000, {"model_seed": 43})

    with pytest.raises(ValueError, match="model_seed"):
        audit_checkpoint_directory(
            checkpoint_dir,
            expected_steps=[1000],
            expected_meta={"model_seed": 42},
        )


def test_replay_frozen_batches_refreshes_derived_then_full_compact_state_each_step():
    raw = np.zeros((2, 2, FEATURE_DIM), dtype=np.uint16)
    raw[:, :, 1] = np.arange(2, dtype=np.uint16)
    raw[0, :, 2:6] = [[1, 1, 5, 5], [2, 2, 6, 6]]
    raw[1, :, 2:6] = [[3, 3, 7, 7], [4, 4, 8, 8]]
    raw[:, :, 6] = [[1, 2], [3, 4]]
    raw[0, :, 7] = 1

    class FakeSimulator:
        def __init__(self):
            self.calls = []
            self.current = None

        def update_derived_state(self, rows, length):
            self.calls.append(("derived", rows.cpu().clone(), int(length)))

        def refresh_compact_state_from_raw_batch(self, batch):
            self.current = batch.cpu().clone()
            self.calls.append(("compact", self.current.clone()))

    simulator = FakeSimulator()
    predictor_positions = []

    def predict(sim, batch):
        assert torch.equal(sim.current, batch.cpu())
        predictor_positions.append(sim.current[:, 2:6].tolist())
        return batch[:, 6].to(torch.int64).cpu().numpy()

    metrics = replay_frozen_batches(
        raw_batches=raw,
        simulator=simulator,
        device="cpu",
        predict_actions=predict,
    )

    assert [call[0] for call in simulator.calls] == ["derived", "compact", "compact"]
    assert predictor_positions == [raw[0, :, 2:6].tolist(), raw[1, :, 2:6].tolist()]
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["nonstay_accuracy"] == 1.0
    assert metrics["overall_total"] == 4
    assert metrics["nonstay_total"] == 4


def test_train_cli_parses_all_controlled_parameters_explicitly():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "train",
            "--num_steps", "10000",
            "--num_agents", "256",
            "--num_experts", "4",
            "--batch_threshold", "1024",
            "--train_batch_size", "1024",
            "--seed", "42",
            "--model_seed", "42",
            "--paired_run_id", "paired",
            "--maps_path", "maps/maps.yaml",
            "--map_names", "m0,m1,m2,m3",
            "--max_episode_steps", "256",
            "--expert_timeouts", "1,5,10,30",
            "--pyg_builder_mode", "local_gather",
            "--pyg_local_gather_impl", "auto",
            "--lr_start", "0.001",
            "--lr_end", "0.000001",
            "--lr_scheduler", "cosine-annealing",
            "--scheduler_total_steps", "100000",
            "--grad_clip_norm", "0",
            "--checkpoint_interval", "1000",
            "--checkpoint_dir_with_arrived", "a",
            "--checkpoint_dir_without_arrived", "b",
            "--summary_json", "summary.json",
        ]
    )

    assert args.command == "train"
    assert args.model_seed == 42
    assert args.map_names == "m0,m1,m2,m3"
    assert args.max_episode_steps == 256
    assert args.expert_timeouts == "1,5,10,30"
    assert args.scheduler_total_steps == 100000
    assert args.checkpoint_dir_with_arrived == "a"
    assert args.checkpoint_dir_without_arrived == "b"


def test_main_forwards_train_freeze_and_evaluate_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(
        controlled_module,
        "run_controlled_paired_training",
        lambda **kwargs: calls.append(("train", kwargs)),
    )
    monkeypatch.setattr(
        controlled_module,
        "collect_frozen_validation_trajectory",
        lambda **kwargs: calls.append(("freeze", kwargs)) or {"sha256": "abc"},
    )
    monkeypatch.setattr(
        controlled_module,
        "evaluate_paired_checkpoint_directories",
        lambda **kwargs: calls.append(("evaluate", kwargs)),
    )

    controlled_module.main(
        [
            "train", "--num_steps", "10000", "--num_agents", "256",
            "--num_experts", "4", "--batch_threshold", "1024",
            "--train_batch_size", "1024", "--seed", "42", "--model_seed", "42",
            "--paired_run_id", "paired", "--maps_path", "maps/maps.yaml",
            "--map_names", "m0,m1,m2,m3", "--max_episode_steps", "256",
            "--expert_timeouts", "1,5,10,30", "--pyg_builder_mode", "local_gather",
            "--pyg_local_gather_impl", "auto", "--lr_start", "0.001",
            "--lr_end", "0.000001", "--lr_scheduler", "cosine-annealing",
            "--scheduler_total_steps", "100000", "--grad_clip_norm", "0",
            "--checkpoint_interval", "1000", "--checkpoint_dir_with_arrived", "a",
            "--checkpoint_dir_without_arrived", "b", "--summary_json", "summary.json",
        ]
    )
    controlled_module.main(
        [
            "freeze-validation", "--output", "trajectory.npz", "--maps_path", "maps/maps.yaml",
            "--map_name", "test-map", "--num_agents", "256", "--num_steps", "100",
            "--seed", "42", "--max_episode_steps", "256", "--expert_timeouts", "1,5,10,30",
        ]
    )
    controlled_module.main(
        [
            "evaluate", "--trajectory", "trajectory.npz",
            "--checkpoint_dir_with_arrived", "a", "--checkpoint_dir_without_arrived", "b",
            "--output_json", "accuracy.json", "--output_csv", "accuracy.csv", "--device", "cuda:0",
        ]
    )

    assert calls[0][0] == "train"
    assert calls[0][1]["model_seed"] == 42
    assert calls[0][1]["map_names"] == ["m0", "m1", "m2", "m3"]
    assert calls[0][1]["expert_timeouts"] == [1.0, 5.0, 10.0, 30.0]
    assert calls[1] == (
        "freeze",
        {
            "output": "trajectory.npz", "maps_path": "maps/maps.yaml", "map_name": "test-map",
            "num_agents": 256, "num_steps": 100, "seed": 42, "max_episode_steps": 256,
            "expert_timeouts": [1.0, 5.0, 10.0, 30.0],
        },
    )
    assert calls[2][0] == "evaluate"
    assert calls[2][1]["trajectory"] == "trajectory.npz"
    assert calls[2][1]["device"] == "cuda:0"


def test_compose_training_summary_persists_pilot_and_explicit_branch_contract():
    runtime_a = type("Runtime", (), {"training_config": lambda self: {"train_on_arrived_agents": True}})()
    runtime_b = type("Runtime", (), {"training_config": lambda self: {"train_on_arrived_agents": False}})()
    pilot = {"rows_s": 25_000.0, "updates_a": 1000, "updates_b": 1000}
    summary = compose_training_summary(
        base={"paired_run_id": "paired", "model_seed": 42, "samples_processed": 1024},
        runtime_a=runtime_a,
        runtime_b=runtime_b,
        checkpoints_a=[{"optimizer_step": 1000}],
        checkpoints_b=[{"optimizer_step": 1000}],
        pilot_diagnostic=pilot,
    )

    assert summary["paired_run_id"] == "paired"
    assert summary["model_seed"] == 42
    assert summary["pilot_diagnostic"] == pilot
    assert summary["branches"]["with_arrived"]["branch"] == "with_arrived"
    assert summary["branches"]["with_arrived"]["train_on_arrived_agents"] is True
    assert summary["branches"]["without_arrived"]["branch"] == "without_arrived"
    assert summary["branches"]["without_arrived"]["train_on_arrived_agents"] is False
    assert summary["branches"]["with_arrived"]["checkpoints"] == [{"optimizer_step": 1000}]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_magat_materialized_batch_has_identical_paired_initial_logits():
    from expert.expert_running import initial_refresh_flags, put_maps_into_registry
    from mapf_cuda.simulation.grids import extract_single_env_grid
    from mapf_cuda.training import topology_async as topology_training
    import grid_world_cpp as ext

    put_maps_into_registry("maps/maps.yaml")
    env = topology_training._build_pogema_topology_map_env(
        num_agents=4,
        map_name="mazes-s0_wc8_od55",
        seed=42,
        max_episode_steps=32,
        collision_system="soft",
    )
    observations = env.env.unwrapped._obs()
    raw_np = topology_training._build_step_data(
        observations,
        np.zeros(4, dtype=np.uint16),
        env_id=0,
        reset_flag=initial_refresh_flags(4),
    )
    raw = torch.from_numpy(raw_np.astype(np.int16, copy=False)).to("cuda:0")
    simulator = ext.StatelessGridWorldSimulator(
        extract_single_env_grid(env, device="cuda:0"),
        4,
        3,
    )
    simulator.update_derived_state(raw, raw.shape[0])
    simulator.refresh_compact_state_from_raw_batch(raw)
    runtime_a, runtime_b = create_paired_runtimes(
        model_seed=42,
        device="cuda:0",
        lr=1e-3,
        lr_end=1e-6,
        lr_scheduler="cosine-annealing",
        scheduler_total_steps=100000,
        grad_clip_norm=None,
    )
    batch = runtime_a.build_batch(simulator, raw, materialize_edges=True)
    runtime_a.model.train()
    runtime_b.model.train()

    logits_a, logits_b = run_paired_rng(
        lambda: runtime_a.model(batch.x, batch),
        lambda: runtime_b.model(batch.x, batch),
    )

    assert torch.equal(logits_a, logits_b)
