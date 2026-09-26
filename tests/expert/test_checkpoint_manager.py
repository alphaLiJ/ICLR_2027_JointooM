from __future__ import annotations

from pathlib import Path

import pytest
import torch

from expert.checkpoint_manager import TopKCheckpointManager, model_state_sha256


class _Runtime:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


class _StatefulRuntime(_Runtime):
    def checkpoint_state(self):
        return {"counter": 17, "buffer": torch.tensor([1, 2, 3])}


class _Evaluator:
    def __init__(self, values):
        self.values = {int(step): float(value) for step, value in values.items()}
        self.calls = []

    def __call__(self, runtime, optimizer_step):
        step = int(optimizer_step)
        self.calls.append(step)
        value = self.values[step]
        return {
            "val_overall_accuracy": value,
            "val_nonstay_accuracy": value,
            "validation_num_trajectories": 1,
            "validation_trajectory_sha256s": [f"sha-{step}"],
        }


def test_model_state_hash_supports_scalar_buffers_and_detects_changes():
    model = torch.nn.BatchNorm1d(2)
    first = model_state_sha256(model)
    assert first == model_state_sha256(model)
    model.num_batches_tracked.add_(1)
    assert model_state_sha256(model) != first


def test_periodic_manager_skips_non_boundary_and_saves_boundary(tmp_path):
    runtime = _Runtime()
    manager = TopKCheckpointManager(
        tmp_path,
        top_k=2,
        save_interval_steps=1000,
    )

    assert manager.maybe_save(loss=1.0, runtime=runtime, optimizer_step=999) is None
    assert list(tmp_path.iterdir()) == []

    saved = manager.maybe_save(
        loss=0.9,
        runtime=runtime,
        optimizer_step=1000,
        extra_meta={"checkpoint_interval": 7, "tag": "boundary"},
    )

    assert saved is not None
    assert saved.optimizer_step == 1000
    payload = torch.load(saved.path, map_location="cpu", weights_only=False)
    assert payload["meta"]["checkpoint_interval"] == 1000
    assert payload["meta"]["tag"] == "boundary"


def test_checkpoint_payload_includes_optional_runtime_state(tmp_path):
    runtime = _StatefulRuntime()
    manager = TopKCheckpointManager(tmp_path, save_interval_steps=1)

    saved = manager.maybe_save(
        loss=0.5, runtime=runtime, optimizer_step=1
    )

    payload = torch.load(saved.path, map_location="cpu", weights_only=False)
    assert payload["runtime_state"]["counter"] == 17
    assert torch.equal(
        payload["runtime_state"]["buffer"], torch.tensor([1, 2, 3])
    )


def test_forced_final_save_writes_non_boundary_without_duplicate_rewrite(tmp_path):
    runtime = _Runtime()
    manager = TopKCheckpointManager(
        tmp_path,
        top_k=3,
        save_interval_steps=1000,
    )

    forced = manager.maybe_save(
        loss=0.8,
        runtime=runtime,
        optimizer_step=1500,
        force=True,
    )
    assert forced is not None
    assert forced.optimizer_step == 1500
    initial_mtime_ns = Path(forced.path).stat().st_mtime_ns

    duplicate = manager.maybe_save(
        loss=0.1,
        runtime=runtime,
        optimizer_step=1500,
        force=True,
    )

    assert duplicate == forced
    assert Path(forced.path).stat().st_mtime_ns == initial_mtime_ns
    assert len(manager.snapshot()) == 1


def test_periodic_manager_retains_latest_k_saved_steps(tmp_path):
    runtime = _Runtime()
    manager = TopKCheckpointManager(
        tmp_path,
        top_k=2,
        save_interval_steps=1000,
    )

    first = manager.maybe_save(loss=0.9, runtime=runtime, optimizer_step=1000)
    second = manager.maybe_save(loss=0.8, runtime=runtime, optimizer_step=2000)
    final = manager.maybe_save(
        loss=0.7,
        runtime=runtime,
        optimizer_step=2500,
        force=True,
    )

    assert first is not None and not Path(first.path).exists()
    assert second is not None and Path(second.path).exists()
    assert final is not None and Path(final.path).exists()
    assert [item["optimizer_step"] for item in manager.snapshot()] == [2000, 2500]
    assert not (tmp_path / "ckpt_latest.pt").exists()


def test_validation_accuracy_manager_retains_best_k(tmp_path):
    runtime = _Runtime()
    evaluator = _Evaluator({1000: 0.60, 2000: 0.80, 3000: 0.70})
    manager = TopKCheckpointManager(
        tmp_path,
        top_k=2,
        save_interval_steps=1000,
        selection_mode="validation_accuracy",
        validation_evaluator=evaluator,
    )

    first = manager.maybe_save(loss=0.9, runtime=runtime, optimizer_step=1000)
    second = manager.maybe_save(loss=0.8, runtime=runtime, optimizer_step=2000)
    third = manager.maybe_save(loss=0.7, runtime=runtime, optimizer_step=3000)

    assert first is not None and not Path(first.path).exists()
    assert second is not None and Path(second.path).exists()
    assert third is not None and Path(third.path).exists()
    assert [item["optimizer_step"] for item in manager.snapshot()] == [2000, 3000]
    assert [item["val_overall_accuracy"] for item in manager.snapshot()] == [0.8, 0.7]
    history = manager.checkpoint_history()
    assert [item["optimizer_step"] for item in history] == [1000, 2000, 3000]
    assert [item["training_loss"] for item in history] == pytest.approx([0.9, 0.8, 0.7])
    assert [item["val_overall_accuracy"] for item in history] == pytest.approx(
        [0.6, 0.8, 0.7]
    )
    assert all(item["checkpoint_elapsed_s"] >= 0.0 for item in history)
    assert (tmp_path / "ckpt_latest.pt").exists()
    latest_payload = torch.load(tmp_path / "ckpt_latest.pt", map_location="cpu", weights_only=False)
    assert latest_payload["optimizer_step"] == 3000
    assert latest_payload["val_overall_accuracy"] == pytest.approx(0.7)


def test_validation_accuracy_manager_prefers_later_step_on_tie(tmp_path):
    runtime = _Runtime()
    evaluator = _Evaluator({1000: 0.75, 2000: 0.75, 3000: 0.74})
    manager = TopKCheckpointManager(
        tmp_path,
        top_k=1,
        save_interval_steps=1000,
        selection_mode="validation_accuracy",
        validation_evaluator=evaluator,
    )

    manager.maybe_save(loss=0.9, runtime=runtime, optimizer_step=1000)
    saved = manager.maybe_save(loss=0.8, runtime=runtime, optimizer_step=2000)
    skipped = manager.maybe_save(loss=0.7, runtime=runtime, optimizer_step=3000)

    assert saved is not None
    assert skipped is None
    snapshot = manager.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0]["optimizer_step"] == 2000
    assert snapshot[0]["val_overall_accuracy"] == pytest.approx(0.75)


def test_validation_accuracy_requires_evaluator(tmp_path):
    with pytest.raises(ValueError, match="validation_evaluator is required"):
        TopKCheckpointManager(
            tmp_path,
            top_k=1,
            save_interval_steps=1000,
            selection_mode="validation_accuracy",
        )


def test_periodic_manager_rejects_non_positive_interval(tmp_path):
    with pytest.raises(ValueError, match="save_interval_steps must be positive"):
        TopKCheckpointManager(tmp_path, save_interval_steps=0)


def test_disabled_periodic_manager_never_saves_even_when_forced():
    runtime = _Runtime()
    manager = TopKCheckpointManager(
        None,
        top_k=2,
        save_interval_steps=1000,
    )

    assert (
        manager.maybe_save(
            loss=0.5,
            runtime=runtime,
            optimizer_step=1500,
            force=True,
        )
        is None
    )
    assert manager.snapshot() == []
