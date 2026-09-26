from __future__ import annotations

from unittest import mock

import pytest

from expert.training_baseline_engines import (
    _advance_wall_checkpoint_deadline,
    _make_runtime,
    _resolve_checkpoint_wall_interval,
    _resolve_training_wall_budget,
    _training_wall_reached,
)


def test_wall_budget_is_optional_and_must_be_positive():
    assert _resolve_training_wall_budget({}) is None
    assert _resolve_training_wall_budget({"max_training_wall_s": 2}) == 2.0
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_training_wall_budget({"max_training_wall_s": 0})


def test_wall_budget_check_uses_steady_loop_start():
    with mock.patch(
        "expert.training_baseline_engines.time.perf_counter", return_value=13.0
    ):
        assert _training_wall_reached(10.0, 2.5) is True
        assert _training_wall_reached(10.0, 4.0) is False
        assert _training_wall_reached(10.0, None) is False


def test_checkpoint_wall_interval_is_optional_and_must_be_positive():
    assert _resolve_checkpoint_wall_interval({}) is None
    assert _resolve_checkpoint_wall_interval({"checkpoint_interval_s": 300}) == 300.0
    with pytest.raises(ValueError, match="must be positive"):
        _resolve_checkpoint_wall_interval({"checkpoint_interval_s": 0})


def test_wall_checkpoint_deadline_skips_missed_intervals():
    assert _advance_wall_checkpoint_deadline(300.0, 300.0, 300.0) == 600.0
    assert _advance_wall_checkpoint_deadline(300.0, 300.0, 901.0) == 1200.0


def test_magat_baseline_runtime_receives_matched_optimizer_protocol():
    adapter = mock.Mock()
    with mock.patch("torch.manual_seed"), mock.patch(
        "torch.cuda.manual_seed_all"
    ), mock.patch(
        "expert.fixed_magat_plus_runtime.MAGATRuntimeAdapter",
        return_value=adapter,
    ) as adapter_cls:
        result = _make_runtime(
            "magat",
            device="cpu",
            seed=7,
            kwargs={
                "lr_start": 1e-3,
                "lr_end": 1e-6,
                "lr_scheduler": "cosine-annealing",
                "scheduler_total_steps": 100_000,
                "grad_clip_norm": 0.5,
                "train_on_arrived_agents": True,
            },
        )

    assert result is adapter
    adapter_cls.assert_called_once_with(
        device="cpu",
        lr=1e-3,
        lr_end=1e-6,
        lr_scheduler="cosine-annealing",
        scheduler_total_steps=100_000,
        grad_clip_norm=0.5,
        train_on_arrived_agents=True,
    )
