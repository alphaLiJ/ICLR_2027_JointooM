import numpy as np
import pytest
import torch


class _PredictingModel(torch.nn.Module):
    def act(self, tokens, **kwargs):
        return tokens[:, 0].to(torch.int64)


class _Runtime:
    def __init__(self):
        self.model = _PredictingModel()
        self.model.train()


def test_frozen_validation_round_trip_and_accuracy(tmp_path):
    from expert.mapf_gpt_validation import (
        FrozenMapfGPTValidationEvaluator,
        save_frozen_mapf_gpt_validation,
    )

    tokens = np.zeros((4, 256), dtype=np.int32)
    tokens[:, 0] = [0, 1, 3, 4]
    labels = np.asarray([0, 1, 2, 0], dtype=np.int64)
    path = save_frozen_mapf_gpt_validation(
        tmp_path / "validation.npz",
        tokens=tokens,
        labels=labels,
        metadata={"map_name": "test-maze", "seed": 7},
    )
    runtime = _Runtime()
    evaluator = FrozenMapfGPTValidationEvaluator(
        [path], device="cpu", batch_size=2
    )

    metrics = evaluator(runtime, optimizer_step=1000)

    assert metrics["optimizer_step"] == 1000
    assert metrics["val_overall_accuracy"] == pytest.approx(0.5)
    assert metrics["val_nonstay_accuracy"] == pytest.approx(0.5)
    assert metrics["validation_num_trajectories"] == 1
    assert len(metrics["validation_trajectory_sha256s"][0]) == 64
    assert runtime.model.training is True


@pytest.mark.parametrize(
    "tokens, labels, message",
    [
        (np.zeros((2, 255), dtype=np.int32), np.zeros(2, dtype=np.int64), "256"),
        (np.zeros((2, 256), dtype=np.int64), np.zeros(2, dtype=np.int64), "int32"),
        (np.zeros((2, 256), dtype=np.int32), np.asarray([0, 5]), r"\[0, 4\]"),
    ],
)
def test_frozen_validation_rejects_invalid_schema(tmp_path, tokens, labels, message):
    from expert.mapf_gpt_validation import save_frozen_mapf_gpt_validation

    with pytest.raises(ValueError, match=message):
        save_frozen_mapf_gpt_validation(
            tmp_path / "invalid.npz", tokens=tokens, labels=labels
        )
