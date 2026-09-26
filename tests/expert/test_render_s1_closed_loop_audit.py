from __future__ import annotations

import pytest

from experiments.renderers.render_s1_closed_loop_audit import paired_seed_summary


def _row(mode: str, seed: int, agents: int, value: float):
    return {
        "mode": mode,
        "seed": seed,
        "num_agents": agents,
        "topology": "random",
        "normalized_arrival_gain": value,
    }


def test_paired_summary_clusters_cells_by_training_seed():
    rows = [
        _row("left", 0, 32, 0.5),
        _row("right", 0, 32, 0.2),
        _row("left", 0, 64, 0.7),
        _row("right", 0, 64, 0.4),
        _row("left", 1, 32, 0.1),
        _row("right", 1, 32, 0.2),
        _row("left", 1, 64, 0.2),
        _row("right", 1, 64, 0.3),
    ]

    result = paired_seed_summary(
        rows,
        metric="normalized_arrival_gain",
        left_mode="left",
        right_mode="right",
    )

    assert result["seed_count"] == 2
    assert [row["mean_paired_difference"] for row in result["per_seed"]] == pytest.approx(
        [0.3, -0.1]
    )
    assert result["mean_paired_difference"] == pytest.approx(0.1)
    assert result["all_seed_means_positive"] is False
