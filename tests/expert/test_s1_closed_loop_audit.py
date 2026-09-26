from __future__ import annotations

import numpy as np

from experiments.runners.run_s1_closed_loop_audit import formal_rows, pilot_rows
from mapf_cuda.evaluation.closed_loop import summarize_closed_loop


def test_s1_pilot_is_paired_across_modes():
    rows = pilot_rows()

    assert len(rows) == 18
    cells = {}
    for row in rows:
        cells.setdefault((row.num_agents, row.topology), []).append(row)
    assert len(cells) == 6
    assert all({row.mode for row in cell} == {
        "strong_online", "compact_sync", "proposed_async"
    } for cell in cells.values())
    assert all({row.seed for row in cell} == {0} for cell in cells.values())


def test_s1_formal_matrix_has_three_modes_three_seeds_and_nine_cells():
    rows = formal_rows()

    assert len(rows) == 81
    cells = {}
    for row in rows:
        cells.setdefault((row.num_agents, row.topology), []).append(row)
    assert len(cells) == 9
    for cell in cells.values():
        assert {row.mode for row in cell} == {
            "strong_online", "compact_sync", "proposed_async"
        }
        assert {row.seed for row in cell} == {0, 1, 2}
        assert len(cell) == 9


def test_closed_loop_summary_separates_initial_arrivals_from_progress():
    result = summarize_closed_loop(
        instance_ids=np.asarray([10, 11], dtype=np.int64),
        num_agents=2,
        horizon=4,
        initial_arrived=np.asarray([[True, False], [False, False]]),
        final_arrived=np.asarray([[True, True], [True, False]]),
        arrival_times=np.asarray([[0, 2], [4, -1]], dtype=np.int32),
        arrival_counts=np.asarray(
            [[1, 0], [1, 0], [2, 0], [2, 0], [2, 1]], dtype=np.int32
        ),
        completion_steps=np.asarray([2, -1], dtype=np.int32),
        last_arrival_steps=np.asarray([2, 4], dtype=np.int32),
        last_motion_steps=np.asarray([2, 4], dtype=np.int32),
        requested_moves=np.asarray([3, 5], dtype=np.int64),
        blocked_moves=np.asarray([1, 2], dtype=np.int64),
    )

    assert result["initial_individual_success_rate"] == 0.25
    assert result["individual_success_rate"] == 0.75
    assert np.isclose(result["normalized_arrival_gain"], 2 / 3)
    assert result["arrival_auc"] == 0.5
    assert result["incremental_arrival_auc"] == 0.4375
    assert result["complete_success_rate"] == 0.5
    assert result["newly_arrived_step_median"] == 3.0
    assert result["blocked_requested_move_rate"] == 3 / 8


def test_closed_loop_summary_names_tail_stall_without_calling_it_deadlock():
    result = summarize_closed_loop(
        instance_ids=np.asarray([3], dtype=np.int64),
        num_agents=1,
        horizon=20,
        initial_arrived=np.asarray([[False]]),
        final_arrived=np.asarray([[False]]),
        arrival_times=np.asarray([[-1]], dtype=np.int32),
        arrival_counts=np.zeros((21, 1), dtype=np.int32),
        completion_steps=np.asarray([-1], dtype=np.int32),
        last_arrival_steps=np.asarray([0], dtype=np.int32),
        last_motion_steps=np.asarray([2], dtype=np.int32),
        requested_moves=np.asarray([10], dtype=np.int64),
        blocked_moves=np.asarray([8], dtype=np.int64),
    )

    assert result["no_arrival_in_final_16_rate"] == 1.0
    assert result["no_motion_in_final_16_rate"] == 1.0
    assert "deadlock" not in result
