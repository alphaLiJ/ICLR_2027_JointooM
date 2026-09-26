from __future__ import annotations

import numpy as np
import pytest


def _state():
    return {
        "positions": np.asarray([[[2, 3]]], dtype=np.uint16),
        "arrived": np.asarray([[False]]),
        "terminated": np.asarray([False]),
        "truncated": np.asarray([False]),
        "step_counts": np.asarray([1], dtype=np.int32),
    }


def test_long_parity_accepts_identical_discrete_state():
    from expert.minimal_p1_magat_long_runner import _assert_state_equal

    expected = _state()
    actual = {key: value.copy() for key, value in expected.items()}
    _assert_state_equal(expected, actual, step=7)


def test_long_parity_reports_first_field_and_step():
    from expert.minimal_p1_magat_long_runner import _assert_state_equal

    expected = _state()
    actual = {key: value.copy() for key, value in expected.items()}
    actual["positions"][0, 0, 1] = 4

    with pytest.raises(
        RuntimeError, match=r"step 9 for positions: 1 mismatches"
    ):
        _assert_state_equal(expected, actual, step=9)
