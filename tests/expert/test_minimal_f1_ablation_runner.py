from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _FakeGrid:
    def __init__(self, positions):
        self._positions = positions

    def get_agents_xy(self, *, ignore_borders):
        assert ignore_borders is True
        return self._positions


def test_fill_compact_rows_uses_env_major_fixed_width_layout():
    from expert.minimal_f1_ablation_runner import _fill_compact_rows

    episodes = SimpleNamespace(
        batch=SimpleNamespace(
            num_envs=2,
            num_agents=2,
            goals=np.asarray(
                [[[10, 11], [12, 13]], [[20, 21], [22, 23]]],
                dtype=np.uint16,
            ),
        ),
        envs=[
            SimpleNamespace(grid=_FakeGrid([[1, 2], [3, 4]])),
            SimpleNamespace(grid=_FakeGrid([[5, 6], [7, 8]])),
        ],
    )
    rows = np.empty((4, 8), dtype=np.int16)

    _fill_compact_rows(episodes, rows, refresh=True)

    assert rows.tolist() == [
        [0, 0, 1, 2, 10, 11, 0, 1],
        [0, 1, 3, 4, 12, 13, 0, 1],
        [1, 0, 5, 6, 20, 21, 0, 1],
        [1, 1, 7, 8, 22, 23, 0, 1],
    ]


def test_fill_compact_rows_can_disable_refresh_flags():
    from expert.minimal_f1_ablation_runner import _fill_compact_rows

    episodes = SimpleNamespace(
        batch=SimpleNamespace(
            num_envs=1,
            num_agents=1,
            goals=np.asarray([[[9, 9]]], dtype=np.uint16),
        ),
        envs=[SimpleNamespace(grid=_FakeGrid([[4, 5]]))],
    )
    rows = np.empty((1, 8), dtype=np.int16)

    _fill_compact_rows(episodes, rows, refresh=False)

    assert rows[0].tolist() == [0, 0, 4, 5, 9, 9, 0, 0]
