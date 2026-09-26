from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("producers", "expected"),
    [
        (1, ((0, 1, 2, 3, 4, 5, 6, 7),)),
        (2, ((0, 2, 4, 6), (1, 3, 5, 7))),
        (4, ((0, 4), (1, 5), (2, 6), (3, 7))),
        (8, ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))),
    ],
)
def test_partition_eight_logical_experts_without_changing_the_frontier(
    producers, expected
):
    from mapf_cuda.training.topology_async import _partition_expert_assignments

    groups = _partition_expert_assignments(8, producers)

    assert groups == expected
    assert sorted(value for group in groups for value in group) == list(range(8))


@pytest.mark.parametrize("producers", [0, 9])
def test_partition_rejects_invalid_producer_counts(producers):
    from mapf_cuda.training.topology_async import _partition_expert_assignments

    with pytest.raises(ValueError, match="num_producer_processes"):
        _partition_expert_assignments(8, producers)
