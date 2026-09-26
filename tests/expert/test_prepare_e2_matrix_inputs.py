from __future__ import annotations

import numpy as np


def _batch(*, envs: int, agents: int = 2, seed_start: int = 100):
    from expert.benchmark_contract import FrozenTransitionBatch

    positions = np.empty((envs, agents, 2), dtype=np.uint16)
    goals = np.empty_like(positions)
    for env_id in range(envs):
        for agent_id in range(agents):
            positions[env_id, agent_id] = (2 + agent_id, 2 + env_id)
            goals[env_id, agent_id] = (6 + agent_id, 6 + env_id)
    return FrozenTransitionBatch(
        instance_ids=np.arange(seed_start, seed_start + envs, dtype=np.int64),
        grids=np.zeros((envs, 128, 128), dtype=np.uint8),
        positions=positions,
        goals=goals,
        arrived=np.zeros((envs, agents), dtype=np.bool_),
        actions=np.zeros((2, envs, agents), dtype=np.uint8),
        horizon=2,
    )


def test_extended_pool_continues_seed_range_and_preserves_prefix():
    from expert.prepare_e2_matrix_inputs import build_extended_pool

    base = _batch(envs=2)

    def builder(*, seeds, num_agents, **_kwargs):
        return _batch(
            envs=len(seeds),
            agents=num_agents,
            seed_start=seeds[0],
        )

    extended = build_extended_pool(
        base,
        env_count=4,
        pool_identity="a2-random-d020-a002",
        pool_builder=builder,
    )

    assert extended.num_envs == 4
    assert extended.instance_ids.tolist() == [100, 101, 102, 103]
    assert extended.take_envs(2).semantic_sha256 == base.semantic_sha256
