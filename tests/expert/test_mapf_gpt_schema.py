import numpy as np
import pytest


def _observations(positions, goals):
    return [
        {"global_xy": tuple(pos), "global_target_xy": tuple(goal)}
        for pos, goal in zip(positions, goals)
    ]


def test_action_history_snapshots_previous_five_actions_and_resets():
    from expert.mapf_gpt_schema import ActionHistory, MAPFGPT_EMPTY_ACTION

    history = ActionHistory(num_agents=1)
    assert history.snapshot().tolist() == [[MAPFGPT_EMPTY_ACTION] * 5]

    snapshots = []
    for action in [0, 1, 2, 3, 4, 0]:
        snapshots.append(history.snapshot()[0].tolist())
        history.append(np.asarray([action], dtype=np.uint16))

    assert snapshots[1] == [5, 5, 5, 5, 0]
    assert snapshots[5] == [0, 1, 2, 3, 4]
    assert history.snapshot()[0].tolist() == [1, 2, 3, 4, 0]

    history.reset()
    assert history.snapshot().tolist() == [[5, 5, 5, 5, 5]]


def test_build_rows_uses_history_before_current_action_append():
    from expert.mapf_gpt_schema import ActionHistory, build_mapf_gpt_rows

    history = ActionHistory(num_agents=2)
    history.append(np.asarray([1, 2], dtype=np.uint16))
    actions = np.asarray([3, 4], dtype=np.uint16)
    rows = build_mapf_gpt_rows(
        _observations([(5, 6), (7, 8)], [(9, 10), (11, 12)]),
        actions,
        env_id=2,
        refresh_flags=np.asarray([1, 0], dtype=np.uint16),
        history=history.snapshot(),
        coordinate_offset=0,
    )

    assert rows.shape == (2, 13)
    assert rows.dtype == np.uint16
    assert rows[:, :8].tolist() == [
        [2, 0, 5, 6, 9, 10, 3, 1],
        [2, 1, 7, 8, 11, 12, 4, 0],
    ]
    assert rows[0, 8:].tolist() == [5, 5, 5, 5, 1]
    assert rows[1, 8:].tolist() == [5, 5, 5, 5, 2]
    assert 3 not in rows[0, 8:]
    assert 4 not in rows[1, 8:]


def test_validate_rows_checks_schema_ranges_and_stage_order():
    from expert.mapf_gpt_schema import validate_mapf_gpt_rows

    rows = np.zeros((4, 13), dtype=np.uint16)
    rows[:, 0] = [0, 0, 1, 1]
    rows[:, 1] = [0, 1, 0, 1]
    rows[:, 6] = [0, 1, 2, 3]
    rows[:, 8:13] = 5
    assert validate_mapf_gpt_rows(rows, num_envs=2, agents_per_env=2)

    bad = rows.copy()
    bad[0, 6] = 5
    with pytest.raises(ValueError, match="current action"):
        validate_mapf_gpt_rows(bad, num_envs=2, agents_per_env=2)

    bad = rows.copy()
    bad[0, 8] = 6
    with pytest.raises(ValueError, match="history"):
        validate_mapf_gpt_rows(bad, num_envs=2, agents_per_env=2)

    bad = rows[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="env-major"):
        validate_mapf_gpt_rows(bad, num_envs=2, agents_per_env=2)


def test_reference_tokenizer_emits_official_vocabulary_and_neighbor_histories():
    from expert.mapf_gpt_schema import tokenize_stage_reference

    grid = np.zeros((11, 11), dtype=np.uint8)
    rows = np.zeros((3, 13), dtype=np.uint16)
    rows[:, 0] = 0
    rows[:, 1] = [0, 1, 2]
    rows[:, 2:4] = [[5, 5], [5, 6], [6, 5]]
    rows[:, 4:6] = [[5, 7], [5, 4], [4, 5]]
    rows[:, 6] = [0, 1, 2]
    rows[:, 7] = 1
    rows[:, 8:13] = [
        [5, 5, 5, 5, 0],
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
    ]

    tokens, labels = tokenize_stage_reference(
        np.asarray([grid]), rows, num_agents=3
    )
    assert tokens.shape == (3, 256)
    assert labels.tolist() == [0, 1, 2]
    assert np.all((tokens >= 0) & (tokens <= 66))

    # Cost patch comes first, followed by ego then equal-distance IDs 1 and 2.
    assert tokens[0, 121:131].tolist()[4:9] == [44, 44, 44, 44, 45]
    assert tokens[0, 131:141].tolist()[4:9] == [45, 46, 47, 48, 49]
    assert tokens[0, 141:151].tolist()[4:9] == [49, 48, 47, 46, 45]
    assert np.all(tokens[0, 151:251] == 66)
    assert np.all(tokens[0, 251:] == 66)


def test_reference_tokenizer_cost_tokens_cover_unreachable_and_clip_sentinels():
    from expert.mapf_gpt_schema import tokenize_stage_reference

    # Spatially adjacent cells on opposite sides of this wall require a long
    # detour through x=0. This makes local cost differences exceed +/-20.
    grid = np.zeros((31, 40), dtype=np.uint8)
    grid[1:, 16] = 1
    rows = np.zeros((2, 13), dtype=np.uint16)
    rows[0, :8] = [0, 0, 15, 17, 15, 39, 0, 1]
    rows[1, :8] = [0, 1, 15, 15, 15, 17, 0, 1]
    rows[:, 8:13] = 5

    tokens, _ = tokenize_stage_reference(np.asarray([grid]), rows, num_agents=2)
    assert 41 in tokens[0, :121]  # obstacle => unreachable / -80
    assert 43 in tokens[0, :121]  # difference above +20 => +40
    assert 42 in tokens[1, :121]  # difference below -20 => -40
    assert tokens[0, 121 + 2] == 20  # relative goal dx == 0
    assert tokens[0, 121 + 3] == 40  # relative goal dy clips to +20


def test_reference_tokenizer_caps_visible_agents_by_manhattan_then_id():
    from expert.mapf_gpt_schema import tokenize_stage_reference

    grid = np.zeros((20, 20), dtype=np.uint8)
    positions = [(10, 10)]
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if (dx, dy) != (0, 0):
                positions.append((10 + dx, 10 + dy))
    positions = positions[:16]
    rows = np.zeros((len(positions), 13), dtype=np.uint16)
    rows[:, 0] = 0
    rows[:, 1] = np.arange(len(positions), dtype=np.uint16)
    rows[:, 2:4] = positions
    rows[:, 4:6] = positions
    rows[:, 6] = 0
    rows[:, 7] = 1
    rows[:, 8:13] = 5

    tokens, _ = tokenize_stage_reference(
        np.asarray([grid]), rows, num_agents=len(positions)
    )
    records = tokens[0, 121:251].reshape(13, 10)
    selected_relative_positions = [tuple((record[:2] - 20).tolist()) for record in records]
    expected_ids = sorted(
        range(len(positions)),
        key=lambda idx: (
            abs(positions[idx][0] - 10) + abs(positions[idx][1] - 10),
            idx,
        ),
    )[:13]
    expected_relative_positions = [
        (positions[idx][0] - 10, positions[idx][1] - 10)
        for idx in expected_ids
    ]
    assert selected_relative_positions == expected_relative_positions
