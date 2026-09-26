from collections import deque

import jax
import jax.numpy as jnp
import pytest
from baselines import jax_sim_baseline as baseline

from baselines.jax_sim_baseline import (
    EnvConfig,
    State,
    batched_step_env,
    block_transition_ready,
    build_aligned_observations,
    build_batched_aligned_observations,
    reset_frozen_jit,
    run_benchmark,
    step_transition_jit,
    step_env,
)


def _bfs_distance(grid, start, goal):
    h, w = grid.shape
    sx, sy = start
    gx, gy = goal
    q = deque([(gx, gy, 0)])
    seen = {(gx, gy)}
    while q:
        x, y, d = q.popleft()
        if (x, y) == (sx, sy):
            return d
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and int(grid[nx, ny]) == 0 and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return None


def _frozen_state(
    config,
    positions,
    goals,
    *,
    arrived=None,
    grids=None,
    seed=0,
):
    positions = jnp.asarray(positions, dtype=jnp.int32)
    goals = jnp.asarray(goals, dtype=jnp.int32)
    batch_size = positions.shape[0]
    if arrived is None:
        arrived = jnp.zeros(positions.shape[:2], dtype=bool)
    if grids is None:
        grids = jnp.zeros(
            (batch_size, config.height, config.width), dtype=jnp.uint8
        )
    rng_keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
    state = reset_frozen_jit(
        config,
        positions,
        goals,
        jnp.asarray(arrived, dtype=bool),
        grids,
        rng_keys,
    )
    return state, grids, rng_keys


def test_unknown_task_mode_is_rejected_before_jit():
    config = EnvConfig(
        height=4,
        width=4,
        num_agents=1,
        task_mode="not-a-mapf-mode",
    )
    with pytest.raises(ValueError, match="Unknown task mode"):
        _frozen_state(config, [[[1, 1]]], [[[2, 1]]])


def test_frozen_reset_preserves_canonical_state_and_explicit_rng_keys():
    config = EnvConfig(
        height=4,
        width=5,
        num_agents=2,
        task_mode="standard_mapf",
        max_episode_steps=8,
    )
    positions = jnp.array([[[1, 1], [3, 3]], [[2, 1], [1, 4]]], dtype=jnp.int32)
    goals = jnp.array([[[1, 1], [3, 4]], [[2, 2], [1, 4]]], dtype=jnp.int32)
    arrived = jnp.array([[True, False], [False, True]])
    grids = jnp.zeros((2, 4, 5), dtype=jnp.uint8).at[0, 0, 4].set(1)
    rng_keys = jax.random.split(jax.random.PRNGKey(29), 2)

    state = reset_frozen_jit(config, positions, goals, arrived, grids, rng_keys)
    block_transition_ready(state)

    assert jnp.array_equal(state.pos, positions)
    assert jnp.array_equal(state.target, goals)
    assert jnp.array_equal(state.arrived, arrived)
    assert jnp.array_equal(state.rng_key, rng_keys)
    assert jnp.array_equal(state.step_counts, jnp.zeros((2,), dtype=jnp.int32))
    assert jnp.array_equal(state.terminated, jnp.array([False, False]))
    assert jnp.array_equal(state.truncated, jnp.array([False, False]))
    assert jnp.array_equal(grids, jnp.zeros((2, 4, 5), dtype=jnp.uint8).at[0, 0, 4].set(1))


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (0, (2, 2)),
        (1, (1, 2)),
        (2, (3, 2)),
        (3, (2, 1)),
        (4, (2, 3)),
    ],
)
def test_transition_action_mapping_matches_canonical_contract(action, expected):
    config = EnvConfig(
        height=5,
        width=5,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=8,
    )
    state, grids, _ = _frozen_state(config, [[[2, 2]]], [[[4, 4]]])

    new_state, _, _, _, _ = step_transition_jit(
        config, state, jnp.array([[action]], dtype=jnp.uint8), grids
    )
    block_transition_ready(new_state)

    assert tuple(map(int, new_state.pos[0, 0])) == expected


def test_standard_mode_keeps_goals_and_previously_arrived_agents_stationary():
    config = EnvConfig(
        height=5,
        width=5,
        num_agents=2,
        task_mode="standard_mapf",
        max_episode_steps=8,
    )
    goals = jnp.array([[[1, 1], [4, 4]]], dtype=jnp.int32)
    state, grids, _ = _frozen_state(
        config,
        [[[1, 1], [3, 3]]],
        goals,
        arrived=[[True, False]],
    )

    new_state, reward, terminated, truncated, info = step_transition_jit(
        config, state, jnp.array([[2, 4]], dtype=jnp.uint8), grids
    )
    block_transition_ready(new_state)

    assert jnp.array_equal(new_state.target, goals)
    assert tuple(map(int, new_state.pos[0, 0])) == (1, 1)
    assert tuple(map(int, new_state.pos[0, 1])) == (3, 4)
    assert jnp.array_equal(new_state.arrived, jnp.array([[True, False]]))
    assert jnp.array_equal(reward, jnp.array([[1.0, 0.0]]))
    assert jnp.array_equal(terminated, jnp.array([False]))
    assert jnp.array_equal(truncated, jnp.array([False]))
    assert jnp.array_equal(info["goal_changed_flags"], jnp.zeros((1, 2), dtype=jnp.int32))


def test_standard_final_arrival_terminates_on_same_transition_and_beats_horizon():
    config = EnvConfig(
        height=4,
        width=4,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=1,
    )
    state, grids, _ = _frozen_state(config, [[[1, 1]]], [[[2, 1]]])

    new_state, reward, terminated, truncated, _ = step_transition_jit(
        config, state, jnp.array([[2]], dtype=jnp.uint8), grids
    )
    block_transition_ready(new_state)

    assert tuple(map(int, new_state.pos[0, 0])) == (2, 1)
    assert jnp.array_equal(new_state.arrived, jnp.array([[True]]))
    assert jnp.array_equal(new_state.step_counts, jnp.array([1], dtype=jnp.int32))
    assert jnp.array_equal(reward, jnp.array([[1.0]]))
    assert jnp.array_equal(terminated, jnp.array([True]))
    assert jnp.array_equal(truncated, jnp.array([False]))


def test_standard_horizon_truncates_only_unfinished_environment():
    config = EnvConfig(
        height=4,
        width=4,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=1,
    )
    state, grids, _ = _frozen_state(
        config,
        [[[1, 1]], [[1, 1]]],
        [[[2, 1]], [[3, 3]]],
    )

    new_state, _, terminated, truncated, _ = step_transition_jit(
        config, state, jnp.array([[2], [0]], dtype=jnp.uint8), grids
    )
    block_transition_ready(new_state)

    assert jnp.array_equal(new_state.step_counts, jnp.array([1, 1], dtype=jnp.int32))
    assert jnp.array_equal(terminated, jnp.array([True, False]))
    assert jnp.array_equal(truncated, jnp.array([False, True]))


@pytest.mark.parametrize("done_kind", ["terminated", "truncated"])
def test_standard_done_environment_does_not_advance(done_kind):
    config = EnvConfig(
        height=4,
        width=4,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=1 if done_kind == "truncated" else 8,
    )
    goal = [[[2, 1]]] if done_kind == "terminated" else [[[3, 3]]]
    action = [[2]] if done_kind == "terminated" else [[0]]
    state, grids, _ = _frozen_state(config, [[[1, 1]]], goal)
    done_state, _, _, _, _ = step_transition_jit(
        config, state, jnp.asarray(action, dtype=jnp.uint8), grids
    )
    block_transition_ready(done_state)
    before = jax.tree.map(lambda x: jnp.array(x), done_state)

    after, reward, terminated, truncated, _ = step_transition_jit(
        config, done_state, jnp.array([[4]], dtype=jnp.uint8), grids
    )
    block_transition_ready(after)

    for actual, expected in zip(jax.tree.leaves(after), jax.tree.leaves(before)):
        assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(reward, jnp.zeros((1, 1), dtype=jnp.float32))
    assert bool(terminated[0]) is (done_kind == "terminated")
    assert bool(truncated[0]) is (done_kind == "truncated")


def test_standard_frozen_reset_is_repeatable():
    config = EnvConfig(
        height=4,
        width=4,
        num_agents=1,
        task_mode="standard_mapf",
    )
    args = (
        jnp.array([[[1, 1]]], dtype=jnp.int32),
        jnp.array([[[2, 1]]], dtype=jnp.int32),
        jnp.array([[False]]),
        jnp.zeros((1, 4, 4), dtype=jnp.uint8),
        jax.random.split(jax.random.PRNGKey(4), 1),
    )

    first = reset_frozen_jit(config, *args)
    second = reset_frozen_jit(config, *args)
    block_transition_ready(first)
    block_transition_ready(second)

    for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
        assert jnp.array_equal(left, right)


def test_canonical_uint16_positions_stay_uint16_without_transition_recompile():
    jax.clear_caches()
    config = EnvConfig(
        height=6,
        width=6,
        num_agents=1,
        task_mode="standard_mapf",
        max_episode_steps=8,
    )
    grids = jnp.zeros((1, 6, 6), dtype=jnp.uint8)
    state = reset_frozen_jit(
        config,
        jnp.array([[[2, 1]]], dtype=jnp.uint16),
        jnp.array([[[5, 5]]], dtype=jnp.uint16),
        jnp.array([[False]]),
        grids,
        jax.random.split(jax.random.PRNGKey(41), 1),
    )
    actions = jnp.array([[4]], dtype=jnp.uint8)
    cache_sizes = []

    for _ in range(3):
        state, _, _, _, _ = step_transition_jit(config, state, actions, grids)
        block_transition_ready(state)
        assert state.pos.dtype == jnp.uint16
        cache_sizes.append(baseline._step_transition_compiled._cache_size())

    assert tuple(map(int, state.pos[0, 0])) == (2, 4)
    assert cache_sizes == [1, 1, 1]


def test_observation_shape_and_four_channels():
    config = EnvConfig(height=128, width=128, num_agents=2, fov_radius=5)
    grid = jnp.zeros((config.height, config.width), dtype=jnp.int32)
    state = State(
        pos=jnp.array([[64, 64], [70, 70]], dtype=jnp.int32),
        target=jnp.array([[67, 64], [75, 75]], dtype=jnp.int32),
        step_count=jnp.int32(0),
        rng_key=jax.random.PRNGKey(0),
    )

    obs = build_aligned_observations(config, state, grid)
    assert obs.shape == (2, 4, 13, 13)


def test_goal_projection_matches_cuda_style_in_and_out_of_fov():
    config = EnvConfig(height=128, width=128, num_agents=2, fov_radius=5)
    grid = jnp.zeros((config.height, config.width), dtype=jnp.int32)
    state = State(
        pos=jnp.array([[64, 64], [64, 64]], dtype=jnp.int32),
        target=jnp.array([[67, 64], [100, 100]], dtype=jnp.int32),
        step_count=jnp.int32(0),
        rng_key=jax.random.PRNGKey(0),
    )

    obs = build_aligned_observations(config, state, grid)

    assert float(obs[0, 2, 9, 6]) == 1.0
    assert float(obs[1, 2].sum()) == 1.0


def test_wall_and_occupancy_channels_match_local_semantics():
    config = EnvConfig(height=128, width=128, num_agents=2, fov_radius=5)
    grid = jnp.zeros((config.height, config.width), dtype=jnp.int32)
    grid = grid.at[64, 65].set(1)
    state = State(
        pos=jnp.array([[64, 64], [64, 66]], dtype=jnp.int32),
        target=jnp.array([[80, 80], [90, 90]], dtype=jnp.int32),
        step_count=jnp.int32(0),
        rng_key=jax.random.PRNGKey(0),
    )

    obs = build_aligned_observations(config, state, grid)

    assert float(obs[0, 0, 6, 7]) == 1.0
    assert float(obs[0, 1, 6, 6]) == 1.0
    assert float(obs[0, 1, 6, 8]) == 1.0


def test_step_reassigns_goal_after_reaching_target_with_unique_free_cells():
    config = EnvConfig(height=4, width=4, num_agents=2, fov_radius=1, max_steps=10)
    grid = jnp.zeros((config.height, config.width), dtype=jnp.int32)
    state = State(
        pos=jnp.array([[1, 1], [2, 2]], dtype=jnp.int32),
        target=jnp.array([[1, 2], [3, 3]], dtype=jnp.int32),
        step_count=jnp.int32(0),
        rng_key=jax.random.PRNGKey(7),
    )

    _, new_state, reward, _, info = step_env(
        config,
        state,
        jnp.array([4, 0], dtype=jnp.int32),
        grid,
    )

    assert float(reward[0]) == 1.0
    assert int(info["goal_changed_flags"][0]) == 1
    assert int(info["goal_changed_flags"][1]) == 0
    assert tuple(map(int, new_state.target[0])) != (1, 2)
    assert tuple(map(int, new_state.target[1])) == (3, 3)
    assert tuple(map(int, new_state.target[0])) != tuple(map(int, new_state.target[1]))
    assert int(grid[tuple(map(int, new_state.target[0]))]) == 0


def test_ctg_channel_uses_obstacle_aware_distance_not_manhattan():
    config = EnvConfig(height=7, width=7, num_agents=1, fov_radius=2, clamp_value=1.0)
    grid = jnp.zeros((config.height, config.width), dtype=jnp.int32)
    wall_cells = [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)]
    for x, y in wall_cells:
        grid = grid.at[x, y].set(1)

    state = State(
        pos=jnp.array([[3, 1]], dtype=jnp.int32),
        target=jnp.array([[3, 5]], dtype=jnp.int32),
        step_count=jnp.int32(0),
        rng_key=jax.random.PRNGKey(0),
    )

    obs = build_aligned_observations(config, state, grid)

    center = (3, 1)
    local_cell = (2, 1)
    goal = (3, 5)
    center_dist = _bfs_distance(grid, center, goal)
    local_dist = _bfs_distance(grid, local_cell, goal)
    assert center_dist is not None and local_dist is not None
    expected = max(-1.0, min(1.0, ((255 - center_dist) - (255 - local_dist)) / (2.0 * config.fov_radius)))

    local_idx = (local_cell[0] - center[0] + config.fov_radius + 1, local_cell[1] - center[1] + config.fov_radius + 1)
    observed = float(obs[0, 3, local_idx[0], local_idx[1]])
    assert observed == expected


def test_benchmark_returns_four_channel_observation_shape():
    metrics = run_benchmark(batch_size=2, steps_to_test=1, warmup_steps=1)
    assert metrics["obs_shape"][-3:] == (4, 13, 13)


def test_batched_observations_match_vmap_single_env_when_env_maps_differ():
    config = EnvConfig(height=8, width=8, num_agents=2, fov_radius=2)
    grids = jnp.zeros((2, config.height, config.width), dtype=jnp.int32)
    grids = grids.at[0, 1:6, 3].set(1)
    grids = grids.at[1, 4, 1:7].set(1)

    state = State(
        pos=jnp.array(
            [
                [[2, 1], [6, 6]],
                [[1, 1], [6, 5]],
            ],
            dtype=jnp.int32,
        ),
        target=jnp.array(
            [
                [[2, 6], [5, 6]],
                [[1, 6], [6, 1]],
            ],
            dtype=jnp.int32,
        ),
        step_count=jnp.array([0, 0], dtype=jnp.int32),
        rng_key=jax.random.split(jax.random.PRNGKey(11), 2),
    )

    expected = jax.vmap(lambda s, g: build_aligned_observations(config, s, g))(state, grids)
    observed = build_batched_aligned_observations(config, state, grids)
    assert jnp.allclose(observed, expected)


def test_batched_step_matches_vmap_single_env_step_when_no_goal_reset_occurs():
    config = EnvConfig(height=8, width=8, num_agents=2, fov_radius=2, max_steps=10)
    grids = jnp.zeros((2, config.height, config.width), dtype=jnp.int32)
    grids = grids.at[0, 2, 2].set(1)
    grids = grids.at[1, 5, 5].set(1)
    state = State(
        pos=jnp.array(
            [
                [[1, 1], [4, 4]],
                [[1, 2], [6, 6]],
            ],
            dtype=jnp.int32,
        ),
        target=jnp.array(
            [
                [[7, 7], [0, 7]],
                [[7, 0], [0, 0]],
            ],
            dtype=jnp.int32,
        ),
        step_count=jnp.array([0, 0], dtype=jnp.int32),
        rng_key=jax.random.split(jax.random.PRNGKey(23), 2),
    )
    actions = jnp.array([[4, 3], [2, 1]], dtype=jnp.int32)

    expected = jax.vmap(lambda s, a, g: step_env(config, s, a, g))(state, actions, grids)
    observed = batched_step_env(config, state, actions, grids)

    exp_obs, exp_state, exp_reward, exp_done, exp_info = expected
    obs, new_state, reward, done, info = observed

    assert jnp.allclose(obs, exp_obs)
    assert jnp.array_equal(new_state.pos, exp_state.pos)
    assert jnp.array_equal(new_state.target, exp_state.target)
    assert jnp.array_equal(new_state.step_count, exp_state.step_count)
    assert jnp.array_equal(reward, exp_reward)
    assert jnp.array_equal(done, exp_done)
    assert jnp.array_equal(info["goal_changed_flags"], exp_info["goal_changed_flags"])
