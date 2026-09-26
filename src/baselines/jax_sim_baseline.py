import time
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EnvConfig(NamedTuple):
    height: int = 128
    width: int = 128
    num_agents: int = 256
    max_steps: int = 100
    fov_radius: int = 5
    clamp_value: float = 1.0
    energy_iters: int = 254
    task_mode: str = "lifelong"
    max_episode_steps: int = 256

    @property
    def inner_obs_diam(self) -> int:
        return 2 * self.fov_radius + 1

    @property
    def pyg_obs_diam(self) -> int:
        return 2 * self.fov_radius + 3


class State(NamedTuple):
    pos: jnp.ndarray
    target: jnp.ndarray
    step_count: jnp.int32
    rng_key: jax.Array
    arrived: jnp.ndarray | None = None
    terminated: jnp.ndarray | None = None
    truncated: jnp.ndarray | None = None

    @property
    def step_counts(self):
        """Plural alias used by the batched transition adapter contract."""

        return self.step_count


MOVES = jnp.array(
    [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ],
    dtype=jnp.int32,
)


_TASK_MODES = frozenset(("lifelong", "standard_mapf"))


def _validate_config(config: EnvConfig) -> None:
    if config.task_mode not in _TASK_MODES:
        expected = ", ".join(sorted(_TASK_MODES))
        raise ValueError(
            f"Unknown task mode {config.task_mode!r}. Expected one of: {expected}"
        )
    if config.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive")


def _state_status_arrays(state: State):
    """Materialize status arrays for legacy four-field State construction."""

    batch_shape = state.pos.shape[:-2]
    agent_shape = state.pos.shape[:-1]
    arrived = (
        jnp.zeros(agent_shape, dtype=bool)
        if state.arrived is None
        else state.arrived
    )
    terminated = (
        jnp.zeros(batch_shape, dtype=bool)
        if state.terminated is None
        else state.terminated
    )
    truncated = (
        jnp.zeros(batch_shape, dtype=bool)
        if state.truncated is None
        else state.truncated
    )
    return arrived, terminated, truncated


def _build_occupancy_grid(config: EnvConfig, pos: jnp.ndarray) -> jnp.ndarray:
    flat_pos = pos[:, 0] * config.width + pos[:, 1]
    occ = jax.ops.segment_sum(
        jnp.ones(config.num_agents, dtype=jnp.float32),
        flat_pos,
        num_segments=config.height * config.width,
    )
    return jnp.clip(occ.reshape(config.height, config.width), 0.0, 1.0)


def _compute_energy_map(config: EnvConfig, grid: jnp.ndarray, goal: jnp.ndarray) -> jnp.ndarray:
    inf = jnp.int32(255)
    free_mask = grid == 0
    gx, gy = goal[0], goal[1]
    dist = jnp.full((config.height, config.width), inf, dtype=jnp.int32)
    goal_valid = free_mask[gx, gy]
    dist = jax.lax.cond(goal_valid, lambda d: d.at[gx, gy].set(0), lambda d: d, dist)

    def body(_, cur_dist):
        up = jnp.pad(cur_dist[:-1, :], ((1, 0), (0, 0)), constant_values=inf)
        down = jnp.pad(cur_dist[1:, :], ((0, 1), (0, 0)), constant_values=inf)
        left = jnp.pad(cur_dist[:, :-1], ((0, 0), (1, 0)), constant_values=inf)
        right = jnp.pad(cur_dist[:, 1:], ((0, 0), (0, 1)), constant_values=inf)
        neigh = jnp.minimum(jnp.minimum(up, down), jnp.minimum(left, right))
        candidate = jnp.where(neigh < inf - 1, neigh + 1, inf)
        new_dist = jnp.where(free_mask, jnp.minimum(cur_dist, candidate), inf)
        new_dist = jax.lax.cond(goal_valid, lambda d: d.at[gx, gy].set(0), lambda d: d, new_dist)
        return new_dist

    dist = jax.lax.fori_loop(0, config.energy_iters, body, dist)
    return jnp.where(dist == inf, 0.0, 255.0 - dist.astype(jnp.float32))


def _build_batched_occupancy_grids(config: EnvConfig, pos: jnp.ndarray) -> jnp.ndarray:
    flat_pos = pos[..., 0] * config.width + pos[..., 1]

    def build_one(env_flat_pos: jnp.ndarray) -> jnp.ndarray:
        occ = jax.ops.segment_sum(
            jnp.ones(env_flat_pos.shape[0], dtype=jnp.float32),
            env_flat_pos,
            num_segments=config.height * config.width,
        )
        return jnp.clip(occ.reshape(config.height, config.width), 0.0, 1.0)

    return jax.vmap(build_one)(flat_pos)


def _expand_frontier(frontier: jnp.ndarray) -> jnp.ndarray:
    up = jnp.pad(frontier[:, :-1, :], ((0, 0), (1, 0), (0, 0)), constant_values=False)
    down = jnp.pad(frontier[:, 1:, :], ((0, 0), (0, 1), (0, 0)), constant_values=False)
    left = jnp.pad(frontier[:, :, :-1], ((0, 0), (0, 0), (1, 0)), constant_values=False)
    right = jnp.pad(frontier[:, :, 1:], ((0, 0), (0, 0), (0, 1)), constant_values=False)
    return up | down | left | right


def _build_batched_energy_maps(config: EnvConfig, grids: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    def open_grid_energy(_: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        coord_x = jnp.arange(config.height, dtype=jnp.int32).reshape(1, 1, config.height, 1)
        coord_y = jnp.arange(config.width, dtype=jnp.int32).reshape(1, 1, 1, config.width)
        goal_x = targets[..., 0][..., None, None]
        goal_y = targets[..., 1][..., None, None]
        dist = jnp.abs(coord_x - goal_x) + jnp.abs(coord_y - goal_y)
        return jnp.maximum(0.0, 255.0 - dist.astype(jnp.float32))

    def general_bfs_energy(_: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        batch_size, agent_count = targets.shape[:2]
        free = (grids == 0)[:, None, :, :]
        free = jnp.broadcast_to(free, (batch_size, agent_count, config.height, config.width)).reshape(
            -1, config.height, config.width
        )

        goal_x = targets[..., 0].reshape(-1)
        goal_y = targets[..., 1].reshape(-1)
        flat_count = free.shape[0]
        batch_idx = jnp.arange(flat_count, dtype=jnp.int32)

        goal_valid = free[batch_idx, goal_x, goal_y]
        frontier = jnp.zeros((flat_count, config.height, config.width), dtype=bool)
        frontier = frontier.at[batch_idx, goal_x, goal_y].set(goal_valid)
        visited = frontier
        energy = jnp.zeros((flat_count, config.height, config.width), dtype=jnp.float32)
        energy = energy.at[batch_idx, goal_x, goal_y].set(jnp.where(goal_valid, 255.0, 0.0))

        def cond_fn(val):
            step_idx, frontier_mask, _, _ = val
            return (step_idx < config.energy_iters) & jnp.any(frontier_mask)

        def body_fn(val):
            step_idx, frontier_mask, visited_mask, energy_map = val
            next_frontier = _expand_frontier(frontier_mask) & free & (~visited_mask)
            next_energy = 255.0 - jnp.float32(step_idx + 1)
            energy_map = jnp.where(next_frontier, next_energy, energy_map)
            visited_mask = visited_mask | next_frontier
            return step_idx + 1, next_frontier, visited_mask, energy_map

        _, _, _, energy = jax.lax.while_loop(cond_fn, body_fn, (jnp.int32(0), frontier, visited, energy))
        return energy.reshape(batch_size, agent_count, config.height, config.width)

    return jax.lax.cond(jnp.all(grids == 0), open_grid_energy, general_bfs_energy, operand=(grids, targets))


def _goal_projection_channel(config: EnvConfig, cx: jnp.ndarray, cy: jnp.ndarray, gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
    obs = jnp.zeros((config.pyg_obs_diam, config.pyg_obs_diam), dtype=jnp.float32)
    rel_gx = gx - cx
    rel_gy = gy - cy
    obs_r = config.fov_radius

    def in_fov_fn(_):
        idx_x = rel_gx + obs_r + 1
        idx_y = rel_gy + obs_r + 1
        return obs.at[idx_x, idx_y].set(1.0)

    def out_fov_fn(_):
        angle = jnp.arctan2(rel_gy.astype(jnp.float32), rel_gx.astype(jnp.float32))
        dist = jnp.float32(config.pyg_obs_diam // 2)
        sign_x = jnp.sign(rel_gx.astype(jnp.float32))
        sign_y = jnp.sign(rel_gy.astype(jnp.float32))
        abs_x = jnp.maximum(jnp.abs(rel_gx.astype(jnp.float32)), 1e-4)
        abs_y = jnp.maximum(jnp.abs(rel_gy.astype(jnp.float32)), 1e-4)

        vertical_sector = (
            ((angle >= jnp.pi / 4.0) & (angle <= 3.0 * jnp.pi / 4.0))
            | ((angle >= -3.0 * jnp.pi / 4.0) & (angle <= -jnp.pi / 4.0))
        )

        goal_y_fov = jnp.int32(dist * (sign_y + 1.0))
        goal_x_from_y = jnp.int32(dist) + jnp.int32(jnp.round(dist * rel_gx.astype(jnp.float32) / abs_y))

        goal_x_fov = jnp.int32(dist * (sign_x + 1.0))
        goal_y_from_x = jnp.int32(dist) + jnp.int32(jnp.round(dist * rel_gy.astype(jnp.float32) / abs_x))

        out_x = jnp.where(vertical_sector, goal_x_from_y, goal_x_fov)
        out_y = jnp.where(vertical_sector, goal_y_fov, goal_y_from_x)
        out_x = jnp.clip(out_x, 0, config.pyg_obs_diam - 1)
        out_y = jnp.clip(out_y, 0, config.pyg_obs_diam - 1)
        return obs.at[out_x, out_y].set(1.0)

    return jax.lax.cond(
        (jnp.abs(rel_gx) <= obs_r) & (jnp.abs(rel_gy) <= obs_r),
        in_fov_fn,
        out_fov_fn,
        operand=None,
    )


def _single_agent_aligned_observation(
    config: EnvConfig,
    grid: jnp.ndarray,
    occupancy_grid: jnp.ndarray,
    pos: jnp.ndarray,
    target: jnp.ndarray,
    energy_map: jnp.ndarray,
) -> jnp.ndarray:
    cx, cy = pos[0], pos[1]
    gx, gy = target[0], target[1]
    obs_r = config.fov_radius
    pyg_obs_diam = config.pyg_obs_diam

    dx = jnp.arange(-obs_r, obs_r + 1, dtype=jnp.int32)
    dy = jnp.arange(-obs_r, obs_r + 1, dtype=jnp.int32)
    tx = cx + dx[:, None]
    ty = cy + dy[None, :]

    too_far_oob = (tx < -1) | (tx > config.height) | (ty < -1) | (ty > config.width)
    virtual_wall = (tx == -1) | (tx == config.height) | (ty == -1) | (ty == config.width)
    in_valid = (tx >= 0) & (tx < config.height) & (ty >= 0) & (ty < config.width)

    clipped_tx = jnp.clip(tx, 0, config.height - 1)
    clipped_ty = jnp.clip(ty, 0, config.width - 1)
    grid_vals = grid[clipped_tx, clipped_ty]
    occ_vals = occupancy_grid[clipped_tx, clipped_ty]

    obs_val = jnp.where(
        virtual_wall,
        1.0,
        jnp.where(in_valid & (grid_vals == 1), 1.0, 0.0),
    )
    obs_val = jnp.where(too_far_oob, 0.0, obs_val)

    occ_val = jnp.where(in_valid & (grid_vals == 0), occ_vals, 0.0)

    center_ctg = energy_map[cx, cy]
    target_ctg = energy_map[clipped_tx, clipped_ty]
    ctg_val = (center_ctg - target_ctg) / (2.0 * float(obs_r))
    ctg_val = jnp.clip(ctg_val, -config.clamp_value, config.clamp_value)
    ctg_val = jnp.where(too_far_oob, config.clamp_value, ctg_val)
    ctg_val = jnp.where(virtual_wall, config.clamp_value, ctg_val)

    goal_channel = _goal_projection_channel(config, cx, cy, gx, gy)

    out = jnp.zeros((4, pyg_obs_diam, pyg_obs_diam), dtype=jnp.float32)
    out = out.at[0, 1:-1, 1:-1].set(obs_val)
    out = out.at[1, 1:-1, 1:-1].set(occ_val)
    out = out.at[2].set(goal_channel)
    out = out.at[3, 1:-1, 1:-1].set(ctg_val)
    return out


def build_aligned_observations(config: EnvConfig, state: State, grid: jnp.ndarray) -> jnp.ndarray:
    arrived, terminated, truncated = _state_status_arrays(state)
    batched_state = State(
        pos=state.pos[None, ...],
        target=state.target[None, ...],
        step_count=state.step_count[None],
        rng_key=state.rng_key[None, ...],
        arrived=arrived[None, ...],
        terminated=terminated[None, ...],
        truncated=truncated[None, ...],
    )
    return build_batched_aligned_observations(config, batched_state, grid[None, ...])[0]


def build_batched_aligned_observations(config: EnvConfig, state: State, grids: jnp.ndarray) -> jnp.ndarray:
    batch_size, agent_count = state.pos.shape[:2]
    occupancy_grids = _build_batched_occupancy_grids(config, state.pos)
    energy_maps = _build_batched_energy_maps(config, grids, state.target)

    grid_flat = jnp.broadcast_to(grids[:, None, :, :], (batch_size, agent_count, config.height, config.width)).reshape(
        -1, config.height, config.width
    )
    occ_flat = jnp.broadcast_to(
        occupancy_grids[:, None, :, :], (batch_size, agent_count, config.height, config.width)
    ).reshape(-1, config.height, config.width)
    pos_flat = state.pos.reshape(-1, 2)
    target_flat = state.target.reshape(-1, 2)
    energy_flat = energy_maps.reshape(-1, config.height, config.width)

    obs_flat = jax.vmap(
        lambda grid, occ, pos, target, energy: _single_agent_aligned_observation(config, grid, occ, pos, target, energy)
    )(grid_flat, occ_flat, pos_flat, target_flat, energy_flat)
    return obs_flat.reshape(batch_size, agent_count, 4, config.pyg_obs_diam, config.pyg_obs_diam)


def reset_env(config: EnvConfig, init_pos: jnp.ndarray, init_target: jnp.ndarray, grid: jnp.ndarray, rng_key: jax.Array):
    batched_obs, batched_state = batched_reset_env(
        config,
        init_pos[None, ...],
        init_target[None, ...],
        grid[None, ...],
        rng_key[None, ...],
    )
    state = State(
        pos=batched_state.pos[0],
        target=batched_state.target[0],
        step_count=batched_state.step_count[0],
        rng_key=batched_state.rng_key[0],
        arrived=batched_state.arrived[0],
        terminated=batched_state.terminated[0],
        truncated=batched_state.truncated[0],
    )
    return batched_obs[0], state


def batched_reset_env(
    config: EnvConfig,
    init_pos: jnp.ndarray,
    init_target: jnp.ndarray,
    grids: jnp.ndarray,
    rng_keys: jax.Array,
):
    _validate_config(config)
    state = _batched_reset_from_frozen(
        config,
        init_pos,
        init_target,
        jnp.zeros(init_pos.shape[:2], dtype=bool),
        grids,
        rng_keys,
    )
    obs = build_batched_aligned_observations(config, state, grids)
    return obs, state


def _batched_reset_from_frozen(
    config: EnvConfig,
    positions: jnp.ndarray,
    goals: jnp.ndarray,
    arrived: jnp.ndarray,
    grids: jnp.ndarray,
    rng_keys: jax.Array,
) -> State:
    del grids  # The immutable map remains an explicit transition input.
    batch_size = positions.shape[0]
    terminated = jnp.all(arrived, axis=1)
    return State(
        pos=positions,
        target=goals,
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        rng_key=rng_keys,
        arrived=arrived,
        terminated=terminated,
        truncated=jnp.zeros((batch_size,), dtype=bool),
    )


_reset_frozen_compiled = jax.jit(_batched_reset_from_frozen, static_argnums=(0,))


def reset_frozen_jit(
    config: EnvConfig,
    positions: jnp.ndarray,
    goals: jnp.ndarray,
    arrived: jnp.ndarray,
    grids: jnp.ndarray,
    rng_keys: jax.Array,
) -> State:
    """Reset a batched state from caller-owned device arrays without observations."""

    _validate_config(config)
    return _reset_frozen_compiled(config, positions, goals, arrived, grids, rng_keys)


def _resolve_collisions(config: EnvConfig, curr_pos: jnp.ndarray, prop_pos: jnp.ndarray) -> jnp.ndarray:
    n = config.num_agents
    indices = jnp.arange(n, dtype=jnp.int32)

    def cond_fn(val):
        _, p_pos, done = val
        return ~done

    def body_fn(val):
        c_pos, p_pos, _ = val
        moved = jnp.any(p_pos != c_pos, axis=-1)

        same_dest = jnp.all(p_pos[:, None, :] == p_pos[None, :, :], axis=-1)
        swap = jnp.all(p_pos[:, None, :] == c_pos[None, :, :], axis=-1) & jnp.all(
            p_pos[None, :, :] == c_pos[:, None, :], axis=-1
        )
        chain = jnp.all(p_pos[:, None, :] == c_pos[None, :, :], axis=-1) & jnp.all(
            p_pos[None, :, :] == c_pos[None, :, :], axis=-1
        )
        higher_id = indices[:, None] > indices[None, :]
        not_self = ~jnp.eye(n, dtype=bool)

        vertex_conflict = same_dest & higher_id & not_self
        swap_conflict = swap & not_self
        chain_conflict = chain & not_self
        collision = moved & jnp.any(vertex_conflict | swap_conflict | chain_conflict, axis=1)

        new_p_pos = jnp.where(collision[:, None], c_pos, p_pos)
        done = jnp.all(new_p_pos == p_pos)
        return c_pos, new_p_pos, done

    _, final_pos, _ = jax.lax.while_loop(cond_fn, body_fn, (curr_pos, prop_pos, False))
    return final_pos


def _assign_new_goals(config: EnvConfig, grid: jnp.ndarray, targets: jnp.ndarray, reached: jnp.ndarray, rng_key: jax.Array):
    cell_count = config.height * config.width
    free_mask_flat = grid.reshape(-1) == 0
    target_flat = targets[:, 0] * config.width + targets[:, 1]
    base_taken = jnp.zeros(cell_count, dtype=bool).at[target_flat].set(~reached)
    start_indices = jax.random.randint(rng_key, (config.num_agents,), 0, cell_count, dtype=jnp.int32)
    goal_changed_flags = jnp.zeros((config.num_agents,), dtype=jnp.int32)

    def body_fn(i, carry):
        cur_targets, taken_mask, changed_flags = carry

        def assign_fn(inner_carry):
            inner_targets, inner_taken, inner_flags = inner_carry
            start = start_indices[i]

            def scan_fn(_, scan_state):
                chosen_idx, found = scan_state
                attempt = _
                idx = (start + attempt) % cell_count
                valid = free_mask_flat[idx] & (~inner_taken[idx])
                chosen_idx = jnp.where((~found) & valid, idx, chosen_idx)
                found = found | valid
                return chosen_idx, found

            chosen_idx, found = jax.lax.fori_loop(0, cell_count, scan_fn, (jnp.int32(0), False))
            new_goal = jnp.array([chosen_idx // config.width, chosen_idx % config.width], dtype=jnp.int32)
            inner_targets = jax.lax.cond(found, lambda t: t.at[i].set(new_goal), lambda t: t, inner_targets)
            inner_taken = jax.lax.cond(found, lambda t: t.at[chosen_idx].set(True), lambda t: t, inner_taken)
            inner_flags = inner_flags.at[i].set(jnp.where(found, 1, 0))
            return inner_targets, inner_taken, inner_flags

        return jax.lax.cond(reached[i], assign_fn, lambda x: x, (cur_targets, taken_mask, changed_flags))

    return jax.lax.fori_loop(0, config.num_agents, body_fn, (targets, base_taken, goal_changed_flags))


def step_env(config: EnvConfig, state: State, actions: jnp.ndarray, grid: jnp.ndarray):
    arrived, terminated, truncated = _state_status_arrays(state)
    batched_state = State(
        pos=state.pos[None, ...],
        target=state.target[None, ...],
        step_count=state.step_count[None],
        rng_key=state.rng_key[None, ...],
        arrived=arrived[None, ...],
        terminated=terminated[None, ...],
        truncated=truncated[None, ...],
    )
    obs, new_state, reward, done, info = batched_step_env(
        config,
        batched_state,
        actions[None, ...],
        grid[None, ...],
    )
    single_state = State(
        pos=new_state.pos[0],
        target=new_state.target[0],
        step_count=new_state.step_count[0],
        rng_key=new_state.rng_key[0],
        arrived=new_state.arrived[0],
        terminated=new_state.terminated[0],
        truncated=new_state.truncated[0],
    )
    single_info = {
        "step_count": info["step_count"][0],
        "reward_sum": info["reward_sum"][0],
        "goal_changed_flags": info["goal_changed_flags"][0],
    }
    return obs[0], single_state, reward[0], done[0], single_info


def batched_step_env(config: EnvConfig, state: State, actions: jnp.ndarray, grids: jnp.ndarray):
    _validate_config(config)
    new_state, reward, terminated, truncated, info = _batched_transition(
        config, state, actions, grids
    )
    obs = build_batched_aligned_observations(config, new_state, grids)
    done = terminated | truncated
    return obs, new_state, reward, done, info


def _batched_transition(
    config: EnvConfig,
    state: State,
    actions: jnp.ndarray,
    grids: jnp.ndarray,
):
    arrived, old_terminated, old_truncated = _state_status_arrays(state)
    done_before = old_terminated | old_truncated

    actions = jnp.where((actions < 0) | (actions > 4), 0, actions)
    if config.task_mode == "standard_mapf":
        actions = jnp.where(arrived, 0, actions)
        actions = jnp.where(done_before[:, None], 0, actions)
    proposed_pos = state.pos + MOVES[actions]

    out_of_bounds = (
        (proposed_pos[..., 0] < 0)
        | (proposed_pos[..., 0] >= config.height)
        | (proposed_pos[..., 1] < 0)
        | (proposed_pos[..., 1] >= config.width)
    )
    safe_prop_pos = jnp.stack(
        [
            jnp.clip(proposed_pos[..., 0], 0, config.height - 1),
            jnp.clip(proposed_pos[..., 1], 0, config.width - 1),
        ],
        axis=-1,
    )
    hit_wall = jax.vmap(lambda grid, pos: grid[pos[:, 0], pos[:, 1]])(grids, safe_prop_pos) == 1
    invalid_move = out_of_bounds | hit_wall
    proposed_pos = jnp.where(invalid_move[..., None], state.pos, proposed_pos)

    final_pos = jax.vmap(lambda curr, prop: _resolve_collisions(config, curr, prop))(state.pos, proposed_pos)
    if config.task_mode == "standard_mapf":
        final_pos = jnp.where(done_before[:, None, None], state.pos, final_pos)
    final_pos = final_pos.astype(state.pos.dtype)
    reward = jnp.where(jnp.all(final_pos == state.target, axis=-1), 1.0, 0.0)
    if config.task_mode == "standard_mapf":
        reward = jnp.where(done_before[:, None], 0.0, reward)
    reached = reward > 0.5

    if config.task_mode == "lifelong":
        split_keys = jax.vmap(lambda key: jax.random.split(key))(state.rng_key)
        new_rng_keys = split_keys[:, 0, :]
        goal_keys = split_keys[:, 1, :]
        new_targets, _, goal_changed_flags = jax.vmap(
            lambda grid, targets, reached_mask, key: _assign_new_goals(
                config, grid, targets, reached_mask, key
            )
        )(grids, state.target, reached, goal_keys)
        new_step_count = state.step_count + jnp.int32(1)
        new_arrived = arrived
        terminated = jnp.zeros_like(new_step_count, dtype=bool)
        truncated = new_step_count >= config.max_steps
    else:
        new_rng_keys = state.rng_key
        new_targets = state.target
        goal_changed_flags = jnp.zeros_like(actions, dtype=jnp.int32)
        active = ~done_before
        new_step_count = state.step_count + active.astype(jnp.int32)
        on_target = jnp.all(final_pos == state.target, axis=-1)
        new_arrived = arrived | (on_target & active[:, None])
        just_terminated = jnp.all(new_arrived, axis=1) & active
        terminated = old_terminated | just_terminated
        hit_horizon = (new_step_count >= config.max_episode_steps) & active
        truncated = old_truncated | (hit_horizon & ~terminated)

    new_state = State(
        pos=final_pos,
        target=new_targets,
        step_count=new_step_count,
        rng_key=new_rng_keys,
        arrived=new_arrived,
        terminated=terminated,
        truncated=truncated,
    )
    info = {
        "step_count": new_state.step_count,
        "reward_sum": jnp.sum(reward, axis=1),
        "goal_changed_flags": goal_changed_flags,
    }
    return new_state, reward, terminated, truncated, info


_step_transition_compiled = jax.jit(_batched_transition, static_argnums=(0,))


def step_transition_jit(
    config: EnvConfig,
    state: State,
    actions: jnp.ndarray,
    grids: jnp.ndarray,
):
    """Run only the device-resident transition; timing belongs to the harness."""

    _validate_config(config)
    return _step_transition_compiled(config, state, actions, grids)


def block_transition_ready(state: State) -> None:
    """Synchronize a transition state without returning or materializing it."""

    jax.block_until_ready(state)


def _sample_batch_init(config: EnvConfig, batch_size: int, key: jax.Array):
    cell_count = config.height * config.width

    def sample_one(k):
        perm = jax.random.permutation(k, cell_count)
        pos_flat = perm[: config.num_agents]
        target_flat = perm[config.num_agents : 2 * config.num_agents]
        pos = jnp.stack([pos_flat // config.width, pos_flat % config.width], axis=-1).astype(jnp.int32)
        target = jnp.stack([target_flat // config.width, target_flat % config.width], axis=-1).astype(jnp.int32)
        return pos, target

    keys = jax.random.split(key, batch_size)
    return jax.vmap(sample_one)(keys)


def run_benchmark(batch_size: int = 1024, steps_to_test: int = 100, warmup_steps: int = 2, config: EnvConfig | None = None):
    config = config or EnvConfig()
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    init_pos, init_target = _sample_batch_init(config, batch_size, k1)
    grids = jnp.zeros((batch_size, config.height, config.width), dtype=jnp.int32)
    env_keys = jax.random.split(k2, batch_size)
    actions = jax.random.randint(k3, (batch_size, config.num_agents), 0, 5, dtype=jnp.int32)

    jit_reset = jax.jit(batched_reset_env, static_argnums=(0,))
    jit_step = jax.jit(batched_step_env, static_argnums=(0,))

    obs, states = jit_reset(config, init_pos, init_target, grids, env_keys)
    jax.block_until_ready(obs)

    for _ in range(warmup_steps):
        obs, states, rewards, dones, infos = jit_step(config, states, actions, grids)
        jax.block_until_ready(obs)

    start_time = time.time()
    for _ in range(steps_to_test):
        obs, states, rewards, dones, infos = jit_step(config, states, actions, grids)
    jax.block_until_ready(obs)
    end_time = time.time()

    elapsed = end_time - start_time
    total_env_steps = batch_size * steps_to_test
    metrics = {
        "elapsed_s": elapsed,
        "env_steps_s": total_env_steps / elapsed,
        "agent_steps_s": total_env_steps * config.num_agents / elapsed,
        "obs_shape": tuple(obs.shape),
        "batch_size": batch_size,
        "num_agents": config.num_agents,
        "height": config.height,
        "width": config.width,
    }

    print(f"Benchmarking with Batch Size: {batch_size}, Agents: {config.num_agents}...")
    print(f"Time taken for {steps_to_test} steps: {elapsed:.4f} seconds")
    print(f"Throughput: {metrics['env_steps_s']:,.0f} Env Steps / Second")
    print(f"Agent Throughput: {metrics['agent_steps_s']:,.0f} Agent Steps / Second")
    print(f"Observation Shape Output: {metrics['obs_shape']}")
    return metrics


if __name__ == "__main__":
    run_benchmark()
