"""MAGAT observation logic migrated from the existing JAXMAPF baseline.

The functions below preserve the reference's byte-energy and border convention.
They are a policy-input adaptation, not original upstream CAMAR functionality.
"""
import bootstrap
import jax
import jax.numpy as jnp
from discrete_mapf import MAPFConfig

def _build_batched_occupancy_grids(config: MAPFConfig, pos: jnp.ndarray) -> jnp.ndarray:
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


def _build_batched_energy_maps(config: MAPFConfig, grids: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
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


def _goal_projection_channel(config: MAPFConfig, cx: jnp.ndarray, cy: jnp.ndarray, gx: jnp.ndarray, gy: jnp.ndarray) -> jnp.ndarray:
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
        goal_x_from_y = jnp.int32(dist) + _round_ratio_even(jnp.int32(dist) * rel_gx, jnp.abs(rel_gy))

        goal_x_fov = jnp.int32(dist * (sign_x + 1.0))
        goal_y_from_x = jnp.int32(dist) + _round_ratio_even(jnp.int32(dist) * rel_gy, jnp.abs(rel_gx))

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
    config: MAPFConfig,
    grid: jnp.ndarray,
    occupancy_grid: jnp.ndarray,
    pos: jnp.ndarray,
    target: jnp.ndarray,
    energy_map: jnp.ndarray,
    static_wall_cost: bool = False,
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

    center_ctg = energy_map[cx, cy].astype(jnp.float32)
    target_ctg = energy_map[clipped_tx, clipped_ty].astype(jnp.float32)
    ctg_val = (center_ctg - target_ctg) / (2.0 * float(obs_r))
    ctg_val = jnp.clip(ctg_val, -config.clamp_value, config.clamp_value)
    if static_wall_cost:
        # The retained E4b JAXMAPF implementation includes this official MAPF
        # obstacle convention, including cells near unreachable goals.
        ctg_val = jnp.where(in_valid & (grid_vals == 1), config.clamp_value, ctg_val)
    ctg_val = jnp.where(too_far_oob, config.clamp_value, ctg_val)
    ctg_val = jnp.where(virtual_wall, config.clamp_value, ctg_val)

    goal_channel = _goal_projection_channel(config, cx, cy, gx, gy)

    out = jnp.zeros((4, pyg_obs_diam, pyg_obs_diam), dtype=jnp.float32)
    out = out.at[0, 1:-1, 1:-1].set(obs_val)
    out = out.at[1, 1:-1, 1:-1].set(occ_val)
    out = out.at[2].set(goal_channel)
    out = out.at[3, 1:-1, 1:-1].set(ctg_val)
    return out

def _round_ratio_even(numerator, denominator):
    """Exact discrete projection: XLA's float reciprocal can perturb half ties.

    E.g. 6 * -25 / 60 must round -2.5 to -2, not -3. This matches the
    nearest-even integer-ratio rule already used by the CUDA reference.
    """
    denominator = jnp.maximum(denominator, 1)
    magnitude = jnp.abs(numerator)
    quotient, remainder = magnitude // denominator, magnitude % denominator
    increase = (2 * remainder > denominator) | ((2 * remainder == denominator) & (quotient % 2 == 1))
    return jnp.sign(numerator) * (quotient + increase.astype(jnp.int32))


build_energy_maps = _build_batched_energy_maps


def build_observations(config, grids, positions, goals, energy, static_wall_cost=False):
    batch, agents = positions.shape[:2]
    occupancy = _build_batched_occupancy_grids(config, positions)
    flat_grid = jnp.broadcast_to(grids[:, None], (batch, agents, config.height, config.width)).reshape(-1, config.height, config.width)
    flat_occ = jnp.broadcast_to(occupancy[:, None], (batch, agents, config.height, config.width)).reshape(-1, config.height, config.width)
    obs = jax.vmap(lambda g, o, p, t, e: _single_agent_aligned_observation(config, g, o, p, t, e, static_wall_cost))(
        flat_grid, flat_occ, positions.reshape(-1, 2), goals.reshape(-1, 2),
        energy.reshape(-1, config.height, config.width))
    return obs.reshape(batch, agents, 4, config.pyg_obs_diam, config.pyg_obs_diam)


class CachedObservations:
    """Batch-level conditional avoids vmap executing both cache branches.

    Goal identity is checked on GPU. Cold construction and goal refresh happen
    in prepare(), inside the complete-decision timer. No CPU distance oracle is
    used. Maps are immutable for this object's lifetime.
    """
    def __init__(self, config, grids):
        self.config = config
        self.grids = grids
        self.energy = None
        self.goals = None

        def refresh(goals, previous_goals, old_energy):
            dirty = jnp.any(goals != previous_goals)
            return jax.lax.cond(dirty,
                lambda _: build_energy_maps(config, grids, goals),
                lambda _: old_energy, operand=None)

        self.refresh = jax.jit(refresh)
        self.cold = jax.jit(lambda goals: build_energy_maps(config, grids, goals))
        self.view = jax.jit(lambda positions, goals, energy: build_observations(
            config, grids, positions, goals, energy))

    def prepare(self, state):
        if self.energy is None:
            self.energy = self.cold(state.goal_pos)
        else:
            self.energy = self.refresh(state.goal_pos, self.goals, self.energy)
        self.goals = state.goal_pos
        return self.view(state.pos, state.goal_pos, self.energy)

    def invalidate(self):
        self.energy = self.goals = None


class PaperCachedObservations:
    """E4b configuration: uint8 fixed-goal cache, 16-environment BFS chunks.

    The static goal contract avoids repeated dirty checks. The obstacle cost
    channel matches the retained E4b JAXMAPF, not the older anonymous snapshot.
    """
    def __init__(self, config, grids):
        if config.task_mode != 'standard_mapf':
            raise ValueError('The paper E4b cache requires immutable fixed goals')
        self.config,self.grids=config,grids
        self.energy=self.goals=None
        self.chunk=jax.jit(lambda g,t:build_energy_maps(config,g,t).astype(jnp.uint8))
        self.view=jax.jit(lambda p,t,e:build_observations(config,grids,p,t,e,static_wall_cost=True))

    def prepare(self,state):
        if self.energy is None:
            parts=[]
            for start in range(0,len(self.grids),16):
                part=self.chunk(self.grids[start:start+16],state.goal_pos[start:start+16])
                jax.block_until_ready(part)
                parts.append(part)
            self.energy=parts[0] if len(parts)==1 else jnp.concatenate(parts,axis=0)
            self.goals=state.goal_pos
        return self.view(state.pos,state.goal_pos,self.energy)

    def invalidate(self):
        self.energy=self.goals=None
