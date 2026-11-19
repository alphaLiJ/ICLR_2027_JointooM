"""Discrete MAPF task adaptation of CAMAR, aligned to our JAXMAPF reference.

CAMAR's Flax physical/environment states and GPU batching API are retained.
Grid motion, soft conflicts, arrival/episode rules and MAGAT observations are
task adaptations, not features of unmodified upstream CAMAR.
"""
import bootstrap
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct

from camar.dynamics import BaseDynamic, PhysicalState
from camar.environment import Camar
from camar.maps.base import base_map
from camar.utils import Box, State as CamarState
from camar.wrappers import BatchEnvWrapper


@dataclass(frozen=True)
class MAPFConfig:
    height: int = 128
    width: int = 128
    num_agents: int = 256
    task_mode: str = "standard_mapf"
    max_episode_steps: int = 256
    max_steps: int = 100
    fov_radius: int = 5
    clamp_value: float = 1.0
    energy_iters: int = 254

    @property
    def pyg_obs_diam(self):
        return 2 * self.fov_radius + 3

    def __post_init__(self):
        if self.task_mode not in ("standard_mapf", "lifelong"):
            raise ValueError(self.task_mode)
        if min(self.height, self.width, self.num_agents, self.max_episode_steps,
               self.max_steps, self.fov_radius) <= 0:
            raise ValueError("MAPF dimensions, agent count and horizons must be positive")


MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)


@struct.dataclass
class GridPhysicalState(PhysicalState):
    grid: jax.Array
    arrived: jax.Array

    @classmethod
    def create(cls, key, landmark_pos, agent_pos, goal_pos, sizes):
        raise NotImplementedError("Grid state is loaded by DiscreteMAPF.reset_from_arrays")


@struct.dataclass
class MAPFState(CamarState):
    terminated: jax.Array
    truncated: jax.Array

    @property
    def pos(self):
        return self.physical_state.agent_pos

    @property
    def target(self):
        return self.goal_pos

    @property
    def arrived(self):
        return self.physical_state.arrived

    @property
    def rng_key(self):
        return self.goal_keys

    @property
    def step_count(self):
        return self.step


class DiscreteActions:
    def __init__(self, agents):
        self.shape = (agents,)
        self.dtype = jnp.int32

    def sample(self, key):
        return jax.random.randint(key, self.shape, 0, 5, dtype=self.dtype)


def resolve_conflicts(current, proposed):
    """Migrate JAXMAPF soft conflicts: lower id wins, no swaps, blocked chains."""
    ids = jnp.arange(current.shape[0])
    other = ~jnp.eye(current.shape[0], dtype=bool)

    def body(value):
        destinations, _ = value
        moving = jnp.any(destinations != current, axis=-1)
        same_destination = jnp.all(destinations[:, None] == destinations[None, :], axis=-1)
        enters_old_cell = jnp.all(destinations[:, None] == current[None, :], axis=-1)
        swap = enters_old_cell & enters_old_cell.T
        stationary = jnp.all(destinations == current, axis=-1)
        blocked = enters_old_cell & stationary[None, :]
        loses_vertex = same_destination & (ids[:, None] > ids[None, :])
        rejected = moving & jnp.any((loses_vertex | swap | blocked) & other, axis=-1)
        updated = jnp.where(rejected[:, None], current, destinations)
        return updated, jnp.all(updated == destinations)

    return jax.lax.while_loop(lambda v: ~v[1], body, (proposed, jnp.array(False)))[0]


class GridDynamics(BaseDynamic):
    def __init__(self, config):
        self.config = config

    @property
    def action_size(self):
        return 1

    @property
    def dt(self):
        return 1.0

    @property
    def state_class(self):
        return GridPhysicalState

    def integrate(self, key, force, physical_state, actions):
        del key, force
        c = self.config
        current = physical_state.agent_pos
        actions = jnp.where((actions < 0) | (actions > 4), 0, actions).astype(jnp.int32)
        if c.task_mode == "standard_mapf":
            actions = jnp.where(physical_state.arrived, 0, actions)
        proposed = current + MOVES[actions]
        outside = ((proposed[:, 0] < 0) | (proposed[:, 0] >= c.height)
                   | (proposed[:, 1] < 0) | (proposed[:, 1] >= c.width))
        safe = jnp.clip(proposed, jnp.array([0, 0]), jnp.array([c.height - 1, c.width - 1]))
        wall = physical_state.grid[safe[:, 0], safe[:, 1]] == 1
        proposed = jnp.where((outside | wall)[:, None], current, proposed)
        position = resolve_conflicts(current, proposed).astype(current.dtype)
        return physical_state.replace(agent_pos=position)


class FrozenGridMap(base_map):
    """CAMAR map interface; obstacles remain grid cells rather than circles."""
    def __init__(self, config, grid=None, positions=None, goals=None):
        self.config = config
        self.grid = jnp.zeros((config.height, config.width), dtype=jnp.uint8) if grid is None else jnp.asarray(grid)
        self.positions = None if positions is None else jnp.asarray(positions, dtype=jnp.int32)
        self.goals = None if goals is None else jnp.asarray(goals, dtype=jnp.int32)
        super().__init__()

    def setup_rad(self):
        self.agent_rad = self.landmark_rad = self.goal_rad = 0.0
        self.proportional_goal_rad = False

    @property
    def height(self):
        return self.config.height

    @property
    def width(self):
        return self.config.width

    @property
    def num_agents(self):
        return self.config.num_agents

    @property
    def num_landmarks(self):
        return 0

    def reset(self, key):
        if self.positions is None or self.goals is None:
            raise ValueError("Use reset_from_arrays, or provide frozen positions and goals")
        return key, jnp.zeros((0, 2), dtype=jnp.int32), self.positions, self.goals, self.generate_sizes(key)


def assign_goals(config, grid, targets, reached, key):
    """Migrate the reference RNG/cyclic free-cell assignment, including id order."""
    cells = config.height * config.width
    free = grid.reshape(-1) == 0
    old_indices = targets[:, 0] * config.width + targets[:, 1]
    taken = jnp.zeros(cells, dtype=bool).at[old_indices].set(~reached)
    starts = jax.random.randint(key, (config.num_agents,), 0, cells, dtype=jnp.int32)
    flags = jnp.zeros((config.num_agents,), dtype=jnp.int32)

    def assign_one(agent, carry):
        def update(carry):
            goals, occupied, changed = carry

            def scan(offset, value):
                chosen, found = value
                candidate = (starts[agent] + offset) % cells
                valid = free[candidate] & ~occupied[candidate]
                return jnp.where(~found & valid, candidate, chosen), found | valid

            chosen, found = jax.lax.fori_loop(0, cells, scan, (jnp.int32(0), jnp.array(False)))
            goal = jnp.array([chosen // config.width, chosen % config.width], dtype=targets.dtype)
            goals = jax.lax.cond(found, lambda g: g.at[agent].set(goal), lambda g: g, goals)
            occupied = jax.lax.cond(found, lambda t: t.at[chosen].set(True), lambda t: t, occupied)
            return goals, occupied, changed.at[agent].set(found.astype(jnp.int32))

        return jax.lax.cond(reached[agent], update, lambda x: x, carry)

    return jax.lax.fori_loop(0, config.num_agents, assign_one, (targets, taken, flags))


class DiscreteMAPF(Camar):
    def __init__(self, config=MAPFConfig(), map_generator=None):
        self.config = config
        super().__init__(map_generator=map_generator or FrozenGridMap(config),
                         dynamic=GridDynamics(config), lifelong=False, frameskip=0,
                         max_steps=config.max_episode_steps, max_obs=0)
        self.action_spaces = DiscreteActions(config.num_agents)
        d = config.pyg_obs_diam
        self.observation_spaces = Box(-jnp.inf, jnp.inf, (config.num_agents, 4, d, d))

    def reset_from_arrays(self, key, grid, positions, goals, arrived):
        positions = jnp.asarray(positions, dtype=jnp.int32)
        goals = jnp.asarray(goals, dtype=jnp.int32)
        arrived = jnp.asarray(arrived, dtype=bool)
        zeros = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        physical = GridPhysicalState(positions, jnp.asarray(grid, dtype=jnp.uint8), arrived)
        return MAPFState(
            physical_state=physical, landmark_pos=jnp.zeros((0, 2), dtype=jnp.int32),
            goal_pos=goals, sizes=self.map_generator.generate_sizes(key),
            is_collision=zeros.astype(bool), step=jnp.int32(0),
            on_goal=jnp.all(positions == goals, axis=-1),
            time_to_reach_goal=jnp.full((self.num_agents,), self.config.max_episode_steps, dtype=jnp.int32),
            num_collisions=zeros, goal_keys=key, terminated=jnp.all(arrived),
            truncated=jnp.array(False))

    def reset(self, key):
        _, _, positions, goals, _ = self.map_generator.reset(key)
        state = self.reset_from_arrays(key, self.map_generator.grid, positions, goals,
                                       jnp.zeros((self.num_agents,), dtype=bool))
        return self.get_obs(state), state

    def transition(self, key, state, actions):
        """CAMAR step signature with a compact position observation, no builder."""
        c = self.config
        done_before = state.terminated | state.truncated
        if c.task_mode == "standard_mapf":
            actions = jnp.where(done_before, 0, actions)
        physical = self.dynamic.integrate(key, None, state.physical_state, actions)
        if c.task_mode == "standard_mapf":
            physical = physical.replace(agent_pos=jnp.where(done_before, state.pos, physical.agent_pos))
        reached = jnp.all(physical.agent_pos == state.goal_pos, axis=-1)
        reward = reached.astype(jnp.float32)
        collision = jnp.any(physical.agent_pos != state.pos + MOVES[jnp.clip(actions, 0, 4)], axis=-1)
        if c.task_mode == "standard_mapf":
            active = ~done_before
            step = state.step + active.astype(jnp.int32)
            reward = jnp.where(active, reward, 0.0)
            arrived = state.arrived | (reached & active)
            physical = physical.replace(arrived=arrived)
            terminated = state.terminated | (jnp.all(arrived) & active)
            truncated = state.truncated | ((step >= c.max_episode_steps) & active & ~terminated)
            goals, key_next = state.goal_pos, state.goal_keys
            changed = jnp.zeros((self.num_agents,), dtype=jnp.int32)
            collision = collision & active & ~state.arrived
        else:
            key_next, goal_key = jax.random.split(state.goal_keys)
            goals, _, changed = assign_goals(c, physical.grid, state.goal_pos, reached, goal_key)
            step = state.step + jnp.int32(1)
            terminated, truncated = jnp.array(False), step >= c.max_steps
        newly_arrived = ~state.on_goal & reached
        updated = state.replace(
            physical_state=physical, goal_pos=goals, goal_keys=key_next, step=step,
            is_collision=collision, on_goal=reached,
            time_to_reach_goal=jnp.where(newly_arrived, step, state.time_to_reach_goal),
            num_collisions=state.num_collisions + collision.astype(jnp.int32),
            terminated=terminated, truncated=truncated)
        if c.task_mode == "standard_mapf":
            updated = jax.tree.map(lambda new, old: jnp.where(done_before, old, new), updated, state)
        info = dict(step_count=updated.step, reward_sum=jnp.sum(reward),
                    goal_changed_flags=changed, terminated=terminated, truncated=truncated)
        return updated.pos, updated, reward, terminated | truncated, info

    def get_obs(self, state):
        from observations import build_observations, build_energy_maps
        grid, positions, goals = state.physical_state.grid, state.pos, state.goal_pos
        energy = build_energy_maps(self.config, grid[None], goals[None])
        return build_observations(self.config, grid[None], positions[None], goals[None], energy)[0]

    def step(self, key, state, actions):
        _, updated, reward, done, info = self.transition(key, state, actions)
        return self.get_obs(updated), updated, reward, done, info


class _TransitionView:
    """Reuse upstream batching while input construction is timed before policy use."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, key):
        return getattr(self.env, key)

    def step(self, key, state, actions):
        return self.env.transition(key, state, actions)


def batch_transition_environment(env, num_envs):
    return BatchEnvWrapper(_TransitionView(env), num_envs)
