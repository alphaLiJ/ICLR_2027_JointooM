"""MAPF-GPT online sample schema and correctness-oriented CPU tokenizer."""

from __future__ import annotations

from collections import deque

import numpy as np


MAPFGPT_FEATURE_DIM = 13
MAPFGPT_EMPTY_ACTION = 5
MAPFGPT_NUM_ACTIONS = 5
MAPFGPT_HISTORY_LEN = 5
MAPFGPT_CONTEXT_SIZE = 256
MAPFGPT_VISIBLE_AGENTS = 13
MAPFGPT_OBS_RADIUS = 5
MAPFGPT_COST_LIMIT = 20

TOKEN_UNREACHABLE = 41
TOKEN_NEGATIVE_CLIP = 42
TOKEN_POSITIVE_CLIP = 43
TOKEN_EMPTY_ACTION = 44
TOKEN_WAIT_ACTION = 45
TOKEN_GREEDY_BASE = 50
TOKEN_PADDING = 66

_MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _as_action_vector(actions: np.ndarray, num_agents: int) -> np.ndarray:
    result = np.asarray(actions)
    if result.shape != (num_agents,):
        raise ValueError(
            f"actions must have shape [{num_agents}], got {result.shape}"
        )
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"actions must use an integer dtype, got {result.dtype}")
    if np.any(result < 0) or np.any(result >= MAPFGPT_NUM_ACTIONS):
        raise ValueError("current action must be in [0, 4]")
    return result.astype(np.uint16, copy=False)


class ActionHistory:
    """Five previous action commands for every agent in one environment."""

    def __init__(self, num_agents: int):
        if int(num_agents) <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        self.num_agents = int(num_agents)
        self._history = np.full(
            (self.num_agents, MAPFGPT_HISTORY_LEN),
            MAPFGPT_EMPTY_ACTION,
            dtype=np.uint16,
        )

    def snapshot(self) -> np.ndarray:
        return self._history.copy()

    def append(self, actions: np.ndarray) -> None:
        action_vector = _as_action_vector(actions, self.num_agents)
        self._history[:, :-1] = self._history[:, 1:]
        self._history[:, -1] = action_vector

    def reset(self) -> None:
        self._history.fill(MAPFGPT_EMPTY_ACTION)


def build_mapf_gpt_rows(
    observations,
    actions: np.ndarray,
    *,
    env_id: int,
    refresh_flags,
    history: np.ndarray,
    coordinate_offset: int = MAPFGPT_OBS_RADIUS,
) -> np.ndarray:
    """Build one self-contained env block without mutating action history."""

    num_agents = len(observations)
    action_vector = _as_action_vector(actions, num_agents)
    history_array = np.asarray(history)
    if history_array.shape != (num_agents, MAPFGPT_HISTORY_LEN):
        raise ValueError(
            "history must have shape "
            f"[{num_agents}, {MAPFGPT_HISTORY_LEN}], got {history_array.shape}"
        )
    if not np.issubdtype(history_array.dtype, np.integer):
        raise ValueError(f"history must use an integer dtype, got {history_array.dtype}")
    if np.any(history_array < 0) or np.any(history_array > MAPFGPT_EMPTY_ACTION):
        raise ValueError("history action must be in [0, 5]")

    flags = np.asarray(refresh_flags)
    if flags.ndim == 0:
        flags = np.full(num_agents, int(flags), dtype=np.uint16)
    else:
        flags = flags.reshape(-1)
    if flags.shape != (num_agents,):
        raise ValueError(
            f"refresh_flags must have shape [{num_agents}], got {flags.shape}"
        )
    if np.any((flags != 0) & (flags != 1)):
        raise ValueError("refresh flag must be 0 or 1")

    positions = np.asarray(
        [obs["global_xy"] for obs in observations], dtype=np.int64
    ) - int(coordinate_offset)
    goals = np.asarray(
        [obs["global_target_xy"] for obs in observations], dtype=np.int64
    ) - int(coordinate_offset)
    if positions.shape != (num_agents, 2) or goals.shape != (num_agents, 2):
        raise ValueError("global_xy and global_target_xy must each contain two coordinates")
    if np.any(positions < 0) or np.any(goals < 0):
        raise ValueError("coordinates become negative after removing the observation border")
    if np.any(positions > np.iinfo(np.uint16).max) or np.any(
        goals > np.iinfo(np.uint16).max
    ):
        raise ValueError("coordinates exceed uint16 range")

    rows = np.zeros((num_agents, MAPFGPT_FEATURE_DIM), dtype=np.uint16)
    rows[:, 0] = int(env_id)
    rows[:, 1] = np.arange(num_agents, dtype=np.uint16)
    rows[:, 2:4] = positions.astype(np.uint16)
    rows[:, 4:6] = goals.astype(np.uint16)
    rows[:, 6] = action_vector
    rows[:, 7] = flags.astype(np.uint16)
    rows[:, 8:13] = history_array.astype(np.uint16, copy=False)
    return rows


def validate_mapf_gpt_rows(
    rows: np.ndarray,
    *,
    num_envs: int | None = None,
    agents_per_env: int | None = None,
) -> bool:
    data = np.asarray(rows)
    if data.dtype != np.uint16:
        raise ValueError(f"dtype must be np.uint16, got {data.dtype}")
    if data.ndim != 2 or data.shape[1] != MAPFGPT_FEATURE_DIM:
        raise ValueError(
            f"shape must be [N, {MAPFGPT_FEATURE_DIM}], got {data.shape}"
        )
    if np.any(data[:, 6] >= MAPFGPT_NUM_ACTIONS):
        raise ValueError("current action must be in [0, 4]")
    if np.any((data[:, 7] != 0) & (data[:, 7] != 1)):
        raise ValueError("refresh flag must be 0 or 1")
    if np.any(data[:, 8:13] > MAPFGPT_EMPTY_ACTION):
        raise ValueError("history action must be in [0, 5]")

    if (num_envs is None) != (agents_per_env is None):
        raise ValueError("num_envs and agents_per_env must be provided together")
    if num_envs is not None:
        num_envs = int(num_envs)
        agents_per_env = int(agents_per_env)
        expected_rows = num_envs * agents_per_env
        if data.shape[0] != expected_rows:
            raise ValueError(
                f"stage row count mismatch: expected {expected_rows}, got {data.shape[0]}"
            )
        blocks = data.reshape(num_envs, agents_per_env, MAPFGPT_FEATURE_DIM)
        expected_envs = np.arange(num_envs, dtype=np.uint16)[:, None]
        if not np.array_equal(blocks[:, :, 0], np.broadcast_to(expected_envs, blocks[:, :, 0].shape)):
            raise ValueError("rows must be strict env-major blocks")
        expected_agents = np.arange(agents_per_env, dtype=np.uint16)[None, :]
        if not np.array_equal(
            blocks[:, :, 1], np.broadcast_to(expected_agents, blocks[:, :, 1].shape)
        ):
            raise ValueError("rows must be agent-major inside each environment")
    return True


def _bfs_cost_to_go(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    height, width = grid.shape
    distances = np.full((height, width), 65535, dtype=np.uint16)
    gx, gy = goal
    if gx < 0 or gx >= height or gy < 0 or gy >= width or grid[gx, gy] != 0:
        return distances
    queue = deque([(gx, gy)])
    distances[gx, gy] = 0
    while queue:
        x, y = queue.popleft()
        next_distance = int(distances[x, y]) + 1
        for dx, dy in _MOVES[1:]:
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < height
                and 0 <= ny < width
                and grid[nx, ny] == 0
                and distances[nx, ny] == 65535
            ):
                distances[nx, ny] = next_distance
                queue.append((nx, ny))
    return distances


def _cost_token(value: int, center: int) -> int:
    if value == 65535:
        return TOKEN_UNREACHABLE
    difference = int(value) - int(center)
    if difference > MAPFGPT_COST_LIMIT:
        return TOKEN_POSITIVE_CLIP
    if difference < -MAPFGPT_COST_LIMIT:
        return TOKEN_NEGATIVE_CLIP
    return difference + MAPFGPT_COST_LIMIT


def _coordinate_token(value: int) -> int:
    return max(-MAPFGPT_COST_LIMIT, min(MAPFGPT_COST_LIMIT, int(value))) + MAPFGPT_COST_LIMIT


def _history_token(action: int) -> int:
    if int(action) == MAPFGPT_EMPTY_ACTION:
        return TOKEN_EMPTY_ACTION
    return TOKEN_WAIT_ACTION + int(action)


def _greedy_token(cost_map: np.ndarray, position: tuple[int, int]) -> int:
    x, y = position
    height, width = cost_map.shape
    current = int(cost_map[x, y]) if 0 <= x < height and 0 <= y < width else 65535
    mask = 0
    for direction, (dx, dy) in enumerate(_MOVES[1:]):
        nx, ny = x + dx, y + dy
        neighbor = (
            int(cost_map[nx, ny])
            if 0 <= nx < height and 0 <= ny < width
            else 65535
        )
        if neighbor != 65535 and current > neighbor:
            mask |= 1 << (3 - direction)
    return TOKEN_GREEDY_BASE + mask


def tokenize_stage_reference(
    grids: np.ndarray,
    rows: np.ndarray,
    *,
    num_agents: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slow, direct oracle matching the official closed-loop C++ semantics."""

    grids_array = np.asarray(grids)
    if grids_array.ndim != 3:
        raise ValueError(f"grids must have shape [E, H, W], got {grids_array.shape}")
    num_envs = grids_array.shape[0]
    validate_mapf_gpt_rows(
        rows, num_envs=num_envs, agents_per_env=int(num_agents)
    )
    blocks = rows.reshape(num_envs, num_agents, MAPFGPT_FEATURE_DIM)
    total_agents = num_envs * num_agents
    tokens = np.full(
        (total_agents, MAPFGPT_CONTEXT_SIZE), TOKEN_PADDING, dtype=np.int32
    )
    labels = blocks[:, :, 6].reshape(-1).astype(np.int64)

    for env_id in range(num_envs):
        grid = grids_array[env_id]
        block = blocks[env_id]
        cost_maps = [
            _bfs_cost_to_go(grid, (int(row[4]), int(row[5]))) for row in block
        ]
        positions = [(int(row[2]), int(row[3])) for row in block]
        goals = [(int(row[4]), int(row[5])) for row in block]

        for ego_id in range(num_agents):
            output = tokens[env_id * num_agents + ego_id]
            ego_x, ego_y = positions[ego_id]
            ego_cost = cost_maps[ego_id]
            center = (
                int(ego_cost[ego_x, ego_y])
                if 0 <= ego_x < grid.shape[0] and 0 <= ego_y < grid.shape[1]
                else 65535
            )

            token_index = 0
            for dx in range(-MAPFGPT_OBS_RADIUS, MAPFGPT_OBS_RADIUS + 1):
                for dy in range(-MAPFGPT_OBS_RADIUS, MAPFGPT_OBS_RADIUS + 1):
                    x, y = ego_x + dx, ego_y + dy
                    value = (
                        int(ego_cost[x, y])
                        if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]
                        else 65535
                    )
                    output[token_index] = _cost_token(value, center)
                    token_index += 1

            visible = [
                agent_id
                for agent_id, (x, y) in enumerate(positions)
                if abs(x - ego_x) <= MAPFGPT_OBS_RADIUS
                and abs(y - ego_y) <= MAPFGPT_OBS_RADIUS
            ]
            visible.sort(
                key=lambda agent_id: (
                    abs(positions[agent_id][0] - ego_x)
                    + abs(positions[agent_id][1] - ego_y),
                    agent_id,
                )
            )

            for slot, agent_id in enumerate(visible[:MAPFGPT_VISIBLE_AGENTS]):
                base = 121 + slot * 10
                pos_x, pos_y = positions[agent_id]
                goal_x, goal_y = goals[agent_id]
                output[base + 0] = _coordinate_token(pos_x - ego_x)
                output[base + 1] = _coordinate_token(pos_y - ego_y)
                output[base + 2] = _coordinate_token(goal_x - ego_x)
                output[base + 3] = _coordinate_token(goal_y - ego_y)
                output[base + 4 : base + 9] = [
                    _history_token(action) for action in block[agent_id, 8:13]
                ]
                output[base + 9] = _greedy_token(
                    cost_maps[agent_id], positions[agent_id]
                )

    return tokens, labels


__all__ = [
    "ActionHistory",
    "MAPFGPT_EMPTY_ACTION",
    "MAPFGPT_FEATURE_DIM",
    "build_mapf_gpt_rows",
    "tokenize_stage_reference",
    "validate_mapf_gpt_rows",
]
