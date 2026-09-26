"""Canonical, backend-neutral inputs for transition-only MAPF benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "frozen-transition-batch-v2"
_ACTION_DELTAS_BACKING = np.asarray(
    [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
    dtype=np.int8,
).tobytes(order="C")
ACTION_DELTAS = np.frombuffer(_ACTION_DELTAS_BACKING, dtype=np.int8).reshape(5, 2)

_ARRAY_DTYPES = {
    "instance_ids": np.dtype(np.int64),
    "grids": np.dtype(np.uint8),
    "positions": np.dtype(np.uint16),
    "goals": np.dtype(np.uint16),
    "arrived": np.dtype(np.bool_),
    "active": np.dtype(np.bool_),
    "actions": np.dtype(np.uint8),
}
_HASH_FIELDS = tuple(_ARRAY_DTYPES) + ("horizon",)
_NPZ_FIELDS = frozenset((*_ARRAY_DTYPES, "horizon", "metadata_json"))


def _owned_read_only_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    expected_dtype = _ARRAY_DTYPES[name]
    if array.dtype != expected_dtype:
        raise ValueError(f"{name} dtype must be {expected_dtype}, got {array.dtype}")
    typed_copy = np.array(array, copy=True, order="C")
    immutable_backing = typed_copy.tobytes(order="C")
    canonical = np.frombuffer(immutable_backing, dtype=expected_dtype).reshape(
        typed_copy.shape
    )
    return canonical


def _has_immutable_bytes_backing(array: np.ndarray) -> bool:
    backing: Any = array
    seen: set[int] = set()
    while isinstance(backing, np.ndarray):
        identity = id(backing)
        if identity in seen:
            return False
        seen.add(identity)
        if backing.base is None:
            return False
        backing = backing.base
    return isinstance(backing, bytes)


def _hash_component(digest: Any, name: str, value: np.ndarray) -> None:
    encoded_name = name.encode("ascii")
    encoded_dtype = value.dtype.str.encode("ascii")
    encoded_shape = json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    payload = value.tobytes(order="C")
    for component in (encoded_name, encoded_dtype, encoded_shape, payload):
        digest.update(len(component).to_bytes(8, "little"))
        digest.update(component)


@dataclass(frozen=True)
class FrozenTransitionBatch:
    """Immutable canonical state and action schedule shared by all backends."""

    instance_ids: np.ndarray = field(repr=False)
    grids: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    goals: np.ndarray = field(repr=False)
    arrived: np.ndarray = field(repr=False)
    actions: np.ndarray = field(repr=False)
    horizon: int
    active: np.ndarray | None = field(default=None, repr=False)
    semantic_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in _ARRAY_DTYPES:
            if name == "active":
                continue
            object.__setattr__(self, name, _owned_read_only_array(name, getattr(self, name)))
        active = self.active
        if active is None:
            active = np.logical_not(self.arrived)
        object.__setattr__(self, "active", _owned_read_only_array("active", active))
        if (
            isinstance(self.horizon, (bool, np.bool_))
            or not isinstance(self.horizon, (int, np.integer))
            or int(self.horizon) <= 0
        ):
            raise ValueError(f"horizon must be a positive integer, got {self.horizon!r}")
        object.__setattr__(self, "horizon", int(self.horizon))
        self.validate()
        object.__setattr__(self, "semantic_sha256", self._compute_semantic_sha256())

    @property
    def num_envs(self) -> int:
        return int(self.instance_ids.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.positions.shape[1])

    def validate(self) -> None:
        """Validate the complete standard-MAPF state and immutability contract."""

        expected_ranks = {
            "instance_ids": (1, "[E]"),
            "grids": (3, "[E, H, W]"),
            "positions": (3, "[E, A, 2]"),
            "goals": (3, "[E, A, 2]"),
            "arrived": (2, "[E, A]"),
            "active": (2, "[E, A]"),
            "actions": (3, "[T, E, A]"),
        }
        for name, (rank, notation) in expected_ranks.items():
            array = getattr(self, name)
            if array.dtype != _ARRAY_DTYPES[name]:
                raise ValueError(
                    f"{name} dtype must be {_ARRAY_DTYPES[name]}, got {array.dtype}"
                )
            if array.ndim != rank:
                raise ValueError(f"{name} shape must be {notation}, got {array.shape}")
            if not array.flags.c_contiguous:
                raise ValueError(f"{name} must use C-contiguous storage")
            if array.flags.writeable:
                raise ValueError(f"{name} must be read-only")
            if not _has_immutable_bytes_backing(array):
                raise ValueError(f"{name} must use immutable bytes backing")

        if self.positions.shape[-1] != 2:
            raise ValueError(
                f"positions shape must be [E, A, 2], got {self.positions.shape}"
            )
        if self.goals.shape[-1] != 2:
            raise ValueError(f"goals shape must be [E, A, 2], got {self.goals.shape}")

        num_envs = self.instance_ids.shape[0]
        if num_envs <= 0:
            raise ValueError("frozen transition batch must contain at least one environment")
        if self.grids.shape[0] != num_envs:
            raise ValueError("cross-field shape mismatch for grids environment dimension")
        num_agents = self.positions.shape[1]
        if num_agents <= 0:
            raise ValueError("frozen transition batch must contain at least one agent")
        expected_agent_shape = (num_envs, num_agents)
        if self.positions.shape != (num_envs, num_agents, 2):
            raise ValueError("cross-field shape mismatch for positions")
        if self.goals.shape != (num_envs, num_agents, 2):
            raise ValueError("cross-field shape mismatch for goals")
        if self.arrived.shape != expected_agent_shape:
            raise ValueError("cross-field shape mismatch for arrived")
        if self.active.shape != expected_agent_shape:
            raise ValueError("cross-field shape mismatch for active")
        if not np.array_equal(self.active, np.logical_not(self.arrived)):
            raise ValueError("active must equal logical_not(arrived) in standard_mapf")
        if self.actions.shape != (self.horizon, num_envs, num_agents):
            if self.actions.shape[0] != self.horizon:
                raise ValueError(
                    "horizon and actions length disagree: "
                    f"horizon={self.horizon}, actions={self.actions.shape[0]}"
                )
            raise ValueError("cross-field shape mismatch for actions")

        if np.unique(self.instance_ids).size != num_envs:
            raise ValueError("instance_ids must be unique")
        height, width = self.grids.shape[1:]
        if height <= 0 or width <= 0:
            raise ValueError("grid height and width must be positive")

        self._validate_coordinates("position", self.positions, height, width)
        self._validate_coordinates("goal", self.goals, height, width)
        for env_idx in range(num_envs):
            if np.unique(self.positions[env_idx], axis=0).shape[0] != num_agents:
                raise ValueError(
                    f"duplicate position in environment index {env_idx}"
                )
            position_cells = self.positions[env_idx]
            goal_cells = self.goals[env_idx]
            if np.any(self.grids[env_idx, position_cells[:, 0], position_cells[:, 1]] != 0):
                raise ValueError(f"position lies on a wall in environment index {env_idx}")
            if np.any(self.grids[env_idx, goal_cells[:, 0], goal_cells[:, 1]] != 0):
                raise ValueError(f"goal lies on a wall in environment index {env_idx}")

        if np.any(self.actions > 4):
            raise ValueError("actions must be in [0, 4]")
        on_goal = np.all(self.positions == self.goals, axis=-1)
        if np.any(self.arrived & ~on_goal):
            raise ValueError("arrived agents must be at their goals")

    @staticmethod
    def _validate_coordinates(
        label: str,
        coordinates: np.ndarray,
        height: int,
        width: int,
    ) -> None:
        if np.any(coordinates[..., 0] >= height) or np.any(coordinates[..., 1] >= width):
            raise ValueError(f"{label} coordinates are outside grid bounds")

    def _compute_semantic_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(SCHEMA_VERSION.encode("ascii"))
        for name in _HASH_FIELDS:
            if name == "horizon":
                value = np.asarray(self.horizon, dtype=np.int64)
            else:
                value = getattr(self, name)
            _hash_component(digest, name, value)
        return digest.hexdigest()

    def take_envs(self, env_count: int) -> "FrozenTransitionBatch":
        """Return the first ``env_count`` instances, preserving nested scans."""

        if (
            isinstance(env_count, (bool, np.bool_))
            or not isinstance(env_count, (int, np.integer))
            or not 1 <= int(env_count) <= self.num_envs
        ):
            raise ValueError(
                f"env_count must be an integer in [1, {self.num_envs}], got {env_count!r}"
            )
        count = int(env_count)
        return FrozenTransitionBatch(
            instance_ids=self.instance_ids[:count],
            grids=self.grids[:count],
            positions=self.positions[:count],
            goals=self.goals[:count],
            arrived=self.arrived[:count],
            active=self.active[:count],
            actions=self.actions[:, :count],
            horizon=self.horizon,
        )


def save_frozen_transition_batch(
    path: str | Path,
    batch: FrozenTransitionBatch,
) -> str:
    """Write a canonical batch plus integrity metadata to one NPZ artifact."""

    if not isinstance(batch, FrozenTransitionBatch):
        raise TypeError("batch must be a FrozenTransitionBatch")
    batch.validate()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "semantic_sha256": batch.semantic_sha256,
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as stream:
        np.savez_compressed(
            stream,
            instance_ids=batch.instance_ids,
            grids=batch.grids,
            positions=batch.positions,
            goals=batch.goals,
            arrived=batch.arrived,
            active=batch.active,
            actions=batch.actions,
            horizon=np.asarray(batch.horizon, dtype=np.int64),
            metadata_json=np.asarray(metadata_json),
        )
    return batch.semantic_sha256


def load_frozen_transition_batch(path: str | Path) -> FrozenTransitionBatch:
    """Load, validate, and verify a frozen transition artifact."""

    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as payload:
        if set(payload.files) != _NPZ_FIELDS:
            raise ValueError(
                f"frozen transition NPZ fields mismatch: expected {sorted(_NPZ_FIELDS)}, "
                f"got {sorted(payload.files)}"
            )
        try:
            metadata = json.loads(str(payload["metadata_json"].item()))
            horizon_array = payload["horizon"]
            if horizon_array.dtype != np.int64 or horizon_array.shape != ():
                raise ValueError(
                    "horizon payload must be a scalar int64, "
                    f"got dtype={horizon_array.dtype}, shape={horizon_array.shape}"
                )
            values = {name: payload[name].copy() for name in _ARRAY_DTYPES}
            horizon = int(horizon_array.item())
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid frozen transition payload: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError("frozen transition metadata must be a JSON object")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            "unsupported frozen transition schema version: "
            f"{metadata.get('schema_version')!r}"
        )
    stored_hash = metadata.get("semantic_sha256")
    if not isinstance(stored_hash, str):
        raise ValueError("frozen transition metadata is missing semantic_sha256")
    batch = FrozenTransitionBatch(**values, horizon=horizon)
    if batch.semantic_sha256 != stored_hash:
        raise ValueError(
            "frozen transition semantic SHA-256 mismatch: "
            f"expected {stored_hash}, got {batch.semantic_sha256}"
        )
    return batch


def build_frozen_pool(
    *,
    seeds: Sequence[int],
    horizon: int,
    instance_factory: Callable[..., Mapping[str, Any]],
) -> FrozenTransitionBatch:
    """Build a deterministic pool by invoking one injected factory per seed."""

    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("seeds must contain at least one explicit seed")
    if any(
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
        for seed in seed_list
    ):
        raise ValueError("seeds must contain only integers")
    seed_list = [int(seed) for seed in seed_list]
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be unique")
    int64 = np.iinfo(np.int64)
    if any(seed < int64.min or seed > int64.max for seed in seed_list):
        raise ValueError("seeds must fit in int64")
    if not callable(instance_factory):
        raise ValueError("instance_factory must be callable")

    instances = []
    required = {"grid", "positions", "goals", "arrived", "actions"}
    optional = {"active"}
    for seed in seed_list:
        instance = instance_factory(seed=seed, horizon=horizon)
        if not isinstance(instance, Mapping):
            raise ValueError("instance_factory must return a mapping")
        missing = required.difference(instance)
        extra = set(instance).difference(required | optional)
        if missing or extra:
            raise ValueError(
                "instance_factory fields mismatch: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        instances.append(instance)

    try:
        return FrozenTransitionBatch(
            instance_ids=np.asarray(seed_list, dtype=np.int64),
            grids=np.stack([item["grid"] for item in instances]),
            positions=np.stack([item["positions"] for item in instances]),
            goals=np.stack([item["goals"] for item in instances]),
            arrived=np.stack([item["arrived"] for item in instances]),
            active=np.stack(
                [item.get("active", np.logical_not(item["arrived"])) for item in instances]
            ),
            actions=np.stack([item["actions"] for item in instances], axis=1),
            horizon=horizon,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"instance_factory produced incompatible instances: {exc}") from exc


__all__ = [
    "ACTION_DELTAS",
    "FrozenTransitionBatch",
    "SCHEMA_VERSION",
    "build_frozen_pool",
    "load_frozen_transition_batch",
    "save_frozen_transition_batch",
]
