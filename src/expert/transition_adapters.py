"""Backend adapters for a frozen, transition-only standard-MAPF workload.

The adapters intentionally keep reset, synchronization policy, and diagnostic
materialization separate.  In particular, ``step_transition`` never builds an
observation, copies state back to the host, or classifies a collision.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from expert.benchmark_contract import ACTION_DELTAS, FrozenTransitionBatch


_CANONICAL_ARRAY_FIELDS = (
    "instance_ids",
    "grids",
    "positions",
    "goals",
    "arrived",
    "active",
    "actions",
)


@dataclass(frozen=True)
class _ActionToken:
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    writeable: bool
    base_pointer: int
    data_pointer: int
    pointer_offset: int


def _array_base_pointer(array: np.ndarray) -> int:
    owner = array
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    return int(owner.__array_interface__["data"][0])


def _action_token(array: np.ndarray, base_pointer: int) -> _ActionToken:
    pointer = int(array.__array_interface__["data"][0])
    return _ActionToken(
        dtype=array.dtype.str,
        shape=tuple(array.shape),
        strides=tuple(array.strides),
        writeable=bool(array.flags.writeable),
        base_pointer=_array_base_pointer(array),
        data_pointer=pointer,
        pointer_offset=pointer - base_pointer,
    )


def _action_chunk_token(array: np.ndarray, base_pointer: int) -> _ActionToken:
    return _action_token(array, base_pointer)


@dataclass(frozen=True)
class _PreparedActionHandle:
    """Opaque identity token created before a measured transition chunk."""

    generation: int
    step: int


def _frozen_array(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    backing = np.ascontiguousarray(array).tobytes(order="C")
    return np.frombuffer(backing, dtype=array.dtype).reshape(array.shape)


class _ArrayDataclassEquality:
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return all(
            np.array_equal(getattr(self, item.name), getattr(other, item.name))
            for item in fields(self)
        )


@dataclass(frozen=True, eq=False)
class CollisionOutcomes(_ArrayDataclassEquality):
    """Accumulated per-environment/per-agent transition outcome counts."""

    requested_moves: np.ndarray
    completed_moves: np.ndarray
    wall_rejections: np.ndarray
    vertex_rejections: np.ndarray
    edge_swap_rejections: np.ndarray
    chain_rejections: np.ndarray
    category_trace: np.ndarray
    semantic_mismatches: np.ndarray

    def __post_init__(self) -> None:
        summary_names = (
            "requested_moves",
            "completed_moves",
            "wall_rejections",
            "vertex_rejections",
            "edge_swap_rejections",
            "chain_rejections",
        )
        for name in summary_names:
            object.__setattr__(
                self,
                name,
                _frozen_array(getattr(self, name), np.int64),
            )
        object.__setattr__(
            self,
            "category_trace",
            _frozen_array(self.category_trace, np.uint8),
        )
        object.__setattr__(
            self,
            "semantic_mismatches",
            _frozen_array(self.semantic_mismatches, np.bool_),
        )
        summary_shape = self.requested_moves.shape
        if len(summary_shape) != 2 or any(
            getattr(self, name).shape != summary_shape for name in summary_names
        ):
            raise ValueError("collision summary arrays must share rank-2 shape [E, A]")
        trace_shape = self.category_trace.shape
        if (
            len(trace_shape) != 3
            or self.semantic_mismatches.shape != trace_shape
            or trace_shape[1:] != summary_shape
        ):
            raise ValueError(
                "collision trace arrays must share shape [T, E, A] aligned "
                "with summary shape [E, A]"
            )
        if np.any(self.category_trace > 4):
            raise ValueError("collision category trace values must be in [0, 4]")


@dataclass(frozen=True, eq=False)
class TransitionSnapshot(_ArrayDataclassEquality):
    """Host-side parity snapshot, materialized only after timing closes."""

    positions: np.ndarray
    goals: np.ndarray
    arrived: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    step_counts: np.ndarray
    collision_outcomes: CollisionOutcomes

    def __post_init__(self) -> None:
        expected = {
            "positions": np.uint16,
            "goals": np.uint16,
            "arrived": np.bool_,
            "terminated": np.bool_,
            "truncated": np.bool_,
            "step_counts": np.int32,
        }
        for name, dtype in expected.items():
            object.__setattr__(self, name, _frozen_array(getattr(self, name), dtype))
        if not isinstance(self.collision_outcomes, CollisionOutcomes):
            raise TypeError("collision_outcomes must be CollisionOutcomes")
        if self.positions.ndim != 3:
            raise ValueError("positions must have rank 3 and shape [E, A, 2]")
        if self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape [E, A, 2]")
        if self.goals.shape != self.positions.shape:
            raise ValueError("goals must have the same shape as positions")
        env_agent_shape = self.positions.shape[:2]
        if self.arrived.shape != env_agent_shape:
            raise ValueError("arrived must have shape [E, A]")
        env_shape = (self.positions.shape[0],)
        for name in ("terminated", "truncated", "step_counts"):
            if getattr(self, name).shape != env_shape:
                raise ValueError(f"{name} must have shape [E]")
        outcomes = self.collision_outcomes
        if outcomes.requested_moves.shape != env_agent_shape:
            raise ValueError("collision summary shape must match snapshot [E, A]")


def _materialize_array(value: Any) -> np.ndarray:
    """Convert a backend array to NumPy; never call this in a timed primitive."""

    module = type(value).__module__
    if module.startswith("torch"):
        return value.detach().cpu().numpy()
    try:
        import jax

        return np.asarray(jax.device_get(value))
    except (ImportError, TypeError):
        return np.asarray(value)


def _verify_backend_roundtrip(
    batch: FrozenTransitionBatch,
    *,
    instance_ids: Any,
    grids: Any,
    positions: Any,
    goals: Any,
    arrived: Any,
    active: Any,
    actions: Any,
) -> str:
    """Fail closed unless backend-consumed data round-trips to canonical bytes."""

    normalized = {
        "instance_ids": np.asarray(instance_ids, dtype=np.int64),
        "grids": np.asarray(grids, dtype=np.uint8),
        "positions": np.asarray(positions, dtype=np.uint16),
        "goals": np.asarray(goals, dtype=np.uint16),
        "arrived": np.asarray(arrived, dtype=np.bool_),
        "active": np.asarray(active, dtype=np.bool_),
        "actions": np.asarray(actions, dtype=np.uint8),
    }
    for name in _CANONICAL_ARRAY_FIELDS:
        canonical = getattr(batch, name)
        consumed = np.ascontiguousarray(normalized[name])
        if consumed.shape != canonical.shape or consumed.tobytes(order="C") != canonical.tobytes(order="C"):
            raise ValueError(
                f"backend-consumed {name} differs from canonical frozen bytes"
            )
    roundtrip = FrozenTransitionBatch(**normalized, horizon=batch.horizon)
    if roundtrip.semantic_sha256 != batch.semantic_sha256:
        raise ValueError(
            "backend-consumed dtype-normalized semantic SHA-256 differs from "
            "canonical frozen batch"
        )
    return roundtrip.semantic_sha256


def _verify_torch_device_consumption(
    batch: FrozenTransitionBatch,
    *,
    grids: Any,
    positions: Any,
    goals: Any,
    arrived: Any,
    actions: Any,
) -> None:
    """Compare actual Torch inputs with independent references on the device.

    Only the final scalar integrity result crosses back to the host.  It is reset
    validation metadata, not a state snapshot.
    """

    import torch

    actual = {
        "grids": grids,
        "positions": positions,
        "goals": goals,
        "arrived": arrived,
        "actions": actions,
    }
    host_reference = {
        "grids": np.array(batch.grids, dtype=np.int32, copy=True),
        "positions": np.array(batch.positions, dtype=np.int16, copy=True),
        "goals": np.array(batch.goals, dtype=np.int16, copy=True),
        "arrived": np.array(batch.arrived, dtype=np.uint8, copy=True),
        "actions": np.array(batch.actions, dtype=np.uint8, copy=True),
    }
    reference = {
        name: torch.as_tensor(value, device=grids.device).contiguous()
        for name, value in host_reference.items()
    }
    metadata_matches = all(
        value.shape == reference[name].shape
        and value.dtype == reference[name].dtype
        and value.device == reference[name].device
        for name, value in actual.items()
    )
    integrity = torch.tensor(metadata_matches, dtype=torch.bool, device=grids.device)
    if metadata_matches:
        for name, value in actual.items():
            integrity = torch.logical_and(
                integrity, torch.all(value == reference[name])
            )
    if not bool(integrity.item()):
        raise ValueError(
            "actual device input differs from canonical frozen transition batch"
        )


def _verify_jax_device_consumption(
    batch: FrozenTransitionBatch,
    *,
    grids: Any,
    positions: Any,
    goals: Any,
    arrived: Any,
    actions: tuple[Any, ...],
) -> None:
    """Compare actual JAX inputs with independent references on the device."""

    import jax
    import jax.numpy as jnp

    device = grids.device

    def upload(value: np.ndarray, dtype: Any):
        host = np.array(value, dtype=np.dtype(dtype), copy=True)
        return jax.device_put(jnp.asarray(host, dtype=dtype), device)

    reference = {
        "grids": upload(batch.grids, jnp.uint8),
        "positions": upload(batch.positions, jnp.uint16),
        "goals": upload(batch.goals, jnp.uint16),
        "arrived": upload(batch.arrived, jnp.bool_),
    }
    reference_actions = tuple(upload(step, jnp.uint8) for step in batch.actions)
    actual = {
        "grids": grids,
        "positions": positions,
        "goals": goals,
        "arrived": arrived,
    }
    metadata_matches = len(actions) == len(reference_actions) and all(
        value.shape == reference[name].shape
        and value.dtype == reference[name].dtype
        and value.device == reference[name].device
        for name, value in actual.items()
    )
    if metadata_matches:
        metadata_matches = all(
            value.shape == expected.shape
            and value.dtype == expected.dtype
            and value.device == expected.device
            for value, expected in zip(actions, reference_actions)
        )
    integrity = jax.device_put(jnp.asarray(metadata_matches, dtype=jnp.bool_), device)
    if metadata_matches:
        for name, value in actual.items():
            integrity = jnp.logical_and(
                integrity, jnp.all(value == reference[name])
            )
        for value, expected in zip(actions, reference_actions):
            integrity = jnp.logical_and(integrity, jnp.all(value == expected))
    if not bool(jax.device_get(integrity)):
        raise ValueError(
            "actual device input differs from canonical frozen transition batch"
        )


@dataclass(frozen=True)
class _ProposalResolution:
    positions: np.ndarray
    categories: np.ndarray
    requested: np.ndarray
    operation_count: int


def _resolve_proposals(
    grid: np.ndarray,
    current: np.ndarray,
    arrived: np.ndarray,
    actions: np.ndarray,
) -> _ProposalResolution:
    """Resolve one environment with event-indexed CUDA-compatible semantics."""

    current = np.asarray(current, dtype=np.int32)
    arrived = np.asarray(arrived, dtype=np.bool_)
    actions = np.asarray(actions, dtype=np.uint8)
    num_agents = current.shape[0]
    active = ~arrived
    effective_actions = np.where(active, actions, 0).astype(np.int64, copy=False)
    requested = active & (effective_actions != 0)
    raw_proposed = current + ACTION_DELTAS[effective_actions].astype(np.int32)
    height, width = grid.shape
    out_of_bounds = (
        (raw_proposed[:, 0] < 0)
        | (raw_proposed[:, 0] >= height)
        | (raw_proposed[:, 1] < 0)
        | (raw_proposed[:, 1] >= width)
    )
    safe_x = np.clip(raw_proposed[:, 0], 0, height - 1)
    safe_y = np.clip(raw_proposed[:, 1], 0, width - 1)
    hit_wall = np.asarray(grid)[safe_x, safe_y] != 0
    wall_rejected = requested & (out_of_bounds | hit_wall)
    proposed = np.where(
        (out_of_bounds | hit_wall)[:, None], current, raw_proposed
    )
    categories = np.zeros(num_agents, dtype=np.uint8)
    categories[wall_rejected] = 1
    current_owner = {
        (int(position[0]), int(position[1])): agent_idx
        for agent_idx, position in enumerate(current)
    }
    operation_count = num_agents

    for _ in range(num_agents):
        destination_min_id: dict[tuple[int, int], int] = {}
        for agent_idx, destination in enumerate(proposed):
            cell = (int(destination[0]), int(destination[1]))
            destination_min_id[cell] = min(
                agent_idx, destination_min_id.get(cell, agent_idx)
            )
            operation_count += 1

        rejected: list[tuple[int, int]] = []
        for agent_idx in range(num_agents):
            operation_count += 1
            if np.array_equal(proposed[agent_idx], current[agent_idx]):
                continue
            destination = (
                int(proposed[agent_idx, 0]),
                int(proposed[agent_idx, 1]),
            )
            candidates: set[int] = set()
            occupant = current_owner.get(destination)
            if occupant is not None and occupant != agent_idx:
                candidates.add(occupant)
            vertex_candidate = destination_min_id[destination]
            if vertex_candidate < agent_idx:
                candidates.add(vertex_candidate)

            category = 0
            for other_idx in sorted(candidates):
                operation_count += 1
                if (
                    np.array_equal(proposed[agent_idx], current[other_idx])
                    and np.array_equal(proposed[other_idx], current[agent_idx])
                ):
                    category = 3
                elif agent_idx > other_idx and np.array_equal(
                    proposed[agent_idx], proposed[other_idx]
                ):
                    category = 2
                elif (
                    np.array_equal(proposed[agent_idx], current[other_idx])
                    and np.array_equal(proposed[other_idx], current[other_idx])
                ):
                    category = 4
                if category:
                    break
            if category:
                rejected.append((agent_idx, category))

        if not rejected:
            break
        for agent_idx, category in rejected:
            proposed[agent_idx] = current[agent_idx]
            if categories[agent_idx] == 0:
                categories[agent_idx] = category

    return _ProposalResolution(
        positions=np.asarray(proposed, dtype=np.int32),
        categories=categories,
        requested=requested,
        operation_count=operation_count,
    )


@dataclass(frozen=True)
class _BackendState:
    positions: np.ndarray
    goals: np.ndarray
    arrived: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    step_counts: np.ndarray

    def __post_init__(self) -> None:
        expected = {
            "positions": np.uint16,
            "goals": np.uint16,
            "arrived": np.bool_,
            "terminated": np.bool_,
            "truncated": np.bool_,
            "step_counts": np.int32,
        }
        for name, dtype in expected.items():
            object.__setattr__(self, name, _frozen_array(getattr(self, name), dtype))


def _state_mismatches(actual: _BackendState, expected: _BackendState) -> np.ndarray:
    mismatch = np.any(actual.positions != expected.positions, axis=-1)
    mismatch |= np.any(actual.goals != expected.goals, axis=-1)
    mismatch |= actual.arrived != expected.arrived
    env_mismatch = (
        (actual.terminated != expected.terminated)
        | (actual.truncated != expected.truncated)
        | (actual.step_counts != expected.step_counts)
    )
    return mismatch | env_mismatch[:, None]


class _TransitionAdapter:
    _backend_name = "abstract"

    def __init__(
        self,
        batch: FrozenTransitionBatch,
        device: str,
        *,
        verify_consumption: bool = True,
    ):
        if not isinstance(batch, FrozenTransitionBatch):
            raise TypeError("batch must be a FrozenTransitionBatch")
        if verify_consumption:
            batch.validate()
        self.batch = batch
        self.device = device
        self.verify_consumption = bool(verify_consumption)
        self._cursor = 0
        self._ready = False
        self._consumed_input_sha256: str | None = None
        self._prepare_generation = 0
        self._prepared_start_cursor: int | None = None
        self._prepared_handles: tuple[_PreparedActionHandle, ...] = ()
        base_pointer = int(batch.actions.__array_interface__["data"][0])
        self._action_tokens = tuple(
            _action_token(batch.actions[step], base_pointer)
            for step in range(batch.horizon)
        )
        self._action_base_pointer = base_pointer

    @property
    def consumed_input_sha256(self) -> str:
        self._require_ready("consumed_input_sha256")
        assert self._consumed_input_sha256 is not None
        return self._consumed_input_sha256

    def _require_ready(self, operation: str) -> None:
        if not self._ready:
            raise RuntimeError(
                f"transition adapter is not ready for {operation}; "
                "call reset_device_state() successfully first"
            )

    def _validate_action_token(self, actions: np.ndarray) -> None:
        if not isinstance(actions, np.ndarray):
            raise ValueError("prepare_action requires the canonical action token")
        observed = _action_token(actions, self._action_base_pointer)
        if observed != self._action_tokens[self._cursor]:
            raise ValueError(
                "prepare_action requires the current canonical action token"
            )

    def _bind_prepared_actions(
        self, count: int
    ) -> tuple[_PreparedActionHandle, ...]:
        self._prepare_generation += 1
        generation = self._prepare_generation
        prepared = tuple(
            _PreparedActionHandle(generation=generation, step=self._cursor + index)
            for index in range(count)
        )
        self._prepared_start_cursor = self._cursor
        self._prepared_handles = prepared
        return prepared

    def prepare_actions(
        self, actions: np.ndarray
    ) -> tuple[_PreparedActionHandle, ...]:
        """Validate and bind one continuous canonical chunk outside timing."""

        self._require_ready("prepare_actions")
        self._check_step_available()
        if (
            isinstance(actions, np.ndarray)
            and actions.ndim == 3
            and actions.shape[0] == 0
        ):
            raise ValueError(
                "prepare_actions requires a non-empty canonical action chunk"
            )
        if not isinstance(actions, np.ndarray):
            raise ValueError("prepare_actions requires the canonical action chunk")
        observed = _action_chunk_token(actions, self._action_base_pointer)
        count = int(actions.shape[0]) if actions.ndim == 3 else 0
        expected_pointer = (
            self._action_base_pointer
            + self._cursor * self.batch.actions.strides[0]
        )
        if (
            actions.ndim != 3
            or count <= 0
            or count > self.batch.horizon - self._cursor
            or observed.dtype != self.batch.actions.dtype.str
            or observed.shape
            != (count, self.batch.num_envs, self.batch.num_agents)
            or observed.strides != tuple(self.batch.actions.strides)
            or observed.writeable
            or observed.base_pointer != self._action_base_pointer
            or observed.data_pointer != expected_pointer
            or observed.pointer_offset
            != self._cursor * self.batch.actions.strides[0]
        ):
            raise ValueError(
                "prepare_actions requires a continuous canonical action chunk "
                "starting at the current cursor"
            )
        return self._bind_prepared_actions(count)

    def prepare_action(self, actions: np.ndarray) -> _PreparedActionHandle:
        """Validate and bind one canonical slice for latency measurement."""

        self._require_ready("prepare_action")
        self._check_step_available()
        self._validate_action_token(actions)
        return self._bind_prepared_actions(1)[0]

    def _consume_prepared_action(self, handle: _PreparedActionHandle) -> None:
        prepared_index = (
            self._cursor - self._prepared_start_cursor
            if self._prepared_start_cursor is not None
            else -1
        )
        if (
            prepared_index < 0
            or prepared_index >= len(self._prepared_handles)
            or handle is not self._prepared_handles[prepared_index]
        ):
            raise RuntimeError(
                "step_transition requires the prepared action identity for "
                "the current cursor"
            )
        if prepared_index + 1 == len(self._prepared_handles):
            self._prepared_start_cursor = None
            self._prepared_handles = ()

    def _check_step_available(self) -> None:
        if self._cursor >= self.batch.horizon:
            raise RuntimeError("frozen action horizon is exhausted; reset the adapter")

    def materialize_snapshot(self) -> TransitionSnapshot:
        self._require_ready("materialize_snapshot")
        final_state = self.materialize_state()
        outcomes = self.diagnose_replay(final_state)
        return TransitionSnapshot(
            positions=final_state.positions,
            goals=final_state.goals,
            arrived=final_state.arrived,
            terminated=final_state.terminated,
            truncated=final_state.truncated,
            step_counts=final_state.step_counts,
            collision_outcomes=outcomes,
        )

    def materialize_state(self) -> _BackendState:
        """Capture state without replaying the full action history for diagnostics."""

        self._require_ready("materialize_state")
        return self._materialize_backend_state()

    def diagnose_replay(self, captured_final_state: _BackendState) -> CollisionOutcomes:
        """Run one complete semantic replay against a captured terminal state."""

        self._require_ready("diagnose_replay")
        if not isinstance(captured_final_state, _BackendState):
            raise TypeError("diagnose_replay requires a backend state captured by materialize_state")
        return self._diagnose_backend_replay(captured_final_state)

    def _predict_backend_step(
        self, before: _BackendState, step: int
    ) -> tuple[_BackendState, np.ndarray, np.ndarray]:
        positions = np.array(before.positions, dtype=np.int32, copy=True)
        arrived = np.array(before.arrived, dtype=np.bool_, copy=True)
        terminated = np.array(before.terminated, dtype=np.bool_, copy=True)
        truncated = np.array(before.truncated, dtype=np.bool_, copy=True)
        step_counts = np.array(before.step_counts, dtype=np.int32, copy=True)
        categories = np.zeros(
            (self.batch.num_envs, self.batch.num_agents), dtype=np.uint8
        )
        requested = np.zeros_like(categories, dtype=np.bool_)
        for env_idx in range(self.batch.num_envs):
            if terminated[env_idx] or truncated[env_idx]:
                continue
            resolution = _resolve_proposals(
                self.batch.grids[env_idx],
                positions[env_idx],
                arrived[env_idx],
                self.batch.actions[step, env_idx],
            )
            positions[env_idx] = resolution.positions
            categories[env_idx] = resolution.categories
            requested[env_idx] = resolution.requested
            arrived[env_idx] |= np.all(
                resolution.positions == self.batch.goals[env_idx], axis=1
            )
            step_counts[env_idx] += 1
            terminated[env_idx] = bool(np.all(arrived[env_idx]))
            truncated[env_idx] = bool(
                step_counts[env_idx] >= self.batch.horizon
                and not terminated[env_idx]
            )
        return (
            _BackendState(
                positions=positions,
                goals=self.batch.goals,
                arrived=arrived,
                terminated=terminated,
                truncated=truncated,
                step_counts=step_counts,
            ),
            categories,
            requested,
        )

    def _diagnose_backend_replay(
        self, captured_final_state: _BackendState
    ) -> CollisionOutcomes:
        consumed_steps = self._cursor
        shape = (self.batch.num_envs, self.batch.num_agents)
        counts = {
            "requested_moves": np.zeros(shape, dtype=np.int64),
            "completed_moves": np.zeros(shape, dtype=np.int64),
            "wall_rejections": np.zeros(shape, dtype=np.int64),
            "vertex_rejections": np.zeros(shape, dtype=np.int64),
            "edge_swap_rejections": np.zeros(shape, dtype=np.int64),
            "chain_rejections": np.zeros(shape, dtype=np.int64),
        }
        category_trace = np.zeros((consumed_steps, *shape), dtype=np.uint8)
        semantic_mismatches = np.zeros_like(category_trace, dtype=np.bool_)
        if consumed_steps == 0:
            return CollisionOutcomes(
                **counts,
                category_trace=category_trace,
                semantic_mismatches=semantic_mismatches,
            )

        self.reset_device_state()
        before = self._materialize_backend_state()
        canonical_reset = _BackendState(
            positions=self.batch.positions,
            goals=self.batch.goals,
            arrived=self.batch.arrived,
            terminated=np.all(self.batch.arrived, axis=1),
            truncated=np.zeros(self.batch.num_envs, dtype=np.bool_),
            step_counts=np.zeros(self.batch.num_envs, dtype=np.int32),
        )
        reset_mismatch = _state_mismatches(before, canonical_reset)
        prepared = self.prepare_actions(self.batch.actions[:consumed_steps])

        for step in range(consumed_steps):
            predicted, categories, requested = self._predict_backend_step(
                before, step
            )
            category_trace[step] = categories
            counts["requested_moves"] += requested
            counts["wall_rejections"] += categories == 1
            counts["vertex_rejections"] += categories == 2
            counts["edge_swap_rejections"] += categories == 3
            counts["chain_rejections"] += categories == 4

            self.step_transition(prepared[step])
            self.synchronize_transition()
            after = self._materialize_backend_state()
            counts["completed_moves"] += requested & np.any(
                after.positions != before.positions, axis=-1
            )
            semantic_mismatches[step] = _state_mismatches(after, predicted)
            if step == 0:
                semantic_mismatches[step] |= reset_mismatch
            before = after

        semantic_mismatches[-1] |= _state_mismatches(
            before, captured_final_state
        )
        return CollisionOutcomes(
            **counts,
            category_trace=category_trace,
            semantic_mismatches=semantic_mismatches,
        )


class _PogemaTransitionAdapter(_TransitionAdapter):
    _backend_name = "pogema"

    def __init__(
        self,
        batch: FrozenTransitionBatch,
        device: str,
        *,
        verify_consumption: bool = True,
    ):
        if device != "cpu":
            raise ValueError("POGEMA transition adapter requires device='cpu'")
        super().__init__(
            batch,
            device,
            verify_consumption=verify_consumption,
        )
        self._envs: list[Any] = []

    def reset_device_state(self) -> None:
        (
            self._ready,
            self._consumed_input_sha256,
            self._prepare_generation,
            self._prepared_start_cursor,
            self._prepared_handles,
        ) = (False, None, self._prepare_generation + 1, None, ())
        from pogema.envs import PogemaCoopFinish
        from pogema.grid_config import GridConfig

        self._envs = []
        for env_idx in range(self.batch.num_envs):
            config = GridConfig(
                map=self.batch.grids[env_idx].tolist(),
                agents_xy=self.batch.positions[env_idx].tolist(),
                targets_xy=self.batch.goals[env_idx].tolist(),
                num_agents=self.batch.num_agents,
                on_target="nothing",
                collision_system="soft",
                observation_type="MAPF",
                max_episode_steps=self.batch.horizon,
                empty_outside=True,
            )
            env = PogemaCoopFinish(grid_config=config)
            env._initialize_grid()
            env.update_was_on_goal()
            self._envs.append(env)

        self._mutable_actions = np.array(self.batch.actions, dtype=np.uint8, copy=True)
        self._arrived = np.array(self.batch.arrived, dtype=np.bool_, copy=True)
        self._terminated = np.all(self._arrived, axis=1)
        self._truncated = np.zeros(self.batch.num_envs, dtype=np.bool_)
        self._step_counts = np.zeros(self.batch.num_envs, dtype=np.int32)
        self._cursor = 0
        if self.verify_consumption:
            consumed_positions = np.asarray(
                [env.grid.get_agents_xy(ignore_borders=True) for env in self._envs],
                dtype=np.uint16,
            )
            consumed_goals = np.asarray(
                [env.grid.get_targets_xy(ignore_borders=True) for env in self._envs],
                dtype=np.uint16,
            )
            consumed_grids = np.asarray(
                [env.grid.get_obstacles(ignore_borders=True) for env in self._envs],
                dtype=np.uint8,
            )
            consumed_hash = _verify_backend_roundtrip(
                self.batch,
                instance_ids=self.batch.instance_ids,
                grids=consumed_grids,
                positions=consumed_positions,
                goals=consumed_goals,
                arrived=self._arrived,
                active=self.batch.active,
                actions=self._mutable_actions,
            )
        else:
            consumed_hash = self.batch.semantic_sha256
        self._consumed_input_sha256, self._ready = consumed_hash, True

    def step_transition(self, handle: _PreparedActionHandle) -> None:
        self._require_ready("step_transition")
        self._check_step_available()
        self._consume_prepared_action(handle)
        for env_idx, env in enumerate(self._envs):
            if self._terminated[env_idx] or self._truncated[env_idx]:
                continue
            mutable = self._mutable_actions[self._cursor, env_idx]
            mutable[self._arrived[env_idx]] = 0
            env.move_agents(mutable)
            env.update_was_on_goal()
            all_arrived = True
            for agent_idx in range(self.batch.num_agents):
                is_arrived = bool(self._arrived[env_idx, agent_idx]) or bool(
                    env.was_on_goal[agent_idx]
                )
                self._arrived[env_idx, agent_idx] = is_arrived
                all_arrived = all_arrived and is_arrived
            self._step_counts[env_idx] += 1
            self._terminated[env_idx] = all_arrived
            if (
                self._step_counts[env_idx] >= self.batch.horizon
                and not self._terminated[env_idx]
            ):
                self._truncated[env_idx] = True
        self._cursor += 1

    def synchronize_transition(self) -> None:
        self._require_ready("synchronize_transition")
        return None

    def _materialize_backend_state(self) -> _BackendState:
        positions = np.asarray(
            [env.grid.get_agents_xy(ignore_borders=True) for env in self._envs],
            dtype=np.uint16,
        )
        goals = np.asarray(
            [env.grid.get_targets_xy(ignore_borders=True) for env in self._envs],
            dtype=np.uint16,
        )
        return _BackendState(
            positions=positions,
            goals=goals,
            arrived=self._arrived,
            terminated=self._terminated,
            truncated=self._truncated,
            step_counts=self._step_counts,
        )


class _CudaTransitionAdapter(_TransitionAdapter):
    _backend_name = "cuda"

    def __init__(
        self,
        batch: FrozenTransitionBatch,
        device: str,
        *,
        verify_consumption: bool = True,
    ):
        if not str(device).startswith("cuda"):
            raise ValueError("CUDA transition adapter requires a CUDA device")
        super().__init__(
            batch,
            device,
            verify_consumption=verify_consumption,
        )

    def reset_device_state(self) -> None:
        (
            self._ready,
            self._consumed_input_sha256,
            self._prepare_generation,
            self._prepared_start_cursor,
            self._prepared_handles,
        ) = (False, None, self._prepare_generation + 1, None, ())
        import torch
        # Loading PyTorch first makes its shared libraries (for example
        # libc10.so) available to the in-place extension.
        import grid_world_cpp as ext

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        compiled_shape = (int(ext.COMPILED_MAP_W), int(ext.COMPILED_MAP_H))
        if self.batch.grids.shape[1:] != compiled_shape:
            raise ValueError(
                f"CUDA grid shape must be {compiled_shape}, got {self.batch.grids.shape[1:]}"
            )
        self._torch = torch
        host_grids = np.array(self.batch.grids, dtype=np.int32, copy=True)
        host_positions = np.array(self.batch.positions, dtype=np.int16, copy=True)
        host_goals = np.array(self.batch.goals, dtype=np.int16, copy=True)
        host_arrived = np.array(self.batch.arrived, dtype=np.uint8, copy=True)
        host_actions = np.array(self.batch.actions, dtype=np.uint8, copy=True)
        host_roundtrip_hash = self.batch.semantic_sha256
        if self.verify_consumption:
            host_roundtrip_hash = _verify_backend_roundtrip(
                self.batch,
                instance_ids=np.array(self.batch.instance_ids, dtype=np.int64, copy=True),
                grids=host_grids,
                positions=host_positions,
                goals=host_goals,
                arrived=host_arrived,
                active=np.array(self.batch.active, dtype=np.uint8, copy=True),
                actions=host_actions,
            )
        self._grids = torch.as_tensor(
            host_grids, device=self.device
        ).contiguous()
        self._positions = torch.as_tensor(
            host_positions, device=self.device,
        ).contiguous()
        self._goals = torch.as_tensor(
            host_goals, device=self.device
        ).contiguous()
        self._arrived = torch.as_tensor(
            host_arrived, device=self.device,
        ).contiguous()
        self._actions = torch.as_tensor(
            host_actions, device=self.device,
        ).contiguous()
        if self.verify_consumption:
            _verify_torch_device_consumption(
                self.batch,
                grids=self._grids,
                positions=self._positions,
                goals=self._goals,
                arrived=self._arrived,
                actions=self._actions,
            )
        self._step_counts = torch.zeros(
            (self.batch.num_envs,), dtype=torch.int32, device=self.device
        )
        pool_capacity = (
            self.batch.num_envs * compiled_shape[0] * compiled_shape[1]
        )
        self._simulator = ext.GridWorldSimulator(
            self._grids,
            self.batch.num_agents,
            0,
            pool_capacity,
            0,
            "standard_mapf",
            self.batch.horizon,
        )
        self._simulator.load_state(
            self._positions, self._goals, self._arrived, self._step_counts
        )
        self._synchronize_backend()
        self._cursor = 0
        self._consumed_input_sha256, self._ready = host_roundtrip_hash, True

    def step_transition(self, handle: _PreparedActionHandle) -> None:
        self._require_ready("step_transition")
        self._check_step_available()
        self._consume_prepared_action(handle)
        self._simulator.update_actions(self._actions[self._cursor])
        self._simulator.step_sim_only()
        self._cursor += 1

    def synchronize_transition(self) -> None:
        self._require_ready("synchronize_transition")
        self._synchronize_backend()

    def _synchronize_backend(self) -> None:
        self._torch.cuda.synchronize(self.device)

    def _materialize_backend_state(self) -> _BackendState:
        positions = np.stack(
            (
                _materialize_array(self._simulator.cur_x),
                _materialize_array(self._simulator.cur_y),
            ),
            axis=-1,
        ).astype(np.uint16, copy=False)
        goals = np.stack(
            (
                _materialize_array(self._simulator.goal_x),
                _materialize_array(self._simulator.goal_y),
            ),
            axis=-1,
        ).astype(np.uint16, copy=False)
        return _BackendState(
            positions=positions,
            goals=goals,
            arrived=_materialize_array(self._simulator.arrived).astype(np.bool_),
            terminated=_materialize_array(self._simulator.terminated).astype(
                np.bool_
            ),
            truncated=_materialize_array(self._simulator.truncated).astype(np.bool_),
            step_counts=_materialize_array(self._simulator.step_counts),
        )


class _JaxTransitionAdapter(_TransitionAdapter):
    _backend_name = "jax"

    def __init__(
        self,
        batch: FrozenTransitionBatch,
        device: str,
        *,
        verify_consumption: bool = True,
    ):
        if not str(device).startswith("cuda"):
            raise ValueError("JAX transition adapter requires a CUDA device")
        super().__init__(
            batch,
            device,
            verify_consumption=verify_consumption,
        )

    def reset_device_state(self) -> None:
        (
            self._ready,
            self._consumed_input_sha256,
            self._prepare_generation,
            self._prepared_start_cursor,
            self._prepared_handles,
        ) = (False, None, self._prepare_generation + 1, None, ())
        import jax
        import jax.numpy as jnp

        from baselines.jax_sim_baseline import EnvConfig, reset_frozen_jit

        gpu_devices = jax.devices("gpu")
        device_index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
        if device_index >= len(gpu_devices):
            raise RuntimeError(f"JAX GPU device {self.device!r} is unavailable")
        self._jax = jax
        self._jax_device = gpu_devices[device_index]
        self._config = EnvConfig(
            height=self.batch.grids.shape[1],
            width=self.batch.grids.shape[2],
            num_agents=self.batch.num_agents,
            task_mode="standard_mapf",
            max_episode_steps=self.batch.horizon,
        )

        host_grids = np.array(self.batch.grids, dtype=np.uint8, copy=True)
        host_positions = np.array(self.batch.positions, dtype=np.uint16, copy=True)
        host_goals = np.array(self.batch.goals, dtype=np.uint16, copy=True)
        host_arrived = np.array(self.batch.arrived, dtype=np.bool_, copy=True)
        host_actions = np.array(self.batch.actions, dtype=np.uint8, copy=True)
        host_roundtrip_hash = self.batch.semantic_sha256
        if self.verify_consumption:
            host_roundtrip_hash = _verify_backend_roundtrip(
                self.batch,
                instance_ids=np.array(self.batch.instance_ids, dtype=np.int64, copy=True),
                grids=host_grids,
                positions=host_positions,
                goals=host_goals,
                arrived=host_arrived,
                active=np.array(self.batch.active, dtype=np.bool_, copy=True),
                actions=host_actions,
            )

        def put(value: np.ndarray, dtype: Any):
            return jax.device_put(jnp.asarray(value, dtype=dtype), self._jax_device)

        self._grids = put(host_grids, jnp.uint8)
        self._positions = put(host_positions, jnp.uint16)
        self._goals = put(host_goals, jnp.uint16)
        self._arrived = put(host_arrived, jnp.bool_)
        self._actions_by_step = tuple(
            put(host_actions[step], jnp.uint8) for step in range(self.batch.horizon)
        )
        if self.verify_consumption:
            _verify_jax_device_consumption(
                self.batch,
                grids=self._grids,
                positions=self._positions,
                goals=self._goals,
                arrived=self._arrived,
                actions=self._actions_by_step,
            )
        self._rng_keys = jax.device_put(
            jnp.zeros((self.batch.num_envs, 2), dtype=jnp.uint32),
            self._jax_device,
        )
        self._state = reset_frozen_jit(
            self._config,
            self._positions,
            self._goals,
            self._arrived,
            self._grids,
            self._rng_keys,
        )
        self._synchronize_backend()
        self._cursor = 0
        self._consumed_input_sha256, self._ready = host_roundtrip_hash, True

    def step_transition(self, handle: _PreparedActionHandle) -> None:
        from baselines.jax_sim_baseline import step_transition_jit

        self._require_ready("step_transition")
        self._check_step_available()
        self._consume_prepared_action(handle)
        outputs = step_transition_jit(
            self._config,
            self._state,
            self._actions_by_step[self._cursor],
            self._grids,
        )
        self._state = outputs[0]
        self._cursor += 1

    def synchronize_transition(self) -> None:
        self._require_ready("synchronize_transition")
        self._synchronize_backend()

    def _synchronize_backend(self) -> None:
        from baselines.jax_sim_baseline import block_transition_ready

        block_transition_ready(self._state)

    def _materialize_backend_state(self) -> _BackendState:
        return _BackendState(
            positions=_materialize_array(self._state.pos),
            goals=_materialize_array(self._state.target),
            arrived=_materialize_array(self._state.arrived),
            terminated=_materialize_array(self._state.terminated),
            truncated=_materialize_array(self._state.truncated),
            step_counts=_materialize_array(self._state.step_counts),
        )


def make_transition_adapter(
    name: str,
    batch: FrozenTransitionBatch,
    device: str,
    *,
    verify_consumption: bool = True,
    **unsupported: Any,
) -> _TransitionAdapter:
    """Create an adapter from explicit frozen input only.

    Seeds and random-generation controls are deliberately unsupported: all
    benchmark semantics must already be represented by ``batch``.
    """

    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(f"unsupported transition adapter argument(s): {names}")
    adapters = {
        "pogema": _PogemaTransitionAdapter,
        "cuda": _CudaTransitionAdapter,
        "jax": _JaxTransitionAdapter,
    }
    try:
        adapter_type = adapters[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown transition adapter {name!r}; expected one of {sorted(adapters)}"
        ) from exc
    return adapter_type(
        batch,
        device,
        verify_consumption=verify_consumption,
    )


__all__ = [
    "CollisionOutcomes",
    "TransitionSnapshot",
    "make_transition_adapter",
]
