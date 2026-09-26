"""Contract and semantic-parity tests for frozen transition adapters."""

from __future__ import annotations

import builtins
import hashlib

import numpy as np
import pytest

from expert.benchmark_contract import FrozenTransitionBatch
from expert.transition_adapters import (
    CollisionOutcomes,
    TransitionSnapshot,
    make_transition_adapter,
)


MAP_SIZE = 128


def _batch(
    *,
    positions,
    goals,
    actions,
    grids=None,
    arrived=None,
    horizon=None,
) -> FrozenTransitionBatch:
    positions = np.asarray(positions, dtype=np.uint16)
    goals = np.asarray(goals, dtype=np.uint16)
    if positions.ndim == 2:
        positions = positions[None]
    if goals.ndim == 2:
        goals = goals[None]
    num_envs, num_agents, _ = positions.shape
    actions = np.asarray(actions, dtype=np.uint8)
    if actions.ndim == 1:
        actions = actions[None, None]
    elif actions.ndim == 2:
        actions = actions[:, None]
    if horizon is None:
        horizon = actions.shape[0]
    if grids is None:
        grids = np.zeros((num_envs, MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    else:
        grids = np.asarray(grids, dtype=np.uint8)
        if grids.ndim == 2:
            grids = grids[None]
    if arrived is None:
        arrived = np.zeros((num_envs, num_agents), dtype=np.bool_)
    return FrozenTransitionBatch(
        instance_ids=np.arange(num_envs, dtype=np.int64),
        grids=grids,
        positions=positions,
        goals=goals,
        arrived=np.asarray(arrived, dtype=np.bool_),
        actions=actions,
        horizon=horizon,
    )


def _snapshot(adapter, batch: FrozenTransitionBatch) -> TransitionSnapshot:
    adapter.reset_device_state()
    prepared = adapter.prepare_actions(batch.actions)
    for handle in prepared:
        adapter.step_transition(handle)
    adapter.synchronize_transition()
    return adapter.materialize_snapshot()


def _stepwise_snapshots(adapter, batch: FrozenTransitionBatch) -> list[TransitionSnapshot]:
    adapter.reset_device_state()
    snapshots: list[TransitionSnapshot] = []
    for step in range(batch.horizon):
        prepared = adapter.prepare_action(batch.actions[step])
        adapter.step_transition(prepared)
        adapter.synchronize_transition()
        snapshots.append(adapter.materialize_snapshot())
    return snapshots


def _exact_bytes(batch: FrozenTransitionBatch) -> str:
    digest = hashlib.sha256()
    for name in ("instance_ids", "grids", "positions", "goals", "arrived", "actions"):
        digest.update(getattr(batch, name).tobytes(order="C"))
    digest.update(np.asarray(batch.horizon, dtype=np.int64).tobytes())
    return digest.hexdigest()


def _zero_outcomes(*, steps=1, num_envs=1, num_agents=1):
    summary = np.zeros((num_envs, num_agents), dtype=np.int64)
    return CollisionOutcomes(
        requested_moves=summary,
        completed_moves=summary,
        wall_rejections=summary,
        vertex_rejections=summary,
        edge_swap_rejections=summary,
        chain_rejections=summary,
        category_trace=np.zeros((steps, num_envs, num_agents), dtype=np.uint8),
        semantic_mismatches=np.zeros(
            (steps, num_envs, num_agents), dtype=np.bool_
        ),
    )


def test_snapshot_and_collision_outcomes_validate_rank_and_shape_strictly():
    with pytest.raises(ValueError, match="summary.*shape"):
        CollisionOutcomes(
            requested_moves=np.zeros((2, 1), dtype=np.int64),
            completed_moves=np.zeros((1, 1), dtype=np.int64),
            wall_rejections=np.zeros((2, 1), dtype=np.int64),
            vertex_rejections=np.zeros((2, 1), dtype=np.int64),
            edge_swap_rejections=np.zeros((2, 1), dtype=np.int64),
            chain_rejections=np.zeros((2, 1), dtype=np.int64),
            category_trace=np.zeros((1, 2, 1), dtype=np.uint8),
            semantic_mismatches=np.zeros((1, 2, 1), dtype=np.bool_),
        )

    with pytest.raises(ValueError, match="trace.*shape"):
        CollisionOutcomes(
            requested_moves=np.zeros((1, 1), dtype=np.int64),
            completed_moves=np.zeros((1, 1), dtype=np.int64),
            wall_rejections=np.zeros((1, 1), dtype=np.int64),
            vertex_rejections=np.zeros((1, 1), dtype=np.int64),
            edge_swap_rejections=np.zeros((1, 1), dtype=np.int64),
            chain_rejections=np.zeros((1, 1), dtype=np.int64),
            category_trace=np.zeros((1, 1, 1), dtype=np.uint8),
            semantic_mismatches=np.zeros((2, 1, 1), dtype=np.bool_),
        )

    with pytest.raises(ValueError, match="positions.*rank"):
        TransitionSnapshot(
            positions=np.zeros((1, 2), dtype=np.uint16),
            goals=np.zeros((1, 1, 2), dtype=np.uint16),
            arrived=np.zeros((1, 1), dtype=np.bool_),
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
            step_counts=np.zeros((1,), dtype=np.int32),
            collision_outcomes=_zero_outcomes(),
        )


def test_event_resolver_avoids_cubic_work_for_512_agent_chain():
    import expert.transition_adapters as module

    num_agents = 512
    current = np.stack(
        (
            np.zeros(num_agents, dtype=np.int32),
            np.arange(num_agents, dtype=np.int32),
        ),
        axis=-1,
    )
    grid = np.zeros((1, num_agents + 1), dtype=np.uint8)
    actions = np.full(num_agents, 4, dtype=np.uint8)
    actions[-1] = 0
    resolved = module._resolve_proposals(
        grid,
        current,
        np.zeros(num_agents, dtype=np.bool_),
        actions,
    )

    assert np.array_equal(resolved.positions, current)
    assert np.all(resolved.categories[:-1] == 4)
    assert resolved.categories[-1] == 0
    assert resolved.operation_count <= 4 * num_agents * num_agents

    no_conflict_operations = 0
    separated = current.copy()
    separated[:, 1] *= 2
    wide_grid = np.zeros((1, 2 * num_agents + 1), dtype=np.uint8)
    for _ in range(3):
        one_step = module._resolve_proposals(
            wide_grid,
            separated,
            np.zeros(num_agents, dtype=np.bool_),
            np.zeros(num_agents, dtype=np.uint8),
        )
        no_conflict_operations += one_step.operation_count
    assert no_conflict_operations <= 12 * num_agents


def test_event_resolver_matches_naive_cuda_scan_order_on_seeded_cases():
    import expert.transition_adapters as module

    rng = np.random.default_rng(9917)
    moves = np.asarray([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

    def naive(grid, current, arrived, actions):
        effective = np.where(arrived, 0, actions)
        raw = current + moves[effective]
        height, width = grid.shape
        invalid = (
            (raw[:, 0] < 0)
            | (raw[:, 0] >= height)
            | (raw[:, 1] < 0)
            | (raw[:, 1] >= width)
        )
        safe = np.clip(raw, [0, 0], [height - 1, width - 1])
        invalid |= grid[safe[:, 0], safe[:, 1]] != 0
        proposed = np.where(invalid[:, None], current, raw)
        categories = np.zeros(len(current), dtype=np.uint8)
        categories[(~arrived) & (actions != 0) & invalid] = 1
        for _ in range(len(current)):
            next_proposed = proposed.copy()
            changed = False
            for agent_idx in range(len(current)):
                if np.array_equal(proposed[agent_idx], current[agent_idx]):
                    continue
                category = 0
                for other_idx in range(len(current)):
                    if other_idx == agent_idx:
                        continue
                    if (
                        np.array_equal(proposed[agent_idx], current[other_idx])
                        and np.array_equal(
                            proposed[other_idx], current[agent_idx]
                        )
                    ):
                        category = 3
                    elif agent_idx > other_idx and np.array_equal(
                        proposed[agent_idx], proposed[other_idx]
                    ):
                        category = 2
                    elif (
                        np.array_equal(proposed[agent_idx], current[other_idx])
                        and np.array_equal(
                            proposed[other_idx], current[other_idx]
                        )
                    ):
                        category = 4
                    if category:
                        break
                if category:
                    next_proposed[agent_idx] = current[agent_idx]
                    if categories[agent_idx] == 0:
                        categories[agent_idx] = category
                    changed = True
            proposed = next_proposed
            if not changed:
                break
        return proposed, categories

    for _ in range(80):
        num_agents = 16
        flat = rng.choice(12 * 12, size=num_agents, replace=False)
        current = np.stack((flat // 12, flat % 12), axis=-1).astype(np.int32)
        grid = (rng.random((12, 12)) < 0.12).astype(np.uint8)
        grid[current[:, 0], current[:, 1]] = 0
        arrived = rng.random(num_agents) < 0.15
        actions = rng.integers(0, 5, size=num_agents, dtype=np.uint8)
        expected_positions, expected_categories = naive(
            grid, current, arrived, actions
        )
        observed = module._resolve_proposals(grid, current, arrived, actions)
        assert np.array_equal(observed.positions, expected_positions)
        assert np.array_equal(observed.categories, expected_categories)


def test_public_contract_rejects_random_generation_inputs():
    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[0])
    with pytest.raises(TypeError, match="seed"):
        make_transition_adapter("pogema", batch, device="cpu", seed=7)
    with pytest.raises(TypeError, match="random"):
        make_transition_adapter(
            "pogema", batch, device="cpu", random_generation=True
        )


def test_prepare_actions_validates_chunk_and_timed_steps_use_handle_identity(
    monkeypatch,
):
    import expert.transition_adapters as module

    batch = _batch(
        positions=[[10, 10]],
        goals=[[30, 30]],
        actions=[[4], [4], [4]],
    )
    adapter = make_transition_adapter("pogema", batch, device="cpu")
    adapter.reset_device_state()

    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(batch.actions[0])

    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(batch.actions[1:])

    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(batch.actions[::2])

    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(batch.actions[0])

    wrong_dtype = np.array(batch.actions[:2], dtype=np.uint16, copy=True)
    wrong_dtype.flags.writeable = False
    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(wrong_dtype)

    copied = np.array(batch.actions[:2], copy=True)
    copied.flags.writeable = False
    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(copied)

    other = _batch(
        positions=[[10, 10]], goals=[[30, 30]], actions=[[4], [4], [4]]
    )
    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(other.actions[:2])

    writable = np.array(batch.actions[:2], copy=True)
    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(writable)

    with pytest.raises(ValueError, match="non-empty canonical action chunk"):
        adapter.prepare_actions(batch.actions[:0])

    oversized = np.lib.stride_tricks.as_strided(
        batch.actions,
        shape=(batch.horizon + 1, batch.num_envs, batch.num_agents),
        strides=batch.actions.strides,
        writeable=False,
    )
    with pytest.raises(ValueError, match="canonical action chunk"):
        adapter.prepare_actions(oversized)

    prepared = adapter.prepare_actions(batch.actions[:2])
    assert len(prepared) == 2
    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(prepared[1])
    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(object())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("timed step allocated action-token metadata")

    monkeypatch.setattr(module, "_action_token", forbidden)
    monkeypatch.setattr(module, "_action_chunk_token", forbidden, raising=False)
    adapter.step_transition(prepared[0])
    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(prepared[0])
    adapter.step_transition(prepared[1])
    monkeypatch.undo()

    single = adapter.prepare_action(batch.actions[2])
    adapter.step_transition(single)

    adapter.reset_device_state()
    stale_prepared = adapter.prepare_actions(batch.actions[:2])
    replacement = adapter.prepare_actions(batch.actions[:2])
    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(stale_prepared[0])
    adapter.step_transition(replacement[0])

    adapter.reset_device_state()
    stale_prepared = adapter.prepare_actions(batch.actions[:2])
    adapter.reset_device_state()
    with pytest.raises(RuntimeError, match="prepared action identity"):
        adapter.step_transition(stale_prepared[0])


def test_chunk_prepare_precedes_one_timed_multistep_region(monkeypatch):
    import expert.transition_adapters as module

    batch = _batch(
        positions=[[10, 10]],
        goals=[[30, 30]],
        actions=[[4], [3], [4]],
    )
    adapter = make_transition_adapter("pogema", batch, device="cpu")
    adapter.reset_device_state()
    events = []

    events.append("prepare_chunk")
    prepared = adapter.prepare_actions(batch.actions)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("timed step allocated action-token metadata")

    monkeypatch.setattr(module, "_action_token", forbidden)
    monkeypatch.setattr(module, "_action_chunk_token", forbidden, raising=False)
    events.append("timer_start")
    for handle in prepared:
        events.append("step")
        adapter.step_transition(handle)
    events.append("sync")
    adapter.synchronize_transition()
    events.append("timer_stop")

    assert events == [
        "prepare_chunk",
        "timer_start",
        "step",
        "step",
        "step",
        "sync",
        "timer_stop",
    ]


def test_all_operations_require_a_successful_reset():
    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter("pogema", batch, device="cpu")

    operations = (
        lambda: adapter.prepare_actions(batch.actions),
        lambda: adapter.prepare_action(batch.actions[0]),
        lambda: adapter.step_transition(batch.actions[0]),
        adapter.synchronize_transition,
        adapter.materialize_snapshot,
        lambda: adapter.consumed_input_sha256,
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="not ready.*reset_device_state"):
            operation()


def test_pogema_reports_canonical_hash_and_preserves_frozen_actions_on_replay():
    batch = _batch(
        positions=[[10, 10], [10, 12], [20, 20]],
        goals=[[30, 30], [31, 31], [20, 20]],
        arrived=[[False, False, True]],
        actions=[[4, 3, 2], [0, 0, 4]],
    )
    before_bytes = _exact_bytes(batch)
    before_actions = batch.actions.tobytes(order="C")
    before_hash = batch.semantic_sha256
    adapter = make_transition_adapter("pogema", batch, device="cpu")

    first = _snapshot(adapter, batch)
    second = _snapshot(adapter, batch)

    assert isinstance(first, TransitionSnapshot)
    assert first == second
    assert adapter.consumed_input_sha256 == before_hash
    assert batch.semantic_sha256 == before_hash
    assert _exact_bytes(batch) == before_bytes
    assert batch.actions.tobytes(order="C") == before_actions
    assert not batch.actions.flags.writeable


def test_pogema_transition_bypasses_step_and_observation(monkeypatch):
    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter("pogema", batch, device="cpu")
    adapter.reset_device_state()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrapper step/observation path entered")

    for env in adapter._envs:
        monkeypatch.setattr(env, "step", forbidden)
        monkeypatch.setattr(env, "_obs", forbidden)
    prepared = adapter.prepare_action(batch.actions[0])
    adapter.step_transition(prepared)
    assert adapter.materialize_snapshot().positions.tolist() == [[[10, 11]]]


def test_pogema_timed_transition_does_not_convert_arrival_state(monkeypatch):
    batch = _batch(
        positions=[[10, 10], [20, 20]],
        goals=[[10, 11], [30, 30]],
        actions=[4, 0],
    )
    adapter = make_transition_adapter("pogema", batch, device="cpu")
    adapter.reset_device_state()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("timed POGEMA transition allocated via np.asarray")

    prepared = adapter.prepare_action(batch.actions[0])
    monkeypatch.setattr(np, "asarray", forbidden)
    adapter.step_transition(prepared)
    assert adapter._arrived.tolist() == [[True, False]]


def test_step_transition_does_not_materialize_or_classify(monkeypatch):
    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter("jax", batch, device="cuda:0")
    adapter.reset_device_state()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("diagnostics entered timed primitive")

    import expert.transition_adapters as module

    prepared = adapter.prepare_action(batch.actions[0])
    monkeypatch.setattr(module, "_materialize_array", forbidden)
    monkeypatch.setattr(adapter, "_diagnose_backend_replay", forbidden)
    adapter.step_transition(prepared)
    adapter.synchronize_transition()


def test_backend_diagnostic_replay_does_not_reuse_stale_actual_outcomes(
    monkeypatch,
):
    import expert.transition_adapters as module

    batch = _batch(
        positions=[[10, 10]],
        goals=[[30, 30]],
        actions=[[4], [3]],
    )
    adapter = make_transition_adapter("pogema", batch, device="cpu")
    adapter.reset_device_state()
    for step in range(batch.horizon):
        prepared = adapter.prepare_action(batch.actions[step])
        adapter.step_transition(prepared)
    assert adapter._materialize_backend_state().positions.tolist() == [[[10, 10]]]

    first = adapter.materialize_snapshot()
    assert first.positions.tolist() == [[[10, 10]]]
    assert not np.any(first.collision_outcomes.semantic_mismatches)
    assert first.collision_outcomes.completed_moves.tolist() == [[2]]

    real_materialize = adapter._materialize_backend_state

    def report_one_intermediate_deviation():
        state = real_materialize()
        if adapter._cursor != 1:
            return state
        positions = np.array(state.positions, copy=True)
        positions[0, 0] = [10, 10]
        return module._BackendState(
            positions=positions,
            goals=state.goals,
            arrived=state.arrived,
            terminated=state.terminated,
            truncated=state.truncated,
            step_counts=state.step_counts,
        )

    reset_count = 0
    real_reset = adapter.reset_device_state

    def counted_reset():
        nonlocal reset_count
        reset_count += 1
        return real_reset()

    monkeypatch.setattr(
        adapter, "_materialize_backend_state", report_one_intermediate_deviation
    )
    monkeypatch.setattr(adapter, "reset_device_state", counted_reset)
    second = adapter.materialize_snapshot()
    assert second.positions.tolist() == [[[10, 10]]]
    assert np.any(second.collision_outcomes.semantic_mismatches[0])
    assert second.collision_outcomes.completed_moves.tolist() == [[0]]
    assert reset_count == 1


def test_dtype_normalized_backend_roundtrip_fails_closed_on_changed_bytes():
    import expert.transition_adapters as module

    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    changed_actions = np.array(batch.actions, copy=True)
    changed_actions[0, 0, 0] = 3
    with pytest.raises(ValueError, match="actions.*canonical frozen bytes"):
        module._verify_backend_roundtrip(
            batch,
            instance_ids=np.array(batch.instance_ids, dtype=np.int64),
            grids=np.array(batch.grids, dtype=np.int32),
            positions=np.array(batch.positions, dtype=np.int16),
            goals=np.array(batch.goals, dtype=np.int32),
            arrived=np.array(batch.arrived, dtype=np.uint8),
            active=np.array(batch.active, dtype=np.uint8),
            actions=changed_actions,
        )


@pytest.mark.parametrize(
    ("backend", "device"), [("jax", "cuda:0"), ("cuda", "cuda:0")]
)
def test_device_reset_hash_verification_does_not_materialize_device_arrays(
    monkeypatch, backend, device
):
    import expert.transition_adapters as module

    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter(backend, batch, device=device)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("reset performed device-to-host materialization")

    monkeypatch.setattr(module, "_materialize_array", forbidden)
    adapter.reset_device_state()
    assert adapter.consumed_input_sha256 == batch.semantic_sha256


@pytest.mark.parametrize(
    ("backend", "device"),
    [("pogema", "cpu"), ("cuda", "cuda:0"), ("jax", "cuda:0")],
)
def test_failed_repeated_reset_immediately_invalidates_previous_consumed_hash(
    monkeypatch, backend, device
):
    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter(backend, batch, device=device)
    adapter.reset_device_state()
    assert adapter.consumed_input_sha256 == batch.semantic_sha256

    if backend == "pogema":
        original_import = builtins.__import__

        def fail_early_import(name, *args, **kwargs):
            if name == "pogema.envs":
                raise RuntimeError("injected early POGEMA import failure")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_early_import)
        error = "injected early POGEMA"
    elif backend == "cuda":
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        error = "CUDA is unavailable"
    else:
        import jax

        def fail_early_devices(*_args, **_kwargs):
            raise RuntimeError("injected early JAX device failure")

        monkeypatch.setattr(jax, "devices", fail_early_devices)
        error = "injected early JAX"

    with pytest.raises(RuntimeError, match=error):
        adapter.reset_device_state()
    invalidated_operations = (
        lambda: adapter.prepare_actions(batch.actions),
        lambda: adapter.prepare_action(batch.actions[0]),
        lambda: adapter.step_transition(batch.actions[0]),
        adapter.synchronize_transition,
        adapter.materialize_snapshot,
        lambda: adapter.consumed_input_sha256,
    )
    for operation in invalidated_operations:
        with pytest.raises(RuntimeError, match="not ready.*reset_device_state"):
            operation()


@pytest.mark.parametrize(
    ("backend", "helper_name", "field"),
    [
        ("cuda", "_verify_torch_device_consumption", "actions"),
        ("cuda", "_verify_torch_device_consumption", "positions"),
        ("jax", "_verify_jax_device_consumption", "actions"),
        ("jax", "_verify_jax_device_consumption", "positions"),
    ],
)
def test_reset_fails_closed_when_actual_device_input_is_corrupted(
    monkeypatch, backend, helper_name, field
):
    import expert.transition_adapters as module

    batch = _batch(positions=[[10, 10]], goals=[[20, 20]], actions=[4])
    adapter = make_transition_adapter(backend, batch, device="cuda:0")
    original = getattr(module, helper_name)

    def corrupt_then_verify(frozen_batch, **device_fields):
        if backend == "cuda":
            tensor = device_fields[field]
            tensor.reshape(-1)[0].add_(1)
        elif field == "actions":
            actions = list(device_fields["actions"])
            actions[0] = actions[0].at[0, 0].set(3)
            device_fields["actions"] = tuple(actions)
        else:
            positions = device_fields["positions"]
            device_fields["positions"] = positions.at[0, 0, 0].set(11)
        return original(frozen_batch, **device_fields)

    monkeypatch.setattr(module, helper_name, corrupt_then_verify)
    with pytest.raises(ValueError, match="actual device input"):
        adapter.reset_device_state()
    for operation in (
        lambda: adapter.prepare_actions(batch.actions),
        lambda: adapter.prepare_action(batch.actions[0]),
        lambda: adapter.step_transition(batch.actions[0]),
        adapter.synchronize_transition,
        adapter.materialize_snapshot,
        lambda: adapter.consumed_input_sha256,
    ):
        with pytest.raises(RuntimeError, match="not ready.*reset_device_state"):
            operation()


def test_jax_timed_transition_uses_prestaged_per_step_action(monkeypatch):
    import jax
    import expert.transition_adapters as module

    batch = _batch(
        positions=[[10, 10]],
        goals=[[20, 20]],
        actions=[[4], [4]],
    )
    adapter = make_transition_adapter("jax", batch, device="cuda:0")
    adapter.reset_device_state()
    assert isinstance(adapter._actions_by_step, tuple)
    assert len(adapter._actions_by_step) == batch.horizon
    assert not hasattr(adapter, "_actions")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("timed JAX transition staged or materialized data")

    monkeypatch.setattr(jax, "device_put", forbidden)
    monkeypatch.setattr(module, "_materialize_array", forbidden)
    prepared = adapter.prepare_action(batch.actions[0])
    adapter.step_transition(prepared)
    adapter.synchronize_transition()


@pytest.mark.parametrize(
    ("backend", "device"),
    [("pogema", "cpu"), ("jax", "cuda:0"), ("cuda", "cuda:0")],
)
def test_done_environments_are_skipped_while_other_environments_advance(
    backend, device
):
    positions = np.asarray([[[10, 10]], [[30, 30]]], dtype=np.uint16)
    goals = np.asarray([[[10, 11]], [[40, 40]]], dtype=np.uint16)
    actions = np.asarray([[[4], [4]], [[2], [4]]], dtype=np.uint8)
    batch = _batch(
        positions=positions,
        goals=goals,
        actions=actions,
        horizon=2,
    )
    adapter = make_transition_adapter(backend, batch, device=device)
    snapshot = _snapshot(adapter, batch)

    assert snapshot.positions.tolist() == [[[10, 11]], [[30, 32]]]
    assert snapshot.arrived.tolist() == [[True], [False]]
    assert snapshot.terminated.tolist() == [True, False]
    assert snapshot.truncated.tolist() == [False, True]
    assert snapshot.step_counts.tolist() == [1, 2]
    for item in (
        snapshot.positions,
        snapshot.goals,
        snapshot.arrived,
        snapshot.terminated,
        snapshot.truncated,
        snapshot.step_counts,
    ):
        assert not item.flags.writeable


def test_stepwise_progression_snapshots_match_across_all_backends():
    batch = _batch(
        positions=[[[10, 10]], [[30, 30]]],
        goals=[[[10, 11]], [[40, 40]]],
        actions=[[[4], [4]], [[2], [4]]],
        horizon=2,
    )

    snapshots = {
        name: _stepwise_snapshots(
            make_transition_adapter(name, batch, device=device), batch
        )
        for name, device in (
            ("pogema", "cpu"),
            ("jax", "cuda:0"),
            ("cuda", "cuda:0"),
        )
    }

    assert len(snapshots["pogema"]) == 2
    assert snapshots["jax"] == snapshots["pogema"]
    assert snapshots["cuda"] == snapshots["pogema"]

    first = snapshots["pogema"][0]
    assert first.positions.tolist() == [[[10, 11]], [[30, 31]]]
    assert first.arrived.tolist() == [[True], [False]]
    assert first.terminated.tolist() == [True, False]
    assert first.truncated.tolist() == [False, False]
    assert first.step_counts.tolist() == [1, 1]
    assert first.collision_outcomes.requested_moves.tolist() == [[1], [1]]
    assert first.collision_outcomes.completed_moves.tolist() == [[1], [1]]
    assert first.collision_outcomes.category_trace.shape == (1, 2, 1)
    assert not np.any(first.collision_outcomes.semantic_mismatches)

    second = snapshots["pogema"][1]
    assert second.positions.tolist() == [[[10, 11]], [[30, 32]]]
    assert second.arrived.tolist() == [[True], [False]]
    assert second.terminated.tolist() == [True, False]
    assert second.truncated.tolist() == [False, True]
    assert second.step_counts.tolist() == [1, 2]
    assert second.collision_outcomes.requested_moves.tolist() == [[1], [2]]
    assert second.collision_outcomes.completed_moves.tolist() == [[1], [2]]
    assert second.collision_outcomes.category_trace.shape == (2, 2, 1)
    assert not np.any(second.collision_outcomes.semantic_mismatches)


def test_stepwise_collision_trace_accumulates_without_losing_progression():
    batch = _batch(
        positions=[[[10, 10], [10, 11]]],
        goals=[[[20, 20], [20, 21]]],
        actions=[[[4, 3]], [[0, 4]]],
        horizon=2,
    )

    snapshots = {
        name: _stepwise_snapshots(
            make_transition_adapter(name, batch, device=device), batch
        )
        for name, device in (
            ("pogema", "cpu"),
            ("jax", "cuda:0"),
            ("cuda", "cuda:0"),
        )
    }

    assert snapshots["jax"] == snapshots["pogema"]
    assert snapshots["cuda"] == snapshots["pogema"]

    first = snapshots["pogema"][0]
    assert first.positions.tolist() == [[[10, 10], [10, 11]]]
    assert first.terminated.tolist() == [False]
    assert first.truncated.tolist() == [False]
    assert first.step_counts.tolist() == [1]
    assert first.collision_outcomes.requested_moves.tolist() == [[1, 1]]
    assert first.collision_outcomes.completed_moves.tolist() == [[0, 0]]
    assert int(first.collision_outcomes.edge_swap_rejections.sum()) == 2
    assert first.collision_outcomes.category_trace.tolist() == [[[3, 3]]]
    assert not np.any(first.collision_outcomes.semantic_mismatches)

    second = snapshots["pogema"][1]
    assert second.positions.tolist() == [[[10, 10], [10, 12]]]
    assert second.terminated.tolist() == [False]
    assert second.truncated.tolist() == [True]
    assert second.step_counts.tolist() == [2]
    assert second.collision_outcomes.requested_moves.tolist() == [[1, 2]]
    assert second.collision_outcomes.completed_moves.tolist() == [[0, 1]]
    assert int(second.collision_outcomes.edge_swap_rejections.sum()) == 2
    assert second.collision_outcomes.category_trace.tolist() == [[[3, 3]], [[0, 0]]]
    assert not np.any(second.collision_outcomes.semantic_mismatches)


def test_seeded_random_multistep_parity_across_all_backends():
    rng = np.random.default_rng(20260719)
    num_envs, num_agents, horizon = 2, 16, 4
    grids = (rng.random((num_envs, MAP_SIZE, MAP_SIZE)) < 0.04).astype(np.uint8)
    positions = np.zeros((num_envs, num_agents, 2), dtype=np.uint16)
    goals = np.zeros_like(positions)
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            positions[env_idx, agent_idx] = (
                10 + 3 * (agent_idx // 4),
                10 + 3 * (agent_idx % 4),
            )
            goals[env_idx, agent_idx] = (
                80 + 3 * (agent_idx // 4),
                80 + 3 * (agent_idx % 4),
            )
        grids[env_idx, positions[env_idx, :, 0], positions[env_idx, :, 1]] = 0
        grids[env_idx, goals[env_idx, :, 0], goals[env_idx, :, 1]] = 0
    actions = rng.integers(
        0, 5, size=(horizon, num_envs, num_agents), dtype=np.uint8
    )
    batch = _batch(
        positions=positions,
        goals=goals,
        grids=grids,
        actions=actions,
        horizon=horizon,
    )

    snapshots = {
        name: _snapshot(
            make_transition_adapter(name, batch, device=device), batch
        )
        for name, device in (
            ("pogema", "cpu"),
            ("jax", "cuda:0"),
            ("cuda", "cuda:0"),
        )
    }
    assert snapshots["jax"] == snapshots["pogema"]
    assert snapshots["cuda"] == snapshots["pogema"]
    assert not np.any(snapshots["cuda"].collision_outcomes.semantic_mismatches)


DIRECTED_FIXTURES = {
    "all_five_actions": {
        "positions": [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]],
        "goals": [[70, 70], [71, 71], [72, 72], [73, 73], [74, 74]],
        "actions": [0, 1, 2, 3, 4],
        "expected": [[10, 10], [19, 20], [31, 30], [40, 39], [50, 51]],
        "collision": {},
    },
    "wall": {
        "positions": [[10, 10]],
        "goals": [[20, 20]],
        "actions": [4],
        "walls": [(10, 11)],
        "expected": [[10, 10]],
        "collision": {"wall_rejections": 1},
    },
    "vertex": {
        "positions": [[10, 10], [10, 12]],
        "goals": [[20, 20], [21, 21]],
        "actions": [4, 3],
        "expected": [[10, 11], [10, 12]],
        "collision": {"vertex_rejections": 1},
    },
    "swap": {
        "positions": [[10, 10], [10, 11]],
        "goals": [[20, 20], [21, 21]],
        "actions": [4, 3],
        "expected": [[10, 10], [10, 11]],
        "collision": {"edge_swap_rejections": 2},
    },
    "stationary_chain": {
        "positions": [[10, 10], [10, 11], [10, 12]],
        "goals": [[20, 20], [21, 21], [22, 22]],
        "actions": [4, 4, 0],
        "expected": [[10, 10], [10, 11], [10, 12]],
        "collision": {"chain_rejections": 2},
    },
    "movement_cycle": {
        "positions": [[10, 10], [10, 11], [11, 11], [11, 10]],
        "goals": [[20, 20], [21, 21], [22, 22], [23, 23]],
        "actions": [4, 2, 3, 1],
        "expected": [[10, 11], [11, 11], [11, 10], [10, 10]],
        "collision": {},
    },
    "arrived_occupies_goal": {
        "positions": [[10, 10], [10, 11]],
        "goals": [[10, 10], [20, 20]],
        "arrived": [[True, False]],
        "actions": [4, 3],
        "expected": [[10, 10], [10, 11]],
        # CUDA/JAX test vertex before stationary-chain, so the higher-id mover
        # is categorized as a vertex rejection against the arrived occupant.
        "collision": {"vertex_rejections": 1},
    },
    "final_arrival": {
        "positions": [[10, 10]],
        "goals": [[10, 11]],
        "actions": [4],
        "expected": [[10, 11]],
        "terminated": [True],
        "truncated": [False],
        "collision": {},
    },
    "truncation": {
        "positions": [[10, 10]],
        "goals": [[20, 20]],
        "actions": [0],
        "expected": [[10, 10]],
        "terminated": [False],
        "truncated": [True],
        "collision": {},
    },
}


def _fixture_batch(name: str) -> FrozenTransitionBatch:
    case = DIRECTED_FIXTURES[name]
    grids = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    for x, y in case.get("walls", ()):
        grids[x, y] = 1
    return _batch(
        positions=case["positions"],
        goals=case["goals"],
        arrived=case.get("arrived"),
        actions=case["actions"],
        grids=grids,
    )


def _assert_case(snapshot: TransitionSnapshot, name: str) -> None:
    case = DIRECTED_FIXTURES[name]
    assert snapshot.positions.tolist() == [case["expected"]]
    assert snapshot.goals.tolist() == [case["goals"]]
    assert snapshot.terminated.tolist() == case.get("terminated", [False])
    # FrozenTransitionBatch has one authoritative horizon.  Every unfinished
    # one-step directed fixture therefore truncates at the end of that step.
    assert snapshot.truncated.tolist() == case.get("truncated", [True])
    assert snapshot.step_counts.tolist() == [1]
    assert isinstance(snapshot.collision_outcomes, CollisionOutcomes)
    assert snapshot.collision_outcomes.category_trace.shape == (
        1,
        1,
        len(case["actions"]),
    )
    assert not np.any(snapshot.collision_outcomes.semantic_mismatches)
    expected = case["collision"]
    for field in (
        "wall_rejections",
        "vertex_rejections",
        "edge_swap_rejections",
        "chain_rejections",
    ):
        assert int(getattr(snapshot.collision_outcomes, field).sum()) == expected.get(
            field, 0
        )


@pytest.mark.parametrize("name", DIRECTED_FIXTURES)
def test_pogema_and_jax_directed_parity(name):
    batch = _fixture_batch(name)
    pogema = make_transition_adapter("pogema", batch, device="cpu")
    jax_adapter = make_transition_adapter("jax", batch, device="cuda:0")

    pogema_snapshot = _snapshot(pogema, batch)
    jax_snapshot = _snapshot(jax_adapter, batch)

    _assert_case(pogema_snapshot, name)
    assert jax_snapshot == pogema_snapshot
    assert pogema.consumed_input_sha256 == batch.semantic_sha256
    assert jax_adapter.consumed_input_sha256 == batch.semantic_sha256


@pytest.mark.parametrize("name", DIRECTED_FIXTURES)
def test_cuda_directed_parity(name):
    batch = _fixture_batch(name)
    cuda = make_transition_adapter("cuda", batch, device="cuda:0")
    pogema = make_transition_adapter("pogema", batch, device="cpu")

    cuda_snapshot = _snapshot(cuda, batch)
    pogema_snapshot = _snapshot(pogema, batch)

    _assert_case(cuda_snapshot, name)
    assert cuda_snapshot == pogema_snapshot
    assert cuda.consumed_input_sha256 == batch.semantic_sha256
