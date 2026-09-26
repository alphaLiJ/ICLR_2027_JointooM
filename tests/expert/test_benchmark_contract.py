from __future__ import annotations

import json

import numpy as np
import pytest

from expert.benchmark_contract import (
    ACTION_DELTAS,
    FrozenTransitionBatch,
    build_frozen_pool,
    load_frozen_transition_batch,
    save_frozen_transition_batch,
)


def _valid_arrays(*, num_envs: int = 2, horizon: int = 3):
    grids = np.zeros((num_envs, 5, 6), dtype=np.uint8)
    positions = np.empty((num_envs, 2, 2), dtype=np.uint16)
    goals = np.empty_like(positions)
    for env_idx in range(num_envs):
        positions[env_idx] = ((1, 1), (2, 2))
        goals[env_idx] = ((3, 4), (2, 2))
    arrived = np.zeros((num_envs, 2), dtype=np.bool_)
    arrived[:, 1] = True
    return {
        "instance_ids": np.arange(10, 10 + num_envs, dtype=np.int64),
        "grids": grids,
        "positions": positions,
        "goals": goals,
        "arrived": arrived,
        "actions": np.zeros((horizon, num_envs, 2), dtype=np.uint8),
        "horizon": horizon,
    }


def _batch(**overrides) -> FrozenTransitionBatch:
    values = _valid_arrays()
    values.update(overrides)
    return FrozenTransitionBatch(**values)


def test_action_deltas_define_all_five_shared_actions():
    before = ACTION_DELTAS.tobytes(order="C")
    assert ACTION_DELTAS.dtype == np.int8
    assert ACTION_DELTAS.tolist() == [
        [0, 0],
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
    ]
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        ACTION_DELTAS.setflags(write=True)
    assert ACTION_DELTAS.tobytes(order="C") == before


def test_constructor_uses_c_contiguous_irreversibly_read_only_copies():
    source = _valid_arrays()
    source["grids"] = source["grids"][:, :, ::-1]
    original_grid = source["grids"].copy()
    batch = FrozenTransitionBatch(**source)

    before_hash = batch.semantic_sha256
    before_bytes = {}

    for name in (
        "instance_ids",
        "grids",
        "positions",
        "goals",
        "arrived",
        "active",
        "actions",
    ):
        array = getattr(batch, name)
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        before_bytes[name] = array.tobytes(order="C")
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            array.setflags(write=True)

    source["grids"][0, 0, 0] = 9
    assert np.array_equal(batch.grids, original_grid)
    assert batch.semantic_sha256 == before_hash
    for name, payload in before_bytes.items():
        assert getattr(batch, name).tobytes(order="C") == payload
    with pytest.raises(ValueError, match="read-only"):
        batch.actions[0, 0, 0] = 1


def test_constructor_materializes_and_validates_explicit_active_mask():
    batch = _batch()

    assert batch.active.dtype == np.bool_
    assert not batch.active.flags.writeable
    assert np.array_equal(batch.active, ~batch.arrived)

    values = _valid_arrays()
    values["active"] = np.array(values["arrived"], copy=True)
    with pytest.raises(ValueError, match=r"active must equal logical_not\(arrived\)"):
        FrozenTransitionBatch(**values)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("instance_ids", np.array([10, 11], dtype=np.int32), "instance_ids.*int64"),
        ("grids", np.zeros((2, 5, 6), dtype=np.int8), "grids.*uint8"),
        ("positions", np.zeros((2, 2, 2), dtype=np.int32), "positions.*uint16"),
        ("goals", np.zeros((2, 2, 2), dtype=np.int64), "goals.*uint16"),
        ("arrived", np.zeros((2, 2), dtype=np.uint8), "arrived.*bool"),
        ("actions", np.zeros((3, 2, 2), dtype=np.int8), "actions.*uint8"),
    ],
)
def test_rejects_wrong_dtype(field, replacement, message):
    with pytest.raises(ValueError, match=message):
        _batch(**{field: replacement})


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("instance_ids", np.array([[10, 11]], dtype=np.int64), r"instance_ids.*\[E\]"),
        ("grids", np.zeros((2, 5), dtype=np.uint8), r"grids.*\[E, H, W\]"),
        ("positions", np.zeros((2, 2), dtype=np.uint16), r"positions.*\[E, A, 2\]"),
        ("goals", np.zeros((2, 2, 3), dtype=np.uint16), r"goals.*\[E, A, 2\]"),
        ("arrived", np.zeros((2, 2, 1), dtype=np.bool_), r"arrived.*\[E, A\]"),
        ("actions", np.zeros((3, 2), dtype=np.uint8), r"actions.*\[T, E, A\]"),
    ],
)
def test_rejects_wrong_rank_or_coordinate_width(field, replacement, message):
    with pytest.raises(ValueError, match=message):
        _batch(**{field: replacement})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda v: v["instance_ids"].__setitem__(1, 10), "instance_ids.*unique"),
        (lambda v: v["positions"].__setitem__((0, 0), (5, 1)), "position.*bounds"),
        (lambda v: v["goals"].__setitem__((0, 0), (1, 6)), "goal.*bounds"),
        (lambda v: v["grids"].__setitem__((0, 1, 1), 1), "position.*wall"),
        (lambda v: v["grids"].__setitem__((0, 3, 4), 1), "goal.*wall"),
        (lambda v: v["positions"].__setitem__((0, 1), (1, 1)), "duplicate position"),
        (lambda v: v["actions"].__setitem__((0, 0, 0), 5), r"actions.*\[0, 4\]"),
        (lambda v: v["arrived"].__setitem__((0, 0), True), "arrived.*goal"),
    ],
)
def test_rejects_invalid_mapf_state(mutation, message):
    values = _valid_arrays()
    mutation(values)
    with pytest.raises(ValueError, match=message):
        FrozenTransitionBatch(**values)


@pytest.mark.parametrize("horizon", [0, -1, True, 3.0])
def test_rejects_non_positive_or_non_integer_horizon(horizon):
    values = _valid_arrays()
    values["horizon"] = horizon
    with pytest.raises(ValueError, match="horizon.*positive integer"):
        FrozenTransitionBatch(**values)


def test_rejects_cross_field_shape_and_horizon_disagreement():
    values = _valid_arrays()
    values["actions"] = np.zeros((4, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="horizon.*actions"):
        FrozenTransitionBatch(**values)

    values = _valid_arrays()
    values["arrived"] = np.zeros((1, 2), dtype=np.bool_)
    with pytest.raises(ValueError, match="shape mismatch"):
        FrozenTransitionBatch(**values)


def test_semantic_hash_is_stable_and_sensitive_to_layout_metadata_and_content():
    first = _batch()
    same = _batch()
    assert first.semantic_sha256 == same.semantic_sha256

    changed_content = _valid_arrays()
    changed_content["actions"][0, 0, 0] = 1
    assert FrozenTransitionBatch(**changed_content).semantic_sha256 != first.semantic_sha256

    changed_shape = _valid_arrays(num_envs=1)
    assert FrozenTransitionBatch(**changed_shape).semantic_sha256 != first.semantic_sha256

    changed_dtype_values = _valid_arrays()
    changed_dtype_values["instance_ids"] = np.array([10, 12], dtype=np.int64)
    assert FrozenTransitionBatch(**changed_dtype_values).semantic_sha256 != first.semantic_sha256


def test_npz_json_metadata_round_trip_revalidates_and_recomputes_hash(tmp_path):
    path = tmp_path / "batch.npz"
    batch = _batch()
    digest = save_frozen_transition_batch(path, batch)

    loaded = load_frozen_transition_batch(path)

    assert digest == batch.semantic_sha256
    assert loaded.semantic_sha256 == batch.semantic_sha256
    assert np.array_equal(loaded.actions, batch.actions)
    for name in ("instance_ids", "grids", "positions", "goals", "arrived", "actions"):
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            getattr(loaded, name).setflags(write=True)
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
    assert metadata["schema_version"]
    assert metadata["semantic_sha256"] == batch.semantic_sha256


def test_load_detects_payload_tampering_with_stale_metadata_hash(tmp_path):
    path = tmp_path / "batch.npz"
    batch = _batch()
    save_frozen_transition_batch(path, batch)
    with np.load(path, allow_pickle=False) as payload:
        saved = {name: payload[name].copy() for name in payload.files}
    saved["actions"][0, 0, 0] = 1
    np.savez_compressed(path, **saved)

    with pytest.raises(ValueError, match="semantic SHA-256 mismatch"):
        load_frozen_transition_batch(path)


def test_load_rejects_metadata_schema_tampering(tmp_path):
    path = tmp_path / "batch.npz"
    save_frozen_transition_batch(path, _batch())
    with np.load(path, allow_pickle=False) as payload:
        saved = {name: payload[name].copy() for name in payload.files}
    metadata = json.loads(str(saved["metadata_json"].item()))
    metadata["schema_version"] = "unknown"
    saved["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **saved)

    with pytest.raises(ValueError, match="schema version"):
        load_frozen_transition_batch(path)


def test_take_envs_selects_a_nested_prefix_and_rejects_invalid_sizes():
    batch = _batch()
    one = batch.take_envs(1)
    two = batch.take_envs(2)

    assert one.instance_ids.tolist() == [10]
    assert two.instance_ids.tolist() == [10, 11]
    assert np.array_equal(one.grids, two.grids[:1])
    assert np.array_equal(one.actions, two.actions[:, :1])
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        one.grids.setflags(write=True)
    for size in (0, -1, 3, True, 1.0):
        with pytest.raises(ValueError, match="env_count"):
            batch.take_envs(size)


def test_build_frozen_pool_is_deterministic_and_uses_explicit_seed_order():
    calls = []

    def factory(*, seed: int, horizon: int):
        calls.append((seed, horizon))
        grid = np.zeros((5, 6), dtype=np.uint8)
        return {
            "grid": grid,
            "positions": np.array([[1, 1], [2, 2]], dtype=np.uint16),
            "goals": np.array([[3, 4], [2, 2]], dtype=np.uint16),
            "arrived": np.array([False, True], dtype=np.bool_),
            "actions": np.full((horizon, 2), seed % 5, dtype=np.uint8),
        }

    first = build_frozen_pool(seeds=[7, 3], horizon=4, instance_factory=factory)
    second = build_frozen_pool(seeds=[7, 3], horizon=4, instance_factory=factory)

    assert calls == [(7, 4), (3, 4), (7, 4), (3, 4)]
    assert first.instance_ids.tolist() == [7, 3]
    assert first.semantic_sha256 == second.semantic_sha256
    assert first.take_envs(1).instance_ids.tolist() == [7]


@pytest.mark.parametrize("seeds", [[], [1, 1], [True]])
def test_build_frozen_pool_rejects_invalid_seed_manifests(seeds):
    def factory(*, seed: int, horizon: int):  # pragma: no cover - invalid before call
        raise AssertionError("factory should not be called")

    with pytest.raises(ValueError, match="seeds"):
        build_frozen_pool(seeds=seeds, horizon=2, instance_factory=factory)
