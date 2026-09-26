from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expert.benchmark_contract import FrozenTransitionBatch
from expert.benchmark_manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_manifest_entry,
    validate_benchmark_manifest,
    write_manifest,
)


def _batch() -> FrozenTransitionBatch:
    return FrozenTransitionBatch(
        instance_ids=np.asarray([101, 102], dtype=np.int64),
        grids=np.zeros((2, 8, 8), dtype=np.uint8),
        positions=np.asarray(
            [
                [[2, 2], [4, 4]],
                [[3, 3], [5, 5]],
            ],
            dtype=np.uint16,
        ),
        goals=np.asarray(
            [
                [[2, 6], [4, 7]],
                [[3, 6], [5, 7]],
            ],
            dtype=np.uint16,
        ),
        arrived=np.zeros((2, 2), dtype=np.bool_),
        actions=np.full((3, 2, 2), 4, dtype=np.uint8),
        horizon=3,
    )


def test_build_manifest_entry_materializes_frozen_contract_fields(tmp_path):
    batch = _batch()
    artifact_path = tmp_path / "pool.npz"

    entry = build_manifest_entry(
        pool_name="pilot-pool",
        topology="maze",
        density=0.15,
        num_agents=2,
        horizon=3,
        output_path=artifact_path,
        batch=batch,
    )

    assert entry == {
        "pool_name": "pilot-pool",
        "topology": "maze",
        "density": 0.15,
        "num_agents": 2,
        "horizon": 3,
        "instance_ids": [101, 102],
        "frozen_batch_path": str(artifact_path.resolve()),
        "semantic_sha256": batch.semantic_sha256,
        "num_envs": 2,
        "grid_shape": [8, 8],
    }


def test_write_manifest_writes_schema_version_and_entries(tmp_path):
    path = tmp_path / "benchmark_manifest.json"
    entries = [
        {
            "pool_name": "pilot-pool",
            "topology": "maze",
            "density": 0.15,
            "num_agents": 2,
            "horizon": 3,
            "instance_ids": [101, 102],
            "frozen_batch_path": str((tmp_path / "pool.npz").resolve()),
            "semantic_sha256": "a" * 64,
            "num_envs": 2,
            "grid_shape": [8, 8],
        }
    ]

    resolved = write_manifest(output_path=path, entries=entries)

    assert resolved == str(path.resolve())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "entries": entries,
    }


def test_validate_benchmark_manifest_requires_complete_frozen_a1_a2_metadata(tmp_path):
    batch = _batch()
    entry = build_manifest_entry(
        pool_name="a1-stress-random-a064",
        topology="random",
        density=0.2,
        num_agents=2,
        horizon=3,
        output_path=tmp_path / "pool.npz",
        batch=batch,
    )
    entry.update(
        {
            "grid_shape": [128, 128],
            "role": "a1_stress",
            "task_semantics": {
                "task_mode": "standard_mapf",
                "collision_system": "soft",
                "on_target": "nothing",
            },
            "topology_parameters": {"generator": "random", "density": 0.2},
            "compiled_profile": {"map_width": 128, "map_height": 128},
        }
    )
    payload = {"schema_version": MANIFEST_SCHEMA_VERSION, "entries": [entry]}

    validate_benchmark_manifest(payload, require_a1_a2_metadata=True)

    entry["compiled_profile"] = {"map_width": 128}
    with np.testing.assert_raises_regex(ValueError, "compiled_profile"):
        validate_benchmark_manifest(payload, require_a1_a2_metadata=True)
