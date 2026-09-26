from __future__ import annotations

import argparse
import json
import pathlib
from collections.abc import Sequence
from typing import Any

from expert.benchmark_contract import (
    FrozenTransitionBatch,
    build_frozen_pool,
    save_frozen_transition_batch,
)


MANIFEST_SCHEMA_VERSION = "benchmark-manifest-v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_benchmark_manifest(
    payload: dict[str, Any], *, require_a1_a2_metadata: bool = False
) -> None:
    """Validate a manifest before a parity or benchmark runner consumes it."""

    if not isinstance(payload, dict) or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("benchmark manifest has an unsupported schema_version")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("benchmark manifest entries must be a non-empty list")
    required = {
        "pool_name",
        "topology",
        "density",
        "num_agents",
        "horizon",
        "instance_ids",
        "frozen_batch_path",
        "semantic_sha256",
        "num_envs",
        "grid_shape",
    }
    seen_pool_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("benchmark manifest entries must be objects")
        missing = required.difference(entry)
        if missing:
            raise ValueError(f"benchmark manifest entry missing fields: {sorted(missing)}")
        pool_name = entry["pool_name"]
        if not isinstance(pool_name, str) or not pool_name or pool_name in seen_pool_names:
            raise ValueError("benchmark manifest pool_name must be non-empty and unique")
        seen_pool_names.add(pool_name)
        if not isinstance(entry["topology"], str) or not entry["topology"]:
            raise ValueError("benchmark manifest topology must be a non-empty string")
        if not isinstance(entry["density"], (int, float)) or isinstance(entry["density"], bool):
            raise ValueError("benchmark manifest density must be numeric")
        if not 0.0 <= float(entry["density"]) < 1.0:
            raise ValueError("benchmark manifest density must be in [0, 1)")
        for field in ("num_agents", "horizon", "num_envs"):
            if not isinstance(entry[field], int) or isinstance(entry[field], bool) or entry[field] <= 0:
                raise ValueError(f"benchmark manifest {field} must be a positive integer")
        instance_ids = entry["instance_ids"]
        if not isinstance(instance_ids, list) or len(instance_ids) != entry["num_envs"]:
            raise ValueError("benchmark manifest instance_ids must match num_envs")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in instance_ids):
            raise ValueError("benchmark manifest instance_ids must be native integers")
        if len(set(instance_ids)) != len(instance_ids):
            raise ValueError("benchmark manifest instance_ids must be unique")
        if not isinstance(entry["frozen_batch_path"], str) or not entry["frozen_batch_path"]:
            raise ValueError("benchmark manifest frozen_batch_path must be non-empty")
        if not _is_sha256(entry["semantic_sha256"]):
            raise ValueError("benchmark manifest semantic_sha256 must be a lowercase SHA-256")
        if entry["grid_shape"] != [128, 128] and require_a1_a2_metadata:
            raise ValueError("benchmark manifest A1/A2 grid_shape must be [128, 128]")
        if require_a1_a2_metadata:
            for field in ("role", "task_semantics", "topology_parameters", "compiled_profile"):
                if field not in entry:
                    raise ValueError(f"benchmark manifest A1/A2 entry missing {field}")
            task_semantics = entry["task_semantics"]
            if task_semantics != {
                "task_mode": "standard_mapf",
                "collision_system": "soft",
                "on_target": "nothing",
            }:
                raise ValueError("benchmark manifest A1/A2 task_semantics must be standard MAPF")
            if not isinstance(entry["topology_parameters"], dict) or not entry["topology_parameters"]:
                raise ValueError("benchmark manifest topology_parameters must be a non-empty object")
            compiled_profile = entry["compiled_profile"]
            if not isinstance(compiled_profile, dict) or {
                "map_width",
                "map_height",
            }.difference(compiled_profile):
                raise ValueError("benchmark manifest compiled_profile is incomplete")
            if compiled_profile["map_width"] != 128 or compiled_profile["map_height"] != 128:
                raise ValueError("benchmark manifest compiled_profile must be 128x128")


def build_manifest_entry(
    *,
    pool_name: str,
    topology: str,
    density: float,
    num_agents: int,
    horizon: int,
    output_path: str | pathlib.Path,
    batch: FrozenTransitionBatch,
) -> dict[str, Any]:
    resolved = pathlib.Path(output_path).expanduser().resolve()
    return {
        "pool_name": pool_name,
        "topology": topology,
        "density": float(density),
        "num_agents": int(num_agents),
        "horizon": int(horizon),
        "instance_ids": batch.instance_ids.tolist(),
        "frozen_batch_path": str(resolved),
        "semantic_sha256": batch.semantic_sha256,
        "num_envs": batch.num_envs,
        "grid_shape": list(batch.grids.shape[1:]),
    }


def write_manifest(
    *,
    output_path: str | pathlib.Path,
    entries: Sequence[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    resolved = pathlib.Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "entries": list(entries),
    }
    if extra_metadata:
        reserved = {"schema_version", "entries"}.intersection(extra_metadata)
        if reserved:
            raise ValueError(f"manifest extra_metadata may not override {sorted(reserved)}")
        payload.update(extra_metadata)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(resolved)


def build_and_write_pool(
    *,
    pool_name: str,
    topology: str,
    density: float,
    num_agents: int,
    horizon: int,
    seeds: Sequence[int],
    artifact_path: str | pathlib.Path,
    manifest_path: str | pathlib.Path,
    instance_factory,
) -> dict[str, Any]:
    batch = build_frozen_pool(
        seeds=seeds,
        horizon=horizon,
        instance_factory=instance_factory,
    )
    save_frozen_transition_batch(artifact_path, batch)
    entry = build_manifest_entry(
        pool_name=pool_name,
        topology=topology,
        density=density,
        num_agents=num_agents,
        horizon=horizon,
        output_path=artifact_path,
        batch=batch,
    )
    write_manifest(output_path=manifest_path, entries=[entry])
    return entry


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a benchmark manifest from an existing frozen batch.")
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--artifact", required=True, type=pathlib.Path)
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--semantic-sha256", required=True)
    args = parser.parse_args(argv)
    entry = {
        "pool_name": args.pool_name,
        "topology": args.topology,
        "density": float(args.density),
        "num_agents": int(args.num_agents),
        "horizon": int(args.horizon),
        "instance_ids": [],
        "frozen_batch_path": str(args.artifact.expanduser().resolve()),
        "semantic_sha256": args.semantic_sha256,
        "num_envs": None,
        "grid_shape": None,
    }
    write_manifest(output_path=args.manifest, entries=[entry])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
