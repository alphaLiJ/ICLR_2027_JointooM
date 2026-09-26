"""Prepare the missing frozen inputs for the E2 agents-by-environments matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from expert.benchmark_contract import (
    FrozenTransitionBatch,
    load_frozen_transition_batch,
    save_frozen_transition_batch,
)
from expert.benchmark_manifest import build_manifest_entry, validate_benchmark_manifest
from mapf_cuda.experiments.frozen_inputs import build_frozen_standard_mapf_pool


E2_AGENT_COUNTS = (64, 128, 256, 512)
E2_ENV_COUNTS = (16, 64, 256, 1024)


def _base_pool_name(num_agents: int) -> str:
    return f"a2-random-d020-a{int(num_agents):03d}"


def _extended_pool_name(num_agents: int, env_count: int) -> str:
    return f"{_base_pool_name(num_agents)}-e{int(env_count)}"


def _find_entry(
    payload: dict[str, Any],
    *,
    pool_name: str | None = None,
    num_agents: int | None = None,
    minimum_envs: int | None = None,
) -> dict[str, Any]:
    matches = []
    for entry in payload.get("entries", []):
        if pool_name is not None and entry.get("pool_name") != pool_name:
            continue
        if num_agents is not None and int(entry.get("num_agents", -1)) != int(num_agents):
            continue
        if minimum_envs is not None and int(entry.get("num_envs", -1)) < int(minimum_envs):
            continue
        if entry.get("topology") != "random" or float(entry.get("density", -1.0)) != 0.2:
            continue
        matches.append(entry)
    if len(matches) != 1:
        raise ValueError(
            "E2 pool entry must resolve exactly once: "
            f"pool_name={pool_name!r}, num_agents={num_agents}, "
            f"minimum_envs={minimum_envs}, matches={len(matches)}"
        )
    return dict(matches[0])


def build_extended_pool(
    base_batch: FrozenTransitionBatch,
    *,
    env_count: int,
    pool_identity: str,
    pool_builder: Callable[..., FrozenTransitionBatch] = build_frozen_standard_mapf_pool,
) -> FrozenTransitionBatch:
    """Continue the base pool's deterministic seed range to ``env_count``."""

    if env_count < base_batch.num_envs:
        raise ValueError("extended E2 pool cannot be smaller than its base pool")
    seed_start = int(base_batch.instance_ids[0])
    expected_ids = tuple(range(seed_start, seed_start + int(env_count)))
    extended = pool_builder(
        topology="random",
        density=0.2,
        num_agents=base_batch.num_agents,
        seeds=expected_ids,
        horizon=base_batch.horizon,
        map_size=int(base_batch.grids.shape[1]),
        pool_identity=pool_identity,
    )
    if extended.instance_ids.tolist() != list(expected_ids):
        raise ValueError("extended E2 pool did not preserve the deterministic seed range")
    prefix = extended.take_envs(base_batch.num_envs)
    if prefix.semantic_sha256 != base_batch.semantic_sha256:
        raise ValueError("extended E2 pool does not preserve the accepted base prefix")
    return extended


def _copy_contract_metadata(
    entry: dict[str, Any], base_entry: dict[str, Any]
) -> dict[str, Any]:
    for key in ("role", "task_semantics", "topology_parameters", "compiled_profile"):
        if key in base_entry:
            entry[key] = base_entry[key]
    return entry


def prepare_e2_matrix_manifest(
    *,
    base_manifest_path: Path,
    existing_a256_e1024_manifest_path: Path,
    output_root: Path,
    agent_counts: Sequence[int] = E2_AGENT_COUNTS,
    env_counts: Sequence[int] = E2_ENV_COUNTS,
) -> Path:
    base_path = base_manifest_path.expanduser().resolve()
    a256_path = existing_a256_e1024_manifest_path.expanduser().resolve()
    root = output_root.expanduser().resolve()
    inputs_root = root / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)

    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    a256_payload = json.loads(a256_path.read_text(encoding="utf-8"))
    validate_benchmark_manifest(base_payload)
    validate_benchmark_manifest(a256_payload)

    selected_entries: list[dict[str, Any]] = []
    base_entries: dict[int, dict[str, Any]] = {}
    extended_entries: dict[int, dict[str, Any]] = {}

    for num_agents in agent_counts:
        base_entry = _find_entry(
            base_payload,
            pool_name=_base_pool_name(num_agents),
            num_agents=num_agents,
            minimum_envs=max(env for env in env_counts if env <= 256),
        )
        base_entries[int(num_agents)] = base_entry
        selected_entries.append(base_entry)

        maximum_envs = max(int(value) for value in env_counts)
        if maximum_envs <= int(base_entry["num_envs"]):
            extended_entries[int(num_agents)] = base_entry
            continue
        if int(num_agents) == 256:
            extended_entry = _find_entry(
                a256_payload,
                num_agents=256,
                minimum_envs=maximum_envs,
            )
        else:
            base_batch = load_frozen_transition_batch(base_entry["frozen_batch_path"])
            pool_name = _extended_pool_name(num_agents, maximum_envs)
            extended_batch = build_extended_pool(
                base_batch,
                env_count=maximum_envs,
                pool_identity=_base_pool_name(num_agents),
            )
            artifact_path = inputs_root / f"{pool_name}.npz"
            save_frozen_transition_batch(artifact_path, extended_batch)
            extended_entry = build_manifest_entry(
                pool_name=pool_name,
                topology="random",
                density=0.2,
                num_agents=int(num_agents),
                horizon=extended_batch.horizon,
                output_path=artifact_path,
                batch=extended_batch,
            )
            _copy_contract_metadata(extended_entry, base_entry)
        extended_entries[int(num_agents)] = extended_entry
        selected_entries.append(extended_entry)

    cells = []
    for num_agents in agent_counts:
        for num_envs in env_counts:
            source_entry = (
                base_entries[int(num_agents)]
                if int(num_envs) <= int(base_entries[int(num_agents)]["num_envs"])
                else extended_entries[int(num_agents)]
            )
            cells.append(
                {
                    "axis": "agents_by_envs",
                    "cell_id": f"matrix-a{int(num_agents):03d}-e{int(num_envs):04d}",
                    "topology": "random",
                    "density": 0.2,
                    "num_agents": int(num_agents),
                    "env_count": int(num_envs),
                    "pool_name": source_entry["pool_name"],
                }
            )

    payload = {
        "schema_version": base_payload["schema_version"],
        "entries": selected_entries,
        "a2_cells": cells,
        "e2_matrix": {
            "agent_counts": [int(value) for value in agent_counts],
            "env_counts": [int(value) for value in env_counts],
            "base_manifest": str(base_path),
            "existing_a256_e1024_manifest": str(a256_path),
            "semantics": "standard_mapf",
            "topology": "random",
            "density": 0.2,
        },
    }
    validate_benchmark_manifest(payload)
    manifest_path = root / "e2_matrix_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare E2 matrix frozen inputs.")
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--existing-a256-e1024-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    path = prepare_e2_matrix_manifest(
        base_manifest_path=args.base_manifest,
        existing_a256_e1024_manifest_path=args.existing_a256_e1024_manifest,
        output_root=args.output_root,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "E2_AGENT_COUNTS",
    "E2_ENV_COUNTS",
    "build_extended_pool",
    "prepare_e2_matrix_manifest",
]
