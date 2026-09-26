"""Freeze the formal S2-A held-out maze and S2-B warehouse inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.runners.run_s2_lagat_guidance import freeze_inputs
from mapf_cuda.evaluation.lagat_export import sha256_file


TEST_MAZES = (
    "test-mazes-s40_wc4_od30",
    "test-mazes-s41_wc5_od50",
    "test-mazes-s42_wc7_od30",
    "test-mazes-s43_wc2_od45",
    "test-mazes-s44_wc2_od30",
    "test-mazes-s45_wc4_od55",
    "test-mazes-s46_wc2_od55",
    "test-mazes-s47_wc2_od25",
    "test-mazes-s48_wc3_od65",
    "test-mazes-s49_wc2_od50",
)
WAREHOUSE_SETTINGS = {
    "warehouse-official-r5c2": (5, 2),
    "warehouse-scaled-r8c4": (8, 4),
    "warehouse-scaled-r10c4": (10, 4),
    "warehouse-scaled-r8c5": (8, 5),
    "warehouse-scaled-r10c5": (10, 5),
}


def generate_official_style_warehouse(
    *,
    num_wall_rows: int,
    num_wall_cols: int,
    wall_width: int = 8,
    wall_height: int = 2,
    side_pad: int = 3,
    horizontal_gap: int = 1,
    vertical_gap: int = 1,
) -> str:
    used_height = vertical_gap * (num_wall_rows + 1) + wall_height * num_wall_rows
    used_width = (
        side_pad * 2
        + wall_width * num_wall_cols
        + horizontal_gap * (num_wall_cols - 1)
    )
    size = max(used_height, used_width)
    grid = np.zeros((size, size), dtype=np.uint8)
    for row in range(num_wall_rows):
        row_start = vertical_gap * (row + 1) + wall_height * row
        for col in range(num_wall_cols):
            col_start = side_pad + col * (wall_width + horizontal_gap)
            grid[
                row_start : row_start + wall_height,
                col_start : col_start + wall_width,
            ] = 1
    # Match the upstream generator's block_extra_space=True behavior.
    grid[used_height:, :] = 1
    grid[:, used_width:] = 1
    return "\n".join(
        "".join("#" if value else "." for value in row) for row in grid
    )


def _write_warehouse_yaml(source_maps: Path, output: Path) -> tuple[str, ...]:
    source = yaml.safe_load(source_maps.read_text(encoding="utf-8"))
    maps = {
        name: generate_official_style_warehouse(
            num_wall_rows=rows,
            num_wall_cols=cols,
        )
        for name, (rows, cols) in WAREHOUSE_SETTINGS.items()
    }
    maps["kiva"] = source["kiva"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(maps, sort_keys=False), encoding="utf-8")
    return tuple(maps)


def _read_movingai_map(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = lines.index("map")
    return lines[marker + 1 :]


def audit_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[tuple[str, int]] = Counter()
    overlaps = 0
    for record in manifest["records"]:
        map_path = Path(record["map_path"])
        scenario_path = Path(record["scenario_path"])
        if sha256_file(map_path) != record["map_sha256"]:
            raise RuntimeError(f"map hash mismatch: {map_path}")
        if sha256_file(scenario_path) != record["scenario_sha256"]:
            raise RuntimeError(f"scenario hash mismatch: {scenario_path}")
        grid = _read_movingai_map(map_path)
        free = {
            (row, col)
            for row, line in enumerate(grid)
            for col, value in enumerate(line)
            if value == "."
        }
        rows = [
            line.split("\t")
            for line in scenario_path.read_text(encoding="utf-8").splitlines()[1:]
        ]
        if len(rows) != int(record["num_agents"]):
            raise RuntimeError(f"agent count mismatch: {scenario_path}")
        starts = [(int(row[5]), int(row[4])) for row in rows]
        goals = [(int(row[7]), int(row[6])) for row in rows]
        if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
            raise RuntimeError(f"non-unique starts or goals: {scenario_path}")
        if not set(starts + goals) <= free:
            raise RuntimeError(f"task uses blocked cells: {scenario_path}")
        if any(
            abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            < int(record["minimum_manhattan"])
            for start, goal in zip(starts, goals)
        ):
            raise RuntimeError(f"minimum distance violated: {scenario_path}")
        reachable = {starts[0]}
        queue = deque((starts[0],))
        while queue:
            row, col = queue.popleft()
            for neighbour in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbour in free and neighbour not in reachable:
                    reachable.add(neighbour)
                    queue.append(neighbour)
        if not set(starts + goals) <= reachable:
            raise RuntimeError(f"unreachable task cells: {scenario_path}")
        overlaps += len(set(starts) & set(goals))
        counts[(str(record["map_name"]), int(record["num_agents"]))] += 1
    return {
        "valid": True,
        "scenario_count": len(manifest["records"]),
        "map_agent_setting_count": len(counts),
        "instances_per_setting": sorted(set(counts.values())),
        "cross_set_overlap_cells": overlaps,
    }


def _write_subset_manifest(
    manifest: dict[str, Any],
    output: Path,
    *,
    map_names: tuple[str, ...],
    max_instance_index: int,
) -> dict[str, Any]:
    subset = {
        **manifest,
        "records": [
            record
            for record in manifest["records"]
            if record["map_name"] in map_names
            and int(record["instance_index"]) < max_instance_index
        ],
        "subset_of": "formal_manifest",
    }
    output.write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return subset


def _write_filtered_manifest(
    manifest: dict[str, Any],
    output: Path,
    *,
    agent_counts: tuple[int, ...] | None = None,
    include_maps: tuple[str, ...] | None = None,
    exclude_maps: tuple[str, ...] = (),
) -> dict[str, Any]:
    subset = {
        **manifest,
        "records": [
            record
            for record in manifest["records"]
            if (agent_counts is None or int(record["num_agents"]) in agent_counts)
            and (include_maps is None or record["map_name"] in include_maps)
            and record["map_name"] not in exclude_maps
        ],
        "subset_of": "formal_manifest",
    }
    output.write_text(
        json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return subset


def prepare_ab_inputs(*, maps_yaml: Path, output_root: Path) -> dict[str, Any]:
    maps_yaml = maps_yaml.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    maze = freeze_inputs(
        maps_yaml,
        output_root / "a-heldout-maze",
        map_names=TEST_MAZES,
        agent_counts=(64, 128, 256),
        instances_per_setting=5,
        seed_start=9_100_000,
        minimum_manhattan=10,
        disjoint_start_goal_sets=False,
    )

    warehouse_yaml = output_root / "warehouse-maps.yaml"
    warehouse_names = _write_warehouse_yaml(maps_yaml, warehouse_yaml)
    warehouse_low = freeze_inputs(
        warehouse_yaml,
        output_root / "b-warehouse",
        map_names=warehouse_names,
        agent_counts=(64, 128),
        instances_per_setting=5,
        seed_start=9_200_000,
        minimum_manhattan=10,
        disjoint_start_goal_sets=False,
    )
    scalable_names = tuple(
        name for name in warehouse_names if name != "warehouse-official-r5c2"
    )
    warehouse_high = freeze_inputs(
        warehouse_yaml,
        output_root / "b-warehouse",
        map_names=scalable_names,
        agent_counts=(256,),
        instances_per_setting=5,
        seed_start=9_300_000,
        minimum_manhattan=10,
        disjoint_start_goal_sets=False,
    )
    warehouse = {
        **warehouse_low,
        "records": warehouse_low["records"] + warehouse_high["records"],
        "capacity_rule": {
            "warehouse-official-r5c2": "64_and_128_only",
            "all_other_maps": "64_128_256",
        },
    }
    warehouse_manifest = output_root / "b-warehouse" / "manifest.json"
    warehouse_manifest.write_text(
        json.dumps(warehouse, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    maze_audit = audit_manifest(maze)
    warehouse_audit = audit_manifest(warehouse)
    (output_root / "a-heldout-maze" / "audit.json").write_text(
        json.dumps(maze_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "b-warehouse" / "audit.json").write_text(
        json.dumps(warehouse_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    maze_pilot_path = output_root / "a-heldout-maze" / "pilot-manifest.json"
    warehouse_pilot_path = output_root / "b-warehouse" / "pilot-manifest.json"
    maze_pilot = _write_subset_manifest(
        maze,
        maze_pilot_path,
        map_names=TEST_MAZES[:2],
        max_instance_index=2,
    )
    warehouse_pilot = _write_subset_manifest(
        warehouse,
        warehouse_pilot_path,
        map_names=("warehouse-official-r5c2", "kiva"),
        max_instance_index=2,
    )
    formal_subsets = {
        "a_main_64_128": _write_filtered_manifest(
            maze,
            output_root / "a-heldout-maze" / "formal-main-64-128-manifest.json",
            agent_counts=(64, 128),
        ),
        "a_stress_256": _write_filtered_manifest(
            maze,
            output_root / "a-heldout-maze" / "formal-stress-256-manifest.json",
            agent_counts=(256,),
        ),
        "b_exact": _write_filtered_manifest(
            warehouse,
            output_root / "b-warehouse" / "formal-exact-manifest.json",
            include_maps=("warehouse-official-r5c2",),
        ),
        "b_scaled": _write_filtered_manifest(
            warehouse,
            output_root / "b-warehouse" / "formal-scaled-manifest.json",
            exclude_maps=("warehouse-official-r5c2",),
        ),
    }

    summary = {
        "schema_version": 1,
        "a": {
            "manifest": str(output_root / "a-heldout-maze" / "manifest.json"),
            "manifest_sha256": sha256_file(
                output_root / "a-heldout-maze" / "manifest.json"
            ),
            "scenarios": len(maze["records"]),
            "audit": maze_audit,
            "pilot_manifest": str(maze_pilot_path),
            "pilot_manifest_sha256": sha256_file(maze_pilot_path),
            "pilot_scenarios": len(maze_pilot["records"]),
        },
        "b": {
            "manifest": str(warehouse_manifest),
            "manifest_sha256": sha256_file(warehouse_manifest),
            "scenarios": len(warehouse["records"]),
            "warehouse_maps_yaml": str(warehouse_yaml),
            "warehouse_maps_yaml_sha256": sha256_file(warehouse_yaml),
            "audit": warehouse_audit,
            "pilot_manifest": str(warehouse_pilot_path),
            "pilot_manifest_sha256": sha256_file(warehouse_pilot_path),
            "pilot_scenarios": len(warehouse_pilot["records"]),
        },
        "formal_subsets": {
            label: {
                "manifest": str(
                    (
                        output_root / "a-heldout-maze"
                        if label.startswith("a_")
                        else output_root / "b-warehouse"
                    )
                    / {
                        "a_main_64_128": "formal-main-64-128-manifest.json",
                        "a_stress_256": "formal-stress-256-manifest.json",
                        "b_exact": "formal-exact-manifest.json",
                        "b_scaled": "formal-scaled-manifest.json",
                    }[label]
                ),
                "scenarios": len(manifest["records"]),
            }
            for label, manifest in formal_subsets.items()
        },
    }
    (output_root / "ab-input-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-yaml", type=Path, default=Path("maps/maps.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_ab_inputs(maps_yaml=args.maps_yaml, output_root=args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
