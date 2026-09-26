"""Freeze and run the S2 MAGAT-guided LaCAM downstream-utility pilot."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import yaml

from mapf_cuda.evaluation.lagat_export import export_lagat_checkpoint, sha256_file
from mapf_cuda.evaluation.lagat_solver import LaGATRunSpec, run_lagat_row


DEFAULT_CHECKPOINT_ROOT = Path("artifacts/magat_checkpoints")
MODES = ("strong_online", "compact_sync", "proposed_async")
MODE_LABELS = {
    "strong_online": "strong",
    "compact_sync": "compact",
    "proposed_async": "proposed",
}


def _largest_free_component(lines: list[str]) -> list[tuple[int, int]]:
    height, width = len(lines), len(lines[0])
    free = {
        (row, col)
        for row, line in enumerate(lines)
        for col, value in enumerate(line)
        if value == "."
    }
    components: list[list[tuple[int, int]]] = []
    while free:
        start = free.pop()
        queue = deque((start,))
        component = [start]
        while queue:
            row, col = queue.popleft()
            for neighbour in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbour in free:
                    free.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(component)
    return max(components, key=len)


def _sample_task(
    cells: list[tuple[int, int]],
    *,
    num_agents: int,
    minimum_manhattan: int,
    rng: random.Random,
    attempts: int = 1000,
    disjoint_start_goal_sets: bool = True,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    required_cells = 2 * num_agents if disjoint_start_goal_sets else num_agents
    if len(cells) < required_cells:
        raise ValueError(
            f"component has {len(cells)} cells but {required_cells} are required"
        )
    for _ in range(attempts):
        shuffled = cells.copy()
        rng.shuffle(shuffled)
        starts = shuffled[:num_agents]
        candidates = (
            shuffled[num_agents:] if disjoint_start_goal_sets else shuffled.copy()
        )
        goals: list[tuple[int, int]] = []
        failed = False
        for start in starts:
            eligible = [
                index
                for index, goal in enumerate(candidates)
                if abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                >= minimum_manhattan
            ]
            if not eligible:
                failed = True
                break
            selected = eligible[rng.randrange(len(eligible))]
            goals.append(candidates.pop(selected))
        if not failed:
            return starts, goals
    raise RuntimeError(
        f"could not sample {num_agents} start/goal pairs after {attempts} attempts"
    )


def freeze_inputs(
    maps_yaml: Path,
    output_dir: Path,
    *,
    map_names: Iterable[str],
    agent_counts: Iterable[int],
    instances_per_setting: int,
    seed_start: int,
    minimum_manhattan: int,
    disjoint_start_goal_sets: bool = True,
) -> dict[str, Any]:
    maps_yaml = maps_yaml.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    maps = yaml.safe_load(maps_yaml.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for map_index, map_name in enumerate(map_names):
        if map_name not in maps:
            raise KeyError(f"map {map_name!r} is absent from {maps_yaml}")
        lines = [line for line in str(maps[map_name]).splitlines() if line]
        if len({len(line) for line in lines}) != 1:
            raise ValueError(f"map {map_name} is not rectangular")
        height, width = len(lines), len(lines[0])
        map_path = output_dir / f"{map_name}.map"
        map_path.write_text(
            "\n".join(
                ("type octile", f"height {height}", f"width {width}", "map", *[line.replace("#", "@") for line in lines])
            )
            + "\n",
            encoding="utf-8",
        )
        component = _largest_free_component(lines)

        for num_agents in agent_counts:
            for instance_index in range(instances_per_setting):
                task_seed = seed_start + map_index * 100_000 + num_agents * 100 + instance_index
                starts, goals = _sample_task(
                    component,
                    num_agents=num_agents,
                    minimum_manhattan=minimum_manhattan,
                    rng=random.Random(task_seed),
                    disjoint_start_goal_sets=disjoint_start_goal_sets,
                )
                scenario_path = output_dir / (
                    f"{map_name}-a{num_agents:03d}-i{instance_index:03d}.scen"
                )
                scenario_lines = ["version 1"]
                for agent_id, (start, goal) in enumerate(zip(starts, goals)):
                    scenario_lines.append(
                        "\t".join(
                            map(
                                str,
                                (
                                    agent_id,
                                    map_path.name,
                                    width,
                                    height,
                                    start[1],
                                    start[0],
                                    goal[1],
                                    goal[0],
                                    0,
                                ),
                            )
                        )
                    )
                scenario_path.write_text("\n".join(scenario_lines) + "\n", encoding="utf-8")
                records.append(
                    {
                        "instance_id": f"{map_name}-a{num_agents:03d}-i{instance_index:03d}",
                        "map_name": map_name,
                        "num_agents": num_agents,
                        "instance_index": instance_index,
                        "task_seed": task_seed,
                        "minimum_manhattan": minimum_manhattan,
                        "disjoint_start_goal_sets": disjoint_start_goal_sets,
                        "map_path": str(map_path),
                        "map_sha256": sha256_file(map_path),
                        "scenario_path": str(scenario_path),
                        "scenario_sha256": sha256_file(scenario_path),
                    }
                )

    manifest = {
        "schema_version": 1,
        "maps_yaml": str(maps_yaml),
        "maps_yaml_sha256": sha256_file(maps_yaml),
        "disjoint_start_goal_sets": disjoint_start_goal_sets,
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def best_validation_checkpoint(root: Path, mode: str, seed: int) -> Path:
    result_path = root / f"{mode}-s{seed}" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = result.get("saved_checkpoints", [])
    if not candidates:
        raise RuntimeError(f"no validation-selected checkpoints in {result_path}")
    selected = max(candidates, key=lambda item: float(item["val_overall_accuracy"]))
    checkpoint = Path(selected["path"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def latest_checkpoint(root: Path, mode: str, seed: int) -> Path:
    checkpoint = (
        root / f"{mode}-s{seed}" / "checkpoints" / "ckpt_latest.pt"
    ).expanduser().resolve()
    if checkpoint.is_file():
        return checkpoint

    # Wall-capped runs stop between periodic checkpoint intervals and force a
    # final ranked checkpoint.  Those runs intentionally do not manufacture a
    # ``ckpt_latest.pt`` alias, so resolve the greatest completed optimizer
    # step recorded in the row result instead.
    result_path = root / f"{mode}-s{seed}" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    candidates = result.get("saved_checkpoints", [])
    if not candidates:
        raise RuntimeError(f"no checkpoints recorded in {result_path}")
    selected = max(candidates, key=lambda item: int(item["optimizer_step"]))
    checkpoint = Path(selected["path"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def prepare_models(
    checkpoint_root: Path,
    output_dir: Path,
    *,
    seed: int,
    checkpoint_selection: str,
) -> dict[str, Path]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, Path] = {}
    for mode in MODES:
        if checkpoint_selection == "latest":
            checkpoint = latest_checkpoint(checkpoint_root, mode, seed)
        elif checkpoint_selection == "best-validation":
            checkpoint = best_validation_checkpoint(checkpoint_root, mode, seed)
        else:
            raise ValueError(
                f"unsupported checkpoint selection: {checkpoint_selection}"
            )
        output = model_dir / f"{mode}-s{seed}-{checkpoint_selection}.pt"
        if not output.is_file():
            export_lagat_checkpoint(checkpoint, output)
        models[mode] = output
    return models


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["policy"]].append(record)
    policies: dict[str, Any] = {}
    for policy, rows in grouped.items():
        solved = [row for row in rows if row["metrics"]["solved"]]
        policies[policy] = {
            "rows": len(rows),
            "solved": len(solved),
            "solve_rate": len(solved) / len(rows),
            "median_soc_solved": (
                None
                if not solved
                else statistics.median(row["metrics"]["soc"] for row in solved)
            ),
            "median_solver_ms_wo_model_load": statistics.median(
                row["metrics"]["comp_time_ms_wo_model_load"] for row in rows
            ),
        }
    by_agents: dict[str, Any] = {}
    for num_agents in sorted({int(row["num_agents"]) for row in records}):
        by_agents[str(num_agents)] = {}
        for policy in sorted(grouped):
            rows = [
                row
                for row in grouped[policy]
                if int(row["num_agents"]) == num_agents
            ]
            solved = [row for row in rows if row["metrics"]["solved"]]
            normalized_soc = [
                row["metrics"]["soc"] / row["metrics"]["soc_lb"]
                for row in solved
            ]
            by_agents[str(num_agents)][policy] = {
                "rows": len(rows),
                "solved": len(solved),
                "solve_rate": len(solved) / len(rows),
                "median_soc_over_lower_bound_solved": (
                    None if not normalized_soc else statistics.median(normalized_soc)
                ),
            }

    baselines = {
        row["instance_id"]: row for row in records if row["policy"] == "baseline"
    }
    paired: dict[str, Any] = {}
    for policy in sorted(set(grouped) - {"baseline"}):
        deltas = []
        wins = ties = losses = 0
        for row in grouped[policy]:
            baseline = baselines[row["instance_id"]]
            if not row["metrics"]["solved"] or not baseline["metrics"]["solved"]:
                continue
            delta = (
                row["metrics"]["soc"] - baseline["metrics"]["soc"]
            ) / baseline["metrics"]["soc"]
            deltas.append(delta)
            if delta < 0:
                wins += 1
            elif delta > 0:
                losses += 1
            else:
                ties += 1
        paired[policy] = {
            "paired_solved_instances": len(deltas),
            "median_relative_soc_delta": (
                None if not deltas else statistics.median(deltas)
            ),
            "wins_lower_soc": wins,
            "ties": ties,
            "losses_higher_soc": losses,
        }
    return {
        "schema_version": 1,
        "policies": policies,
        "by_num_agents": by_agents,
        "paired_vs_baseline": paired,
    }


def run_pilot(
    binary: Path,
    manifest_path: Path,
    checkpoint_root: Path,
    output_dir: Path,
    *,
    training_seed: int,
    time_limit_s: float,
    lns_refiners: int | None,
    checkpoint_selection: str,
    named_models: dict[str, Path] | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if named_models:
        models = {
            label: path.expanduser().resolve()
            for label, path in named_models.items()
        }
        if "baseline" in models:
            raise ValueError("the model label 'baseline' is reserved")
        missing = [str(path) for path in models.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing deployment models: {missing}")
        policies: list[tuple[str, Path | None]] = [("baseline", None), *models.items()]
    else:
        prepared = prepare_models(
            checkpoint_root,
            output_dir,
            seed=training_seed,
            checkpoint_selection=checkpoint_selection,
        )
        models = {MODE_LABELS[mode]: prepared[mode] for mode in MODES}
        policies = [("baseline", None), *models.items()]
    records: list[dict[str, Any]] = []
    registry = output_dir / "registry.jsonl"
    registered_row_ids: set[str] = set()
    if registry.is_file():
        registered_row_ids = {
            json.loads(line)["row_id"]
            for line in registry.read_text(encoding="utf-8").splitlines()
            if line
        }

    for instance in manifest["records"]:
        for policy, model_path in policies:
            row_id = f"{instance['instance_id']}-{policy}"
            spec = LaGATRunSpec(
                row_id=row_id,
                map_path=Path(instance["map_path"]),
                scenario_path=Path(instance["scenario_path"]),
                num_agents=int(instance["num_agents"]),
                planner_seed=int(instance["task_seed"]) % 2_147_483_647,
                time_limit_s=time_limit_s,
                model_path=model_path,
                lns_refiners=lns_refiners,
            )
            cached_result = output_dir / "rows" / row_id / "result.json"
            result = (
                json.loads(cached_result.read_text(encoding="utf-8"))
                if cached_result.is_file()
                else run_lagat_row(binary, spec, output_dir / "rows")
            )
            record = {
                "row_id": row_id,
                "instance_id": instance["instance_id"],
                "map_name": instance["map_name"],
                "num_agents": instance["num_agents"],
                "policy": policy,
                "training_seed": None if policy == "baseline" else training_seed,
                "metrics": result["metrics"],
                "process_wall_s": result["process_wall_s"],
                "row_result_path": str(output_dir / "rows" / row_id / "result.json"),
            }
            records.append(record)
            if row_id not in registered_row_ids:
                with registry.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                registered_row_ids.add(row_id)

    summary = summarize(records)
    summary.update(
        {
            "binary": str(binary.expanduser().resolve()),
            "binary_sha256": sha256_file(binary.expanduser().resolve()),
            "manifest": str(manifest_path.expanduser().resolve()),
            "manifest_sha256": sha256_file(manifest_path.expanduser().resolve()),
            "checkpoint_root": str(checkpoint_root.expanduser().resolve()),
            "models": {
                label: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for label, path in models.items()
            },
            "training_seed": training_seed,
            "checkpoint_selection": checkpoint_selection,
            "time_limit_s": time_limit_s,
            "lns_refiners": lns_refiners,
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--maps-yaml", type=Path, default=Path("maps/maps.yaml"))
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--map-names", nargs="+", default=("mazes-s2_wc8_od25", "mazes-s3_wc6_od35"))
    freeze.add_argument("--agent-counts", nargs="+", type=int, default=(64, 128))
    freeze.add_argument("--instances-per-setting", type=int, default=2)
    freeze.add_argument("--seed-start", type=int, default=8_210_000)
    freeze.add_argument("--minimum-manhattan", type=int, default=10)
    freeze.add_argument(
        "--allow-start-goal-overlap",
        action="store_true",
        help="use standard MAPF semantics: starts and goals are each unique, but the two sets may overlap",
    )

    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--binary", type=Path, required=True)
    pilot.add_argument("--manifest", type=Path, required=True)
    pilot.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument("--training-seed", type=int, default=0)
    pilot.add_argument(
        "--checkpoint-selection",
        choices=("latest", "best-validation"),
        default="latest",
    )
    pilot.add_argument("--time-limit-s", type=float, default=3.0)
    pilot.add_argument("--lns-refiners", type=int)
    pilot.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=TORCHSCRIPT_PATH",
        help="evaluate explicit deployment models instead of the three-mode checkpoint root",
    )
    args = parser.parse_args()

    if args.command == "freeze":
        result = freeze_inputs(
            args.maps_yaml,
            args.output_dir,
            map_names=args.map_names,
            agent_counts=args.agent_counts,
            instances_per_setting=args.instances_per_setting,
            seed_start=args.seed_start,
            minimum_manhattan=args.minimum_manhattan,
            disjoint_start_goal_sets=not args.allow_start_goal_overlap,
        )
    else:
        named_models: dict[str, Path] = {}
        for value in args.model:
            if "=" not in value:
                raise ValueError(f"invalid --model value {value!r}; expected LABEL=PATH")
            label, path = value.split("=", 1)
            if not label or label in named_models:
                raise ValueError(f"invalid or duplicate model label: {label!r}")
            named_models[label] = Path(path)
        result = run_pilot(
            args.binary,
            args.manifest,
            args.checkpoint_root,
            args.output_dir,
            training_seed=args.training_seed,
            time_limit_s=args.time_limit_s,
            lns_refiners=args.lns_refiners,
            checkpoint_selection=args.checkpoint_selection,
            named_models=named_models or None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
