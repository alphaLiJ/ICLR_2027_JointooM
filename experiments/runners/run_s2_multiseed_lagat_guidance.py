"""Evaluate validation-best matched-budget MAGAT guides across training seeds.

The no-guide LaCAM row is shared across all training seeds.  Statistical
summaries retain the two-level design: instances are paired repetitions within
one trained checkpoint, while training seeds are the independent model-level
repetitions.  In particular, this runner never pools seed-instance rows into a
single inferential confidence interval.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from mapf_cuda.evaluation.lagat_export import export_lagat_checkpoint, sha256_file
from mapf_cuda.evaluation.lagat_solver import LaGATRunSpec, run_lagat_row


DEFAULT_TRAINING_ROOT = Path("artifacts/matched_training")
DEFAULT_MANIFEST = Path("artifacts/lagat_inputs/formal-main-64-128-manifest.json")
DEFAULT_BINARY = Path("artifacts/build-lagat/main")
SEEDS = (0, 1, 2)
METHODS = ("conventional_sync", "proposed_async")
METHOD_ORDER_BY_SEED = {
    0: ("proposed_async", "conventional_sync"),
    1: ("conventional_sync", "proposed_async"),
    2: ("proposed_async", "conventional_sync"),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _median_range(values: Iterable[float]) -> dict[str, float] | None:
    materialized = [float(value) for value in values]
    if not materialized:
        return None
    return {
        "median": statistics.median(materialized),
        "minimum": min(materialized),
        "maximum": max(materialized),
    }


def select_validation_best_checkpoint(
    training_root: Path, method: str, seed: int
) -> dict[str, Any]:
    """Audit one training row and resolve its validation-best checkpoint."""

    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    training_root = training_root.expanduser().resolve()
    result_path = training_root / "formal" / f"seed-{seed}" / method / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = _read_json(result_path)
    if result.get("status") != "ok":
        raise RuntimeError(f"training row is not complete: {result_path}")
    if result.get("method") != method or int(result.get("seed", -1)) != int(seed):
        raise RuntimeError(f"training row identity mismatch: {result_path}")
    candidates = result.get("saved_checkpoints", [])
    if not candidates:
        raise RuntimeError(f"no validation-selected checkpoints: {result_path}")
    selected = max(
        candidates,
        key=lambda item: (
            float(item["val_overall_accuracy"]),
            int(item["optimizer_step"]),
        ),
    )
    checkpoint_path = Path(selected["path"]).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    return {
        "method": method,
        "training_seed": int(seed),
        "training_result_path": str(result_path),
        "training_result_sha256": sha256_file(result_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "optimizer_step": int(selected["optimizer_step"]),
        "validation_overall_accuracy": float(selected["val_overall_accuracy"]),
        "validation_nonstay_accuracy": float(selected["val_nonstay_accuracy"]),
        "completed_optimizer_steps": int(result["num_optimizer_steps"]),
        "samples_processed": int(result["samples_processed"]),
        "actual_training_wall_s": float(result["total_wall_s"]),
        "wall_budget_reached": bool(result.get("wall_budget_reached", False)),
        "wall_budget_s": float(result["protocol"]["wall_budget_s"]),
        "initial_model_state_sha256": result["initial_model_state_sha256"],
        "validation_trajectory_sha256s": [
            item["sha256"]
            for item in result["protocol"]["validation_trajectories"]
        ],
    }


def audit_training_suite(
    training_root: Path, *, seeds: Sequence[int] = SEEDS
) -> dict[str, Any]:
    """Validate the paired training contract and summarize model-level effects."""

    training_root = training_root.expanduser().resolve()
    suite_status_path = training_root / "suite-status.json"
    preflight_path = training_root / "preflight-audit.json"
    suite_status = _read_json(suite_status_path)
    preflight = _read_json(preflight_path)
    if suite_status.get("status") != "training_complete":
        raise RuntimeError(f"training suite is not complete: {suite_status_path}")
    if not preflight.get("valid"):
        raise RuntimeError(f"training preflight parity failed: {preflight_path}")

    rows: list[dict[str, Any]] = []
    by_seed: dict[str, Any] = {}
    expected_validation: tuple[str, ...] | None = None
    expected_budget: float | None = None
    for seed in seeds:
        selected = {
            method: select_validation_best_checkpoint(training_root, method, seed)
            for method in METHODS
        }
        proposed = selected["proposed_async"]
        conventional = selected["conventional_sync"]
        if (
            proposed["initial_model_state_sha256"]
            != conventional["initial_model_state_sha256"]
        ):
            raise RuntimeError(f"initial model mismatch for training seed {seed}")
        for item in selected.values():
            validation = tuple(item["validation_trajectory_sha256s"])
            expected_validation = validation if expected_validation is None else expected_validation
            if validation != expected_validation:
                raise RuntimeError("validation trajectory set differs across training rows")
            budget = float(item["wall_budget_s"])
            expected_budget = budget if expected_budget is None else expected_budget
            if abs(budget - expected_budget) > 1e-9:
                raise RuntimeError("wall-clock budget differs across training rows")
            rows.append(item)
        by_seed[str(seed)] = {
            "methods": selected,
            "paired_training_effect": {
                "optimizer_step_ratio_proposed_over_conventional": (
                    proposed["completed_optimizer_steps"]
                    / conventional["completed_optimizer_steps"]
                ),
                "validation_accuracy_difference": (
                    proposed["validation_overall_accuracy"]
                    - conventional["validation_overall_accuracy"]
                ),
                "validation_accuracy_difference_percentage_points": 100.0
                * (
                    proposed["validation_overall_accuracy"]
                    - conventional["validation_overall_accuracy"]
                ),
            },
        }

    step_ratios = [
        by_seed[str(seed)]["paired_training_effect"][
            "optimizer_step_ratio_proposed_over_conventional"
        ]
        for seed in seeds
    ]
    accuracy_differences_pp = [
        by_seed[str(seed)]["paired_training_effect"][
            "validation_accuracy_difference_percentage_points"
        ]
        for seed in seeds
    ]
    return {
        "schema_version": 1,
        "training_root": str(training_root),
        "suite_status_path": str(suite_status_path),
        "suite_status_sha256": sha256_file(suite_status_path),
        "preflight_audit_path": str(preflight_path),
        "preflight_audit_sha256": sha256_file(preflight_path),
        "preflight_valid": True,
        "seeds": [int(seed) for seed in seeds],
        "wall_budget_s": expected_budget,
        "validation_trajectory_sha256s": list(expected_validation or ()),
        "selected_checkpoints": rows,
        "by_seed": by_seed,
        "across_training_seeds_descriptive": {
            "unit": "training_seed",
            "num_training_seeds": len(seeds),
            "optimizer_step_ratio_proposed_over_conventional": _median_range(
                step_ratios
            ),
            "validation_accuracy_difference_percentage_points": _median_range(
                accuracy_differences_pp
            ),
            "all_seeds_favor_proposed_validation_accuracy": all(
                value > 0 for value in accuracy_differences_pp
            ),
        },
    }


def prepare_deployment_models(
    training_audit: dict[str, Any], output_dir: Path
) -> dict[tuple[int, str], dict[str, Any]]:
    """Export or audit all six validation-best LaGAT deployment models."""

    model_dir = output_dir.expanduser().resolve() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[tuple[int, str], dict[str, Any]] = {}
    for selected in training_audit["selected_checkpoints"]:
        seed = int(selected["training_seed"])
        method = str(selected["method"])
        checkpoint = Path(selected["checkpoint_path"])
        output = model_dir / f"{method}-s{seed}-best-validation.pt"
        audit_path = output.with_suffix(output.suffix + ".audit.json")
        if output.is_file() or audit_path.is_file():
            if not output.is_file() or not audit_path.is_file():
                raise RuntimeError(f"partial deployment export exists: {output}")
            export_audit = _read_json(audit_path)
            if (
                export_audit.get("checkpoint_sha256")
                != selected["checkpoint_sha256"]
                or export_audit.get("output_sha256") != sha256_file(output)
            ):
                raise RuntimeError(f"stale deployment export: {output}")
        else:
            export_audit = export_lagat_checkpoint(checkpoint, output)
        prepared[(seed, method)] = {
            **selected,
            "deployment_model_path": str(output),
            "deployment_model_sha256": sha256_file(output),
            "export_audit_path": str(audit_path),
            "export_contract": export_audit["export_contract"],
        }
    return prepared


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    instance_ids: set[str] = set()
    for record in manifest.get("records", []):
        instance_id = str(record["instance_id"])
        if instance_id in instance_ids:
            raise RuntimeError(f"duplicate instance id: {instance_id}")
        instance_ids.add(instance_id)
        for prefix in ("map", "scenario"):
            path = Path(record[f"{prefix}_path"]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if sha256_file(path) != record[f"{prefix}_sha256"]:
                raise RuntimeError(f"{prefix} hash mismatch for {instance_id}")
    if not instance_ids:
        raise RuntimeError(f"manifest is empty: {manifest_path}")
    return manifest


def build_row_plan(
    instances: Sequence[dict[str, Any]], *, seeds: Sequence[int] = SEEDS
) -> list[dict[str, Any]]:
    """Build one shared baseline row plus two guide rows per seed and instance."""

    plan: list[dict[str, Any]] = []
    for instance in instances:
        instance_id = str(instance["instance_id"])
        plan.append(
            {
                "row_id": f"{instance_id}-baseline",
                "instance": instance,
                "method": "baseline",
                "training_seed": None,
            }
        )
        for seed in seeds:
            order = METHOD_ORDER_BY_SEED.get(int(seed), METHODS)
            for method in order:
                plan.append(
                    {
                        "row_id": f"{instance_id}-{method}-s{seed}",
                        "instance": instance,
                        "method": method,
                        "training_seed": int(seed),
                    }
                )
    return plan


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_median_interval(
    values: Sequence[float], *, samples: int, seed: int
) -> list[float] | None:
    if not values or samples <= 0:
        return None
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return [_percentile(medians, 0.025), _percentile(medians, 0.975)]


def _compare_rows(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Compare left against right; lower SoC is better and solved dominates failed."""

    right_by_instance = {row["instance_id"]: row for row in right}
    relative_deltas: list[float] = []
    all_instance_wins = all_instance_ties = all_instance_losses = unresolved = 0
    both_solved_wins = both_solved_ties = both_solved_losses = 0
    for left_row in left:
        right_row = right_by_instance[left_row["instance_id"]]
        left_solved = bool(left_row["metrics"]["solved"])
        right_solved = bool(right_row["metrics"]["solved"])
        if left_solved and not right_solved:
            all_instance_wins += 1
            continue
        if right_solved and not left_solved:
            all_instance_losses += 1
            continue
        if not left_solved and not right_solved:
            unresolved += 1
            continue
        left_soc = float(left_row["metrics"]["soc"])
        right_soc = float(right_row["metrics"]["soc"])
        delta = (left_soc - right_soc) / right_soc
        relative_deltas.append(delta)
        if left_soc < right_soc:
            all_instance_wins += 1
            both_solved_wins += 1
        elif left_soc > right_soc:
            all_instance_losses += 1
            both_solved_losses += 1
        else:
            all_instance_ties += 1
            both_solved_ties += 1
    return {
        "num_instances": len(left),
        "left_solved": sum(bool(row["metrics"]["solved"]) for row in left),
        "right_solved": sum(bool(row["metrics"]["solved"]) for row in right),
        "all_instance_outcomes": {
            "left_better": all_instance_wins,
            "tie": all_instance_ties,
            "left_worse": all_instance_losses,
            "both_unsolved": unresolved,
        },
        "both_solved": {
            "instances": len(relative_deltas),
            "median_relative_soc_delta": (
                None
                if not relative_deltas
                else statistics.median(relative_deltas)
            ),
            "instance_bootstrap_95pct_interval": _bootstrap_median_interval(
                relative_deltas,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "left_lower_soc": both_solved_wins,
            "ties": both_solved_ties,
            "left_higher_soc": both_solved_losses,
        },
    }


def summarize_multiseed(
    records: Sequence[dict[str, Any]],
    training_audit: dict[str, Any],
    *,
    seeds: Sequence[int] = SEEDS,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 12_081_200,
) -> dict[str, Any]:
    baselines = [row for row in records if row["method"] == "baseline"]
    expected_instances = {row["instance_id"] for row in baselines}
    if len(expected_instances) != len(baselines):
        raise RuntimeError("shared baseline must have exactly one row per instance")

    seed_summaries: dict[str, Any] = {}
    seed_effects: list[float] = []
    solve_differences: list[int] = []
    for seed_index, seed in enumerate(seeds):
        method_rows: dict[str, list[dict[str, Any]]] = {}
        for method in METHODS:
            rows = [
                row
                for row in records
                if row["method"] == method
                and int(row["training_seed"]) == int(seed)
            ]
            if {row["instance_id"] for row in rows} != expected_instances:
                raise RuntimeError(f"incomplete {method} rows for training seed {seed}")
            method_rows[method] = rows

        direct = _compare_rows(
            method_rows["proposed_async"],
            method_rows["conventional_sync"],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + 1000 * seed_index,
        )
        by_agents: dict[str, Any] = {}
        for num_agents in sorted(
            {int(row["num_agents"]) for row in method_rows["proposed_async"]}
        ):
            by_agents[str(num_agents)] = _compare_rows(
                [
                    row
                    for row in method_rows["proposed_async"]
                    if int(row["num_agents"]) == num_agents
                ],
                [
                    row
                    for row in method_rows["conventional_sync"]
                    if int(row["num_agents"]) == num_agents
                ],
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + 1000 * seed_index + num_agents,
            )
        versus_baseline = {
            method: _compare_rows(
                method_rows[method],
                baselines,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=(
                    bootstrap_seed
                    + 10_000
                    + 1000 * seed_index
                    + METHODS.index(method)
                ),
            )
            for method in METHODS
        }
        effect = direct["both_solved"]["median_relative_soc_delta"]
        if effect is not None:
            seed_effects.append(float(effect))
        solve_differences.append(direct["left_solved"] - direct["right_solved"])
        seed_summaries[str(seed)] = {
            "direct_proposed_vs_conventional": direct,
            "direct_by_num_agents": by_agents,
            "versus_shared_no_guide_baseline": versus_baseline,
        }

    return {
        "schema_version": 1,
        "experimental_unit_contract": {
            "training_level_unit": "training_seed",
            "task_level_unit": "held_out_instance_paired_within_training_seed",
            "shared_no_guide_baseline": True,
            "pooled_seed_instance_inference": False,
            "interpretation": (
                "Instance bootstrap intervals quantify task variation for one "
                "trained checkpoint. Across-seed results are descriptive over "
                "independent trained checkpoints and are not a 300-row pooled CI."
            ),
        },
        "num_shared_baseline_rows": len(baselines),
        "by_training_seed": seed_summaries,
        "across_training_seeds_descriptive": {
            "unit": "training_seed",
            "num_training_seeds": len(seeds),
            "median_relative_soc_delta_proposed_vs_conventional": _median_range(
                seed_effects
            ),
            "solve_count_difference_proposed_minus_conventional": _median_range(
                solve_differences
            ),
            "all_available_seed_medians_favor_proposed": bool(seed_effects)
            and all(effect < 0 for effect in seed_effects),
            "training_progress": training_audit[
                "across_training_seeds_descriptive"
            ],
        },
    }


def _load_or_run_row(
    *,
    binary: Path,
    spec: LaGATRunSpec,
    rows_dir: Path,
) -> dict[str, Any]:
    cached_path = rows_dir / spec.row_id / "result.json"
    if not cached_path.is_file():
        return run_lagat_row(binary, spec, rows_dir)
    cached = _read_json(cached_path)
    expected_model_sha = (
        None if spec.model_path is None else sha256_file(spec.model_path)
    )
    expected = {
        "binary_sha256": sha256_file(binary),
        "map_sha256": sha256_file(spec.map_path),
        "scenario_sha256": sha256_file(spec.scenario_path),
        "model_sha256": expected_model_sha,
    }
    for key, value in expected.items():
        if cached.get(key) != value:
            raise RuntimeError(f"cached row provenance mismatch ({key}): {cached_path}")
    cached_spec = cached["spec"]
    for key, value in {
        "row_id": spec.row_id,
        "map_path": str(spec.map_path),
        "scenario_path": str(spec.scenario_path),
        "num_agents": spec.num_agents,
        "planner_seed": spec.planner_seed,
        "time_limit_s": spec.time_limit_s,
        "model_path": None if spec.model_path is None else str(spec.model_path),
        "sampling": spec.sampling,
        "tau": spec.tau,
        "enable_communication_radius": spec.enable_communication_radius,
        "communication_radius": spec.communication_radius,
        "lns_refiners": spec.lns_refiners,
    }.items():
        if cached_spec.get(key) != value:
            raise RuntimeError(f"cached row protocol mismatch ({key}): {cached_path}")
    return cached


def run_suite(
    *,
    binary: Path,
    manifest_path: Path,
    training_root: Path,
    output_dir: Path,
    seeds: Sequence[int] = SEEDS,
    time_limit_s: float = 30.0,
    lns_refiners: int | None = None,
    bootstrap_samples: int = 10_000,
    max_instances: int | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    binary = binary.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if time_limit_s <= 0:
        raise ValueError("time_limit_s must be positive")
    if max_instances is not None and max_instances <= 0:
        raise ValueError("max_instances must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_audit = audit_training_suite(training_root, seeds=seeds)
    manifest = validate_manifest(manifest_path)
    instances = list(manifest["records"])
    if max_instances is not None:
        instances = instances[:max_instances]
    models = prepare_deployment_models(training_audit, output_dir)

    protocol = {
        "schema_version": 1,
        "experiment": "S2-matched-budget-MAGAT-multiseed-LaCAM-guidance",
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "num_instances": len(instances),
        "seeds": [int(seed) for seed in seeds],
        "methods": list(METHODS),
        "checkpoint_selection": "validation_best",
        "shared_no_guide_baseline": True,
        "time_limit_s": float(time_limit_s),
        "lns_refiners": lns_refiners,
        "bootstrap_samples_per_training_seed": int(bootstrap_samples),
        "pooled_seed_instance_inference": False,
        "training_audit_sha256": None,
    }
    _write_json(output_dir / "training-audit.json", training_audit)
    protocol["training_audit_sha256"] = sha256_file(
        output_dir / "training-audit.json"
    )
    existing_protocol_path = output_dir / "protocol.json"
    if existing_protocol_path.is_file() and _read_json(existing_protocol_path) != protocol:
        raise RuntimeError(f"output directory has a different protocol: {output_dir}")
    _write_json(existing_protocol_path, protocol)
    _write_json(
        output_dir / "model-inventory.json",
        {
            f"{method}-s{seed}": inventory
            for (seed, method), inventory in sorted(models.items())
        },
    )

    plan = build_row_plan(instances, seeds=seeds)
    _write_json(
        output_dir / "row-plan.json",
        {
            "schema_version": 1,
            "num_rows": len(plan),
            "num_shared_baseline_rows": len(instances),
            "num_guided_rows": len(plan) - len(instances),
            "rows": [
                {
                    "row_id": row["row_id"],
                    "instance_id": row["instance"]["instance_id"],
                    "method": row["method"],
                    "training_seed": row["training_seed"],
                }
                for row in plan
            ],
        },
    )
    if prepare_only:
        result = {
            "status": "prepared",
            "output_dir": str(output_dir),
            "num_rows": len(plan),
            "num_models": len(models),
        }
        _write_json(output_dir / "suite-status.json", result)
        return result

    records: list[dict[str, Any]] = []
    rows_dir = output_dir / "rows"
    for row_index, row in enumerate(plan):
        instance = row["instance"]
        method = row["method"]
        training_seed = row["training_seed"]
        model_path = (
            None
            if method == "baseline"
            else Path(models[(int(training_seed), method)]["deployment_model_path"])
        )
        spec = LaGATRunSpec(
            row_id=row["row_id"],
            map_path=Path(instance["map_path"]),
            scenario_path=Path(instance["scenario_path"]),
            num_agents=int(instance["num_agents"]),
            planner_seed=int(instance["task_seed"]) % 2_147_483_647,
            time_limit_s=float(time_limit_s),
            model_path=model_path,
            lns_refiners=lns_refiners,
        )
        raw = _load_or_run_row(binary=binary, spec=spec, rows_dir=rows_dir)
        records.append(
            {
                "row_id": row["row_id"],
                "instance_id": instance["instance_id"],
                "map_name": instance["map_name"],
                "num_agents": int(instance["num_agents"]),
                "method": method,
                "training_seed": training_seed,
                "metrics": raw["metrics"],
                "process_wall_s": raw["process_wall_s"],
                "row_result_path": str(rows_dir / row["row_id"] / "result.json"),
            }
        )
        _write_json(
            output_dir / "suite-status.json",
            {
                "status": "running",
                "completed_rows": row_index + 1,
                "total_rows": len(plan),
                "last_row_id": row["row_id"],
            },
        )
    registry_path = output_dir / "registry.jsonl"
    registry_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize_multiseed(
        records,
        training_audit,
        seeds=seeds,
        bootstrap_samples=bootstrap_samples,
    )
    summary.update(
        {
            "protocol": protocol,
            "training_audit_path": str(output_dir / "training-audit.json"),
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
        }
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "suite-status.json",
        {
            "status": "complete",
            "completed_rows": len(plan),
            "total_rows": len(plan),
            "summary_path": str(output_dir / "summary.json"),
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--time-limit-s", type=float, default=30.0)
    parser.add_argument("--lns-refiners", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_suite(
        binary=args.binary,
        manifest_path=args.manifest,
        training_root=args.training_root,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        time_limit_s=args.time_limit_s,
        lns_refiners=args.lns_refiners,
        bootstrap_samples=args.bootstrap_samples,
        max_instances=args.max_instances,
        prepare_only=args.prepare_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
