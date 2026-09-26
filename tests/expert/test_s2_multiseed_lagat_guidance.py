from __future__ import annotations

import json

import pytest

from experiments.runners.run_s2_multiseed_lagat_guidance import (
    _load_or_run_row,
    build_row_plan,
    select_validation_best_checkpoint,
    summarize_multiseed,
)
from mapf_cuda.evaluation.lagat_solver import LaGATRunSpec
from mapf_cuda.evaluation.lagat_export import sha256_file


def _training_audit():
    return {
        "across_training_seeds_descriptive": {
            "unit": "training_seed",
            "num_training_seeds": 2,
        }
    }


def _solver_row(instance, method, seed, soc, *, solved=True, agents=64):
    return {
        "instance_id": instance,
        "num_agents": agents,
        "method": method,
        "training_seed": seed,
        "metrics": {"solved": solved, "soc": soc},
    }


def test_validation_best_uses_nested_multiseed_training_layout(tmp_path):
    row = tmp_path / "formal" / "seed-2" / "proposed_async"
    checkpoints = row / "checkpoints"
    checkpoints.mkdir(parents=True)
    early = checkpoints / "early.pt"
    best = checkpoints / "best.pt"
    final = checkpoints / "final.pt"
    for path in (early, best, final):
        path.write_bytes(path.name.encode())
    (row / "result.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "method": "proposed_async",
                "seed": 2,
                "num_optimizer_steps": 100,
                "samples_processed": 200,
                "total_wall_s": 9.0,
                "wall_budget_reached": False,
                "initial_model_state_sha256": "model",
                "protocol": {
                    "wall_budget_s": 10.0,
                    "validation_trajectories": [{"sha256": "validation"}],
                },
                "saved_checkpoints": [
                    {
                        "path": str(early),
                        "optimizer_step": 10,
                        "val_overall_accuracy": 0.7,
                        "val_nonstay_accuracy": 0.6,
                    },
                    {
                        "path": str(best),
                        "optimizer_step": 80,
                        "val_overall_accuracy": 0.8,
                        "val_nonstay_accuracy": 0.7,
                    },
                    {
                        "path": str(final),
                        "optimizer_step": 100,
                        "val_overall_accuracy": 0.79,
                        "val_nonstay_accuracy": 0.69,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    selected = select_validation_best_checkpoint(tmp_path, "proposed_async", 2)

    assert selected["checkpoint_path"] == str(best.resolve())
    assert selected["optimizer_step"] == 80
    assert selected["validation_overall_accuracy"] == 0.8


def test_row_plan_shares_one_no_guide_baseline_across_training_seeds():
    instances = [
        {"instance_id": "a", "num_agents": 64},
        {"instance_id": "b", "num_agents": 128},
    ]

    plan = build_row_plan(instances, seeds=(0, 1, 2))

    baseline = [row for row in plan if row["method"] == "baseline"]
    guided = [row for row in plan if row["method"] != "baseline"]
    assert len(plan) == 14
    assert len(baseline) == 2
    assert len(guided) == 12
    assert {row["row_id"] for row in baseline} == {"a-baseline", "b-baseline"}


def test_summary_keeps_training_seeds_as_model_level_repetitions():
    records = [
        _solver_row("a", "baseline", None, 100),
        _solver_row("b", "baseline", None, 200),
        _solver_row("a", "proposed_async", 0, 90),
        _solver_row("b", "proposed_async", 0, 180),
        _solver_row("a", "conventional_sync", 0, 100),
        _solver_row("b", "conventional_sync", 0, 200),
        _solver_row("a", "proposed_async", 1, 105),
        _solver_row("b", "proposed_async", 1, 210),
        _solver_row("a", "conventional_sync", 1, 100),
        _solver_row("b", "conventional_sync", 1, 200),
    ]

    summary = summarize_multiseed(
        records,
        _training_audit(),
        seeds=(0, 1),
        bootstrap_samples=100,
        bootstrap_seed=7,
    )

    assert summary["num_shared_baseline_rows"] == 2
    assert summary["experimental_unit_contract"]["pooled_seed_instance_inference"] is False
    assert set(summary["by_training_seed"]) == {"0", "1"}
    seed_effect = summary["across_training_seeds_descriptive"][
        "median_relative_soc_delta_proposed_vs_conventional"
    ]
    assert abs(seed_effect["minimum"] + 0.1) < 1e-12
    assert abs(seed_effect["maximum"] - 0.05) < 1e-12
    assert abs(seed_effect["median"] + 0.025) < 1e-12
    assert (
        summary["across_training_seeds_descriptive"]
        ["all_available_seed_medians_favor_proposed"]
        is False
    )


def test_solve_failures_are_compared_before_soc():
    records = [
        _solver_row("a", "baseline", None, 100),
        _solver_row("b", "baseline", None, 200),
        _solver_row("a", "proposed_async", 0, 999, solved=True),
        _solver_row("b", "proposed_async", 0, 0, solved=False),
        _solver_row("a", "conventional_sync", 0, 0, solved=False),
        _solver_row("b", "conventional_sync", 0, 200, solved=True),
    ]

    direct = summarize_multiseed(
        records,
        _training_audit(),
        seeds=(0,),
        bootstrap_samples=0,
    )["by_training_seed"]["0"]["direct_proposed_vs_conventional"]

    assert direct["all_instance_outcomes"] == {
        "left_better": 1,
        "tie": 0,
        "left_worse": 1,
        "both_unsolved": 0,
    }
    assert direct["both_solved"]["instances"] == 0


def test_cached_row_requires_exact_provenance_and_protocol(tmp_path):
    binary = tmp_path / "solver"
    grid_map = tmp_path / "map"
    scenario = tmp_path / "scenario"
    model = tmp_path / "model"
    for path in (binary, grid_map, scenario, model):
        path.write_bytes(path.name.encode())
    spec = LaGATRunSpec(
        row_id="row",
        map_path=grid_map,
        scenario_path=scenario,
        num_agents=64,
        planner_seed=123,
        time_limit_s=30.0,
        model_path=model,
    )
    result_path = tmp_path / "rows" / "row" / "result.json"
    result_path.parent.mkdir(parents=True)
    cached = {
        "binary_sha256": sha256_file(binary),
        "map_sha256": sha256_file(grid_map),
        "scenario_sha256": sha256_file(scenario),
        "model_sha256": sha256_file(model),
        "spec": {
            "row_id": "row",
            "map_path": str(grid_map),
            "scenario_path": str(scenario),
            "num_agents": 64,
            "planner_seed": 123,
            "time_limit_s": 30.0,
            "model_path": str(model),
            "sampling": "deterministic",
            "tau": 0.75,
            "enable_communication_radius": True,
            "communication_radius": 7,
            "lns_refiners": None,
        },
        "metrics": {"solved": True, "soc": 100},
    }
    result_path.write_text(json.dumps(cached), encoding="utf-8")

    assert _load_or_run_row(binary=binary, spec=spec, rows_dir=tmp_path / "rows") == cached

    stale = json.loads(result_path.read_text())
    stale["spec"]["communication_radius"] = 8
    result_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="communication_radius"):
        _load_or_run_row(binary=binary, spec=spec, rows_dir=tmp_path / "rows")
