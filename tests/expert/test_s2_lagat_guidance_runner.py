from __future__ import annotations

import json
from pathlib import Path

from experiments.runners.run_s2_lagat_guidance import (
    freeze_inputs,
    latest_checkpoint,
    summarize,
)


def test_freeze_inputs_is_deterministic_and_has_disjoint_tasks(tmp_path):
    maps = tmp_path / "maps.yaml"
    maps.write_text(
        "open: |-\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n"
        "  ....................\n",
        encoding="utf-8",
    )
    first = freeze_inputs(
        maps,
        tmp_path / "first",
        map_names=("open",),
        agent_counts=(8,),
        instances_per_setting=1,
        seed_start=100,
        minimum_manhattan=5,
    )
    second = freeze_inputs(
        maps,
        tmp_path / "second",
        map_names=("open",),
        agent_counts=(8,),
        instances_per_setting=1,
        seed_start=100,
        minimum_manhattan=5,
    )

    assert first["records"][0]["scenario_sha256"] == second["records"][0]["scenario_sha256"]
    scenario = Path(first["records"][0]["scenario_path"])
    rows = [line.split("\t") for line in scenario.read_text(encoding="utf-8").splitlines()[1:]]
    starts = {(row[4], row[5]) for row in rows}
    goals = {(row[6], row[7]) for row in rows}
    assert len(starts) == len(goals) == 8
    assert starts.isdisjoint(goals)
    stored = json.loads((tmp_path / "first" / "manifest.json").read_text(encoding="utf-8"))
    assert stored["records"] == first["records"]


def test_summary_reports_paired_soc_delta_instead_of_unpaired_medians():
    def row(instance, policy, soc, solved=True):
        return {
            "instance_id": instance,
            "num_agents": 64,
            "policy": policy,
            "metrics": {
                "solved": solved,
                "soc": soc,
                "soc_lb": 80,
                "comp_time_ms_wo_model_load": 1,
            },
        }

    result = summarize(
        [
            row("a", "baseline", 100),
            row("b", "baseline", 200),
            row("a", "proposed", 90),
            row("b", "proposed", 220),
        ]
    )

    paired = result["paired_vs_baseline"]["proposed"]
    assert paired["paired_solved_instances"] == 2
    assert paired["wins_lower_soc"] == paired["losses_higher_soc"] == 1
    assert abs(paired["median_relative_soc_delta"]) < 1e-12


def test_latest_checkpoint_falls_back_to_forced_wall_checkpoint(tmp_path):
    row = tmp_path / "proposed_async-s0"
    checkpoints = row / "checkpoints"
    checkpoints.mkdir(parents=True)
    early = checkpoints / "step10.pt"
    final = checkpoints / "step20.pt"
    early.write_bytes(b"early")
    final.write_bytes(b"final")
    (row / "result.json").write_text(
        json.dumps(
            {
                "saved_checkpoints": [
                    {"optimizer_step": 10, "path": str(early)},
                    {"optimizer_step": 20, "path": str(final)},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert latest_checkpoint(tmp_path, "proposed_async", 0) == final.resolve()


def test_standard_mapf_tasks_allow_cross_set_overlap(tmp_path):
    maps = tmp_path / "maps.yaml"
    maps.write_text(
        "open: |-\n" + "\n".join("  ....." for _ in range(5)) + "\n",
        encoding="utf-8",
    )
    result = freeze_inputs(
        maps,
        tmp_path / "standard",
        map_names=("open",),
        agent_counts=(13,),
        instances_per_setting=1,
        seed_start=200,
        minimum_manhattan=1,
        disjoint_start_goal_sets=False,
    )

    scenario = Path(result["records"][0]["scenario_path"])
    rows = [line.split("\t") for line in scenario.read_text().splitlines()[1:]]
    starts = {(row[4], row[5]) for row in rows}
    goals = {(row[6], row[7]) for row in rows}
    assert len(starts) == len(goals) == 13
    assert starts & goals
