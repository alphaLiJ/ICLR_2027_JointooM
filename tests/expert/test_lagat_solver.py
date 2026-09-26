from __future__ import annotations

from pathlib import Path

from mapf_cuda.evaluation.lagat_solver import (
    LaGATRunSpec,
    build_lagat_command,
    parse_lagat_result,
)


RESULT = """agents=64
map_file=test.map
solver=planner
solved=1
soc=812
soc_lb=700
makespan=31
makespan_lb=25
sum_of_loss=748
sum_of_loss_lb=700
comp_time=102.5
comp_time_ms_wo_model_load=88.25
seed=19
solution=
"""


def test_parse_lagat_result_header(tmp_path):
    result = tmp_path / "result.txt"
    result.write_text(RESULT, encoding="utf-8")

    parsed = parse_lagat_result(result)

    assert parsed["solved"] is True
    assert parsed["soc"] == 812
    assert parsed["comp_time_ms_wo_model_load"] == 88.25


def test_learned_lns_command_has_explicit_inference_contract(tmp_path):
    spec = LaGATRunSpec(
        row_id="row",
        map_path=tmp_path / "a.map",
        scenario_path=tmp_path / "a.scen",
        num_agents=128,
        planner_seed=7,
        time_limit_s=3.0,
        model_path=tmp_path / "model.pt",
        lns_refiners=8,
    )

    command = build_lagat_command(Path("/solver/main"), spec, tmp_path / "out")

    assert command[0] == "/solver/main"
    assert command[command.index("--model") + 1].endswith("model.pt")
    assert "--enable_communication_radius" in command
    assert command[command.index("--communication_radius") + 1] == "7"
    assert command[command.index("--plns_num_refiners") + 1] == "8"
