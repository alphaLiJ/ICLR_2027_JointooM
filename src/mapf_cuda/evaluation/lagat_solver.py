"""Isolated LaCAM/LaGAT process execution and result parsing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def parse_lagat_result(path: Path) -> dict[str, Any]:
    """Parse the key/value header emitted by LaGAT's ``make_log``."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "solution=":
            break
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    required = {
        "agents",
        "solved",
        "soc",
        "soc_lb",
        "makespan",
        "makespan_lb",
        "sum_of_loss",
        "sum_of_loss_lb",
        "comp_time",
        "comp_time_ms_wo_model_load",
        "seed",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"LaGAT result is missing fields {missing}: {path}")

    integers = {
        "agents",
        "soc",
        "soc_lb",
        "makespan",
        "makespan_lb",
        "sum_of_loss",
        "sum_of_loss_lb",
        "seed",
    }
    parsed: dict[str, Any] = {}
    for key, value in values.items():
        if key == "solved":
            parsed[key] = value.strip().lower() in {"1", "true"}
        elif key in integers:
            parsed[key] = int(value)
        elif key in {"comp_time", "comp_time_ms_wo_model_load"}:
            parsed[key] = float(value)
        else:
            parsed[key] = value
    return parsed


@dataclass(frozen=True)
class LaGATRunSpec:
    row_id: str
    map_path: Path
    scenario_path: Path
    num_agents: int
    planner_seed: int
    time_limit_s: float
    model_path: Path | None = None
    sampling: str = "deterministic"
    tau: float = 0.75
    enable_communication_radius: bool = True
    communication_radius: int = 7
    lns_refiners: int | None = None


def build_lagat_command(
    binary: Path, spec: LaGATRunSpec, result_path: Path
) -> list[str]:
    command = [
        str(binary),
        "-m",
        str(spec.map_path),
        "-i",
        str(spec.scenario_path),
        "-N",
        str(spec.num_agents),
        "-s",
        str(spec.planner_seed),
        "-t",
        str(spec.time_limit_s),
        "-o",
        str(result_path),
        "--log_short",
    ]
    if spec.model_path is not None:
        command.extend(
            (
                "--model",
                str(spec.model_path),
                "--sampling",
                spec.sampling,
                "--tau",
                str(spec.tau),
            )
        )
        if spec.enable_communication_radius:
            command.extend(
                ("--enable_communication_radius", "--communication_radius", str(spec.communication_radius))
            )
    if spec.lns_refiners is not None:
        command.extend(("--lns", "--plns_num_refiners", str(spec.lns_refiners)))
    return command


def run_lagat_row(
    binary: Path,
    spec: LaGATRunSpec,
    output_dir: Path,
    *,
    timeout_grace_s: float = 60.0,
) -> dict[str, Any]:
    """Run exactly one solver row in a fresh process lifetime."""

    binary = binary.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    for input_path in (spec.map_path, spec.scenario_path):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    if spec.model_path is not None and not spec.model_path.is_file():
        raise FileNotFoundError(spec.model_path)

    row_dir = output_dir / spec.row_id
    row_dir.mkdir(parents=True, exist_ok=False)
    result_path = row_dir / "solver-result.txt"
    command = build_lagat_command(binary, spec, result_path)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, spec.time_limit_s + timeout_grace_s),
    )
    process_wall_s = time.perf_counter() - started
    (row_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (row_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"LaGAT row {spec.row_id} failed with code {completed.returncode}; "
            f"see {row_dir}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"LaGAT row {spec.row_id} did not create {result_path}")

    parsed = parse_lagat_result(result_path)
    record = {
        "schema_version": 1,
        "spec": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(spec).items()
        },
        "command": command,
        "binary_path": str(binary),
        "binary_sha256": sha256_file(binary),
        "map_sha256": sha256_file(spec.map_path),
        "scenario_sha256": sha256_file(spec.scenario_path),
        "model_sha256": (
            None if spec.model_path is None else sha256_file(spec.model_path)
        ),
        "process_wall_s": process_wall_s,
        "returncode": completed.returncode,
        "metrics": parsed,
    }
    (row_dir / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
