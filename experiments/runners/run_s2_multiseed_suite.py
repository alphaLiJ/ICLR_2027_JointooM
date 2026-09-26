"""Orchestrate preflight and six fresh-process S2 matched-wall rows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from experiments.runners.run_s2_multiseed_magat_training import sha256_file


SEEDS = (0, 1, 2)
FORMAL_ORDER = (
    (0, "proposed_async"),
    (0, "conventional_sync"),
    (1, "conventional_sync"),
    (1, "proposed_async"),
    (2, "proposed_async"),
    (2, "conventional_sync"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_busy_fraction(sample_s: float = 2.0) -> float:
    total_before, idle_before = _cpu_counters()
    time.sleep(sample_s)
    total_after, idle_after = _cpu_counters()
    delta_total = max(1, total_after - total_before)
    return 1.0 - (idle_after - idle_before) / delta_total


def _cuda_free_fraction() -> float:
    command = [
        sys.executable,
        "-c",
        (
            "import torch; "
            "free,total=torch.cuda.mem_get_info(0); "
            "print(f'{free / total:.12f}')"
        ),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=True
    )
    return float(completed.stdout.strip().splitlines()[-1])


def _wait_for_formal_resources(
    root: Path,
    *,
    max_cpu_busy_fraction: float = 0.60,
    max_load_per_cpu: float = 0.25,
    min_cuda_free_fraction: float = 0.70,
    consecutive_passes: int = 3,
) -> dict[str, Any]:
    """Wait for a quiet, stable launch window before each timed formal row."""

    passes = 0
    observations = []
    while passes < consecutive_passes:
        busy = _cpu_busy_fraction()
        load_1m = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        cuda_free = _cuda_free_fraction()
        passed = (
            busy <= max_cpu_busy_fraction
            and load_1m / cpu_count <= max_load_per_cpu
            and cuda_free >= min_cuda_free_fraction
        )
        passes = passes + 1 if passed else 0
        observation = {
            "timestamp": time.time(),
            "cpu_busy_fraction": busy,
            "load_1m": load_1m,
            "cpu_count": cpu_count,
            "load_per_cpu": load_1m / cpu_count,
            "cuda_free_fraction": cuda_free,
            "passed": passed,
            "consecutive_passes": passes,
        }
        observations.append(observation)
        _write_json(
            root / "resource-gate.json",
            {
                "status": "passed" if passes >= consecutive_passes else "waiting",
                "thresholds": {
                    "max_cpu_busy_fraction": max_cpu_busy_fraction,
                    "max_load_per_cpu": max_load_per_cpu,
                    "min_cuda_free_fraction": min_cuda_free_fraction,
                    "consecutive_passes": consecutive_passes,
                },
                "latest": observation,
                "observations": observations[-60:],
            },
        )
        if passes < consecutive_passes:
            time.sleep(8.0)
    return observations[-1]


def _run_row(
    *,
    root: Path,
    phase: str,
    seed: int,
    method: str,
    validation: Sequence[Path],
    wall_budget_s: float | None,
    num_steps: int,
    checkpoint_interval_s: float,
) -> dict[str, Any]:
    row_dir = root / phase / f"seed-{seed}" / method
    if (row_dir / "result.json").is_file():
        result = json.loads((row_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("status") == "ok":
            return result
        raise RuntimeError(f"existing row is not successful: {row_dir}")
    if row_dir.exists():
        raise RuntimeError(f"incomplete row directory requires audit: {row_dir}")
    row_dir.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "experiments.runners.run_s2_multiseed_magat_training",
        "--method",
        method,
        "--seed",
        str(seed),
        "--output-dir",
        str(row_dir),
        "--num-steps",
        str(num_steps),
        "--checkpoint-interval-s",
        str(checkpoint_interval_s),
    ]
    if validation:
        command.extend(
            ["--validation-trajectories", ",".join(str(path) for path in validation)]
        )
    if wall_budget_s is not None:
        command.extend(["--wall-budget-s", str(wall_budget_s)])
    if phase == "preflight":
        command.append("--capture-stage-hashes")

    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    with (row_dir.parent / f"{method}.console.log").open(
        "w", encoding="utf-8"
    ) as log_handle:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"row failed with exit code {completed.returncode}: {row_dir}"
        )
    result = json.loads((row_dir / "result.json").read_text(encoding="utf-8"))
    result["orchestrator_elapsed_s"] = time.time() - started
    return result


def _audit_preflight(root: Path, results: dict[tuple[int, str], dict]) -> dict:
    rows = []
    valid = True
    for seed in SEEDS:
        proposed = results[(seed, "proposed_async")]
        conventional = results[(seed, "conventional_sync")]
        model_match = (
            proposed.get("initial_model_state_sha256")
            == conventional.get("initial_model_state_sha256")
        )
        trajectory_match = proposed.get("stage_sha256s") == conventional.get(
            "stage_sha256s"
        )
        row = {
            "seed": seed,
            "initial_model_state_match": model_match,
            "expert_trajectory_match": trajectory_match,
            "proposed_initial_model_state_sha256": proposed.get(
                "initial_model_state_sha256"
            ),
            "conventional_initial_model_state_sha256": conventional.get(
                "initial_model_state_sha256"
            ),
            "num_compared_stages": len(proposed.get("stage_sha256s", [])),
        }
        rows.append(row)
        valid = valid and model_match and trajectory_match
    audit = {"valid": valid, "rows": rows}
    _write_json(root / "preflight-audit.json", audit)
    if not valid:
        raise RuntimeError("preflight parity audit failed")
    return audit


def run_suite(
    *,
    output_root: Path,
    validation_sources: Sequence[Path],
    wall_budget_s: float,
    checkpoint_interval_s: float,
    preflight_steps: int,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and not (output_root / "suite-status.json").is_file():
        raise FileExistsError(f"unmanaged output root exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    frozen_validation_dir = output_root / "inputs" / "validation"
    frozen_validation_dir.mkdir(parents=True, exist_ok=True)
    frozen_validation = []
    for source in validation_sources:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = frozen_validation_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"validation copy hash mismatch: {target}")
        frozen_validation.append(target)

    suite_status = {
        "schema_version": 1,
        "status": "running",
        "seeds": list(SEEDS),
        "wall_budget_s": float(wall_budget_s),
        "checkpoint_interval_s": float(checkpoint_interval_s),
        "preflight_steps": int(preflight_steps),
        "formal_order": [list(row) for row in FORMAL_ORDER],
        "validation": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in frozen_validation
        ],
        "completed_rows": [],
    }
    _write_json(output_root / "suite-status.json", suite_status)

    preflight_results = {}
    for seed in SEEDS:
        for method in ("proposed_async", "conventional_sync"):
            result = _run_row(
                root=output_root,
                phase="preflight",
                seed=seed,
                method=method,
                validation=(),
                wall_budget_s=None,
                num_steps=preflight_steps,
                checkpoint_interval_s=checkpoint_interval_s,
            )
            preflight_results[(seed, method)] = result
    suite_status["preflight_audit"] = _audit_preflight(
        output_root, preflight_results
    )
    _write_json(output_root / "suite-status.json", suite_status)

    formal_results = []
    for seed, method in FORMAL_ORDER:
        suite_status["resource_gate"] = _wait_for_formal_resources(output_root)
        _write_json(output_root / "suite-status.json", suite_status)
        result = _run_row(
            root=output_root,
            phase="formal",
            seed=seed,
            method=method,
            validation=frozen_validation,
            wall_budget_s=wall_budget_s,
            num_steps=100_000,
            checkpoint_interval_s=checkpoint_interval_s,
        )
        formal_results.append(result)
        suite_status["completed_rows"].append(
            {
                "seed": seed,
                "method": method,
                "num_optimizer_steps": result["num_optimizer_steps"],
                "samples_processed": result["samples_processed"],
                "total_wall_s": result["total_wall_s"],
                "samples_s": result["samples_s"],
                "output_dir": result["output_dir"],
            }
        )
        _write_json(output_root / "suite-status.json", suite_status)

    suite_status["status"] = "training_complete"
    _write_json(output_root / "suite-status.json", suite_status)
    return suite_status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validation-sources", required=True)
    parser.add_argument("--wall-budget-s", type=float, default=3035.921933)
    parser.add_argument("--checkpoint-interval-s", type=float, default=300.0)
    parser.add_argument("--preflight-steps", type=int, default=20)
    args = parser.parse_args(argv)
    status = run_suite(
        output_root=args.output_root,
        validation_sources=[
            Path(value) for value in args.validation_sources.split(",") if value
        ],
        wall_budget_s=args.wall_budget_s,
        checkpoint_interval_s=args.checkpoint_interval_s,
        preflight_steps=args.preflight_steps,
    )
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
