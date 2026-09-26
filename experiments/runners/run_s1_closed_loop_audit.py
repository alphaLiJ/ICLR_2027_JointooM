"""Run isolated S1 retained-checkpoint closed-loop audit rows and pilot suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CHECKPOINT_ROOT = Path("artifacts/magat_checkpoints")
DEFAULT_INPUT_ROOT = Path("artifacts/frozen_inputs/pilot")
DEFAULT_FORMAL_INPUT_ROOT = Path("artifacts/frozen_inputs/formal")
MODES = ("strong_online", "compact_sync", "proposed_async")
SEEDS = (0, 1, 2)
PILOT_AGENT_COUNTS = (64, 128)
PILOT_TOPOLOGIES = ("random", "maze", "warehouse")
FORMAL_AGENT_COUNTS = (32, 64, 128)
FORMAL_INSTANCE_COUNT = 128
FORMAL_INSTANCE_SEED_START = 810_000
EXPECTED_OPTIMIZER_STEP = 10_000
EXPECTED_SAMPLES = 10_240_000


@dataclass(frozen=True)
class PilotRow:
    row_id: str
    mode: str
    seed: int
    topology: str
    num_agents: int
    input_path: Path
    checkpoint_path: Path


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_clean(repository: Path) -> bool:
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return not output.strip()


def checkpoint_path(root: Path, mode: str, seed: int) -> Path:
    if mode not in MODES or seed not in SEEDS:
        raise ValueError(f"unsupported matched checkpoint: mode={mode}, seed={seed}")
    return root.expanduser().resolve() / f"{mode}-s{seed}" / "checkpoints" / "ckpt_latest.pt"


def audit_checkpoint(path: Path, *, expected_mode: str, expected_seed: int) -> dict[str, Any]:
    import torch

    resolved = path.expanduser().resolve()
    result_path = resolved.parent.parent / "result.json"
    if not resolved.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"checkpoint or training result is missing for {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    training = json.loads(result_path.read_text(encoding="utf-8"))
    optimizer_step = int(checkpoint.get("optimizer_step", -1))
    if optimizer_step != EXPECTED_OPTIMIZER_STEP:
        raise RuntimeError(
            f"latest checkpoint is not at {EXPECTED_OPTIMIZER_STEP}: {resolved}"
        )
    if training.get("mode") != expected_mode or int(training.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"training provenance mismatch for {resolved}")
    if int(training.get("samples_processed", -1)) != EXPECTED_SAMPLES:
        raise RuntimeError(f"sample budget mismatch for {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
        "mode": expected_mode,
        "seed": expected_seed,
        "optimizer_step": optimizer_step,
        "samples_processed": int(training["samples_processed"]),
        "selection_mode": checkpoint.get("selection_mode"),
        "validation_overall_accuracy": checkpoint.get("val_overall_accuracy"),
        "validation_nonstay_accuracy": checkpoint.get("val_nonstay_accuracy"),
        "training_result_path": str(result_path.resolve()),
        "training_result_sha256": sha256_file(result_path),
    }


def audit_matched_checkpoints(root: Path) -> list[dict[str, Any]]:
    return [
        audit_checkpoint(
            checkpoint_path(root, mode, seed),
            expected_mode=mode,
            expected_seed=seed,
        )
        for mode in MODES
        for seed in SEEDS
    ]


def pilot_rows(
    *, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT, input_root: Path = DEFAULT_INPUT_ROOT
) -> list[PilotRow]:
    rows = []
    for num_agents in PILOT_AGENT_COUNTS:
        for topology in PILOT_TOPOLOGIES:
            input_path = input_root / f"a1-stress-{topology}-a{num_agents}.npz"
            for mode in MODES:
                seed = 0
                rows.append(
                    PilotRow(
                        row_id=f"{mode}-s{seed}-{topology}-a{num_agents}",
                        mode=mode,
                        seed=seed,
                        topology=topology,
                        num_agents=num_agents,
                        input_path=input_path.expanduser().resolve(),
                        checkpoint_path=checkpoint_path(checkpoint_root, mode, seed),
                    )
                )
    return rows


def formal_rows(
    *,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    input_root: Path = DEFAULT_FORMAL_INPUT_ROOT,
) -> list[PilotRow]:
    rows = []
    for num_agents in FORMAL_AGENT_COUNTS:
        for topology in PILOT_TOPOLOGIES:
            input_path = input_root / f"s1-v1-{topology}-a{num_agents}.npz"
            for seed in SEEDS:
                for mode in MODES:
                    rows.append(
                        PilotRow(
                            row_id=f"{mode}-s{seed}-{topology}-a{num_agents}",
                            mode=mode,
                            seed=seed,
                            topology=topology,
                            num_agents=num_agents,
                            input_path=input_path.expanduser().resolve(),
                            checkpoint_path=checkpoint_path(
                                checkpoint_root, mode, seed
                            ),
                        )
                    )
    return rows


def prepare_formal_inputs(output_root: Path) -> dict[str, Any]:
    from expert.benchmark_contract import save_frozen_transition_batch
    from mapf_cuda.experiments.frozen_inputs import build_frozen_standard_mapf_pool

    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"S1 formal input root already exists: {output_root}")
    output_root.mkdir(parents=True)
    seeds = tuple(
        range(FORMAL_INSTANCE_SEED_START, FORMAL_INSTANCE_SEED_START + FORMAL_INSTANCE_COUNT)
    )
    entries = []
    for num_agents in FORMAL_AGENT_COUNTS:
        for topology in PILOT_TOPOLOGIES:
            identity = f"s1-closed-loop-v1-{topology}-a{num_agents}"
            batch = build_frozen_standard_mapf_pool(
                topology=topology,
                density=0.2,
                num_agents=num_agents,
                seeds=seeds,
                horizon=256,
                map_size=128,
                pool_identity=identity,
            )
            path = output_root / f"s1-v1-{topology}-a{num_agents}.npz"
            save_frozen_transition_batch(path, batch)
            entries.append(
                {
                    "path": str(path),
                    "file_sha256": sha256_file(path),
                    "semantic_sha256": batch.semantic_sha256,
                    "pool_identity": identity,
                    "topology": topology,
                    "density": 0.2,
                    "num_agents": num_agents,
                    "num_envs": batch.num_envs,
                    "horizon": batch.horizon,
                    "map_size": [int(batch.grids.shape[1]), int(batch.grids.shape[2])],
                    "instance_seed_start": FORMAL_INSTANCE_SEED_START,
                    "instance_seed_stop_exclusive": FORMAL_INSTANCE_SEED_START
                    + FORMAL_INSTANCE_COUNT,
                }
            )
    manifest = {
        "schema_version": 1,
        "suite": "S1-closed-loop-v1",
        "task_semantics": "standard_mapf",
        "action_selection": "argmax",
        "entries": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_row(
    *,
    input_path: Path,
    checkpoint: Path,
    mode: str,
    seed: int,
    topology: str,
    horizon: int,
    eval_batch_envs: int,
    device: str,
    repository: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    from expert.a4_runtime import _make_runtime
    from expert.benchmark_contract import load_frozen_transition_batch
    from mapf_cuda.evaluation.closed_loop import run_magat_closed_loop_microbatched

    batch = load_frozen_transition_batch(input_path)
    if horizon > batch.horizon:
        raise ValueError("requested S1 horizon exceeds frozen input horizon")
    checkpoint_info = audit_checkpoint(
        checkpoint, expected_mode=mode, expected_seed=seed
    )
    runtime = _make_runtime(str(checkpoint.expanduser().resolve()), device)
    summary, details = run_magat_closed_loop_microbatched(
        runtime,
        batch,
        horizon=horizon,
        device=device,
        max_envs_per_batch=min(int(eval_batch_envs), batch.num_envs),
    )
    properties = torch.cuda.get_device_properties(torch.device(device))
    report = {
        "schema_version": 1,
        "status": "ok",
        "experiment": "S1-retained-checkpoint-closed-loop-audit",
        "task_semantics": "standard_mapf",
        "action_selection": "argmax",
        "topology": topology,
        "mode": mode,
        "seed": int(seed),
        "input": {
            "path": str(input_path.expanduser().resolve()),
            "file_sha256": sha256_file(input_path.expanduser().resolve()),
            "semantic_sha256": batch.semantic_sha256,
            "num_envs": batch.num_envs,
            "num_agents": batch.num_agents,
            "frozen_horizon": batch.horizon,
        },
        "checkpoint": checkpoint_info,
        "code_commit": _git_commit(repository),
        "device": {
            "requested": device,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        },
        "timing_scope": {
            "publication_metric": False,
            "included": ["derived-state", "graph-builder", "model-forward", "transition"],
            "excluded": ["checkpoint-load", "input-load", "simulator-allocation", "report-write"],
            "note": "quality audit only; per-step completion check synchronizes the host",
        },
        **summary,
    }
    return report, {key: np.asarray(value) for key, value in details.items()}


def _write_row(output_dir: Path, report: dict[str, Any], details: dict[str, Any]) -> None:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "details.npz", **details)
    concise = {
        key: report.get(key)
        for key in (
            "status",
            "mode",
            "seed",
            "topology",
            "num_envs",
            "num_agents",
            "initial_individual_success_rate",
            "individual_success_rate",
            "normalized_arrival_gain",
            "arrival_auc",
            "incremental_arrival_auc",
            "complete_success_rate",
        )
    }
    (output_dir / "stdout.log").write_text(
        json.dumps(concise, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_failure(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "stdout.log").write_text(
        json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
    )


def render_suite(suite_root: Path) -> dict[str, Any]:
    config = json.loads((suite_root / "suite-config.json").read_text(encoding="utf-8"))
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((suite_root / "runs").glob("*/result.json"))
    ]
    if not results:
        raise ValueError(f"no S1 results under {suite_root}")
    failures = [row for row in results if row.get("status") != "ok"]
    successful = [row for row in results if row.get("status") == "ok"]
    gains = [float(row["normalized_arrival_gain"]) for row in successful]
    metric_tuples = {
        (
            round(float(row["normalized_arrival_gain"]), 12),
            round(float(row["incremental_arrival_auc"]), 12),
        )
        for row in successful
    }
    gate = {
        "all_rows_succeeded": not failures,
        "row_count_matches_config": len(results) == int(config["expected_rows"]),
        "learned_progress_above_initial_state": bool(gains and max(gains) > 0.0),
        "closed_loop_metrics_not_constant": len(metric_tuples) > 1,
    }
    gate["passed"] = all(gate.values())
    aggregate = {
        "schema_version": 1,
        "experiment": config["suite"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(results),
        "successful_row_count": len(successful),
        "gate": gate,
        "suite_config": config,
        "results": results,
    }
    report_root = suite_root / "report"
    report_root.mkdir(parents=True, exist_ok=False)
    columns = (
        "mode",
        "seed",
        "topology",
        "num_agents",
        "num_envs",
        "initial_individual_success_rate",
        "individual_success_rate",
        "normalized_arrival_gain",
        "arrival_auc",
        "incremental_arrival_auc",
        "complete_success_rate",
        "newly_arrived_step_median",
        "unresolved_agent_fraction",
        "blocked_requested_move_rate",
        "no_arrival_in_final_16_rate",
        "no_motion_in_final_16_rate",
    )
    with (report_root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in successful:
            writer.writerow({key: row.get(key) for key in columns})

    grouped = []
    group_keys = sorted(
        {(row["mode"], int(row["num_agents"]), row["topology"]) for row in successful}
    )
    for mode, num_agents, topology in group_keys:
        rows = [
            row
            for row in successful
            if row["mode"] == mode
            and int(row["num_agents"]) == num_agents
            and row["topology"] == topology
        ]
        entry: dict[str, Any] = {
            "mode": mode,
            "num_agents": num_agents,
            "topology": topology,
            "seed_count": len(rows),
            "seeds": sorted(int(row["seed"]) for row in rows),
        }
        for metric in (
            "individual_success_rate",
            "normalized_arrival_gain",
            "incremental_arrival_auc",
            "complete_success_rate",
            "unresolved_agent_fraction",
            "blocked_requested_move_rate",
            "no_arrival_in_final_16_rate",
            "no_motion_in_final_16_rate",
        ):
            values = [float(row[metric]) for row in rows]
            entry[f"{metric}_mean"] = statistics.mean(values)
            entry[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            entry[f"{metric}_per_seed"] = values
        grouped.append(entry)
    aggregate["grouped_results"] = grouped
    (report_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    grouped_columns = (
        "mode",
        "num_agents",
        "topology",
        "seed_count",
        "individual_success_rate_mean",
        "individual_success_rate_sd",
        "normalized_arrival_gain_mean",
        "normalized_arrival_gain_sd",
        "incremental_arrival_auc_mean",
        "incremental_arrival_auc_sd",
        "complete_success_rate_mean",
        "complete_success_rate_sd",
    )
    with (report_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=grouped_columns)
        writer.writeheader()
        for row in grouped:
            writer.writerow({key: row.get(key) for key in grouped_columns})

    lines = [
        f"# {config['suite']} retained-checkpoint closed-loop audit",
        "",
        f"Gate: **{'PASS' if gate['passed'] else 'FAIL'}**. "
        f"All rows: {gate['all_rows_succeeded']}; progress above initial state: "
        f"{gate['learned_progress_above_initial_state']}; metrics vary: "
        f"{gate['closed_loop_metrics_not_constant']}; row count: "
        f"{len(results)}/{config['expected_rows']}.",
        "",
    ]
    if config["suite"] == "S1-formal":
        lines.extend(
            [
                "| Mode | A | Topology | Final ISR (mean±sd) | Arrival gain (mean±sd) | Incremental AUC (mean±sd) | CSR (mean±sd) |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in grouped:
            lines.append(
                f"| {row['mode']} | {row['num_agents']} | {row['topology']} | "
                f"{row['individual_success_rate_mean']:.4f}±{row['individual_success_rate_sd']:.4f} | "
                f"{row['normalized_arrival_gain_mean']:.4f}±{row['normalized_arrival_gain_sd']:.4f} | "
                f"{row['incremental_arrival_auc_mean']:.4f}±{row['incremental_arrival_auc_sd']:.4f} | "
                f"{row['complete_success_rate_mean']:.4f}±{row['complete_success_rate_sd']:.4f} |"
            )
    else:
        lines.extend(
            [
                "| Mode | A | Topology | Initial ISR | Final ISR | Arrival gain | Incremental AUC | CSR | Median new-arrival step |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in successful:
            median = row.get("newly_arrived_step_median")
            lines.append(
                f"| {row['mode']} | {row['num_agents']} | {row['topology']} | "
                f"{row['initial_individual_success_rate']:.4f} | "
                f"{row['individual_success_rate']:.4f} | "
                f"{row['normalized_arrival_gain']:.4f} | "
                f"{row['incremental_arrival_auc']:.4f} | "
                f"{row['complete_success_rate']:.4f} | "
                f"{'--' if median is None else f'{median:.1f}'} |"
            )
    lines.extend(
        [
            "",
            "Arrival gain and incremental AUC exclude agents already on target at step 0. "
            "The final-16-step metrics are operational stall indicators, not proofs of deadlock.",
            "",
            "Loop wall times in raw rows are diagnostics only and are not publication speed metrics.",
        ]
    )
    (report_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _run_suite(
    *,
    suite_name: str,
    rows: list[PilotRow],
    output_root: Path,
    checkpoint_root: Path,
    repository: Path,
    horizon: int,
    eval_batch_envs: int,
    device: str,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"S1 suite output already exists: {output_root}")
    if not _git_is_clean(repository):
        raise RuntimeError("S1 suite requires a clean committed implementation")
    output_root.mkdir(parents=True)
    audit = audit_matched_checkpoints(checkpoint_root)
    (output_root / "checkpoint-audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "schema_version": 1,
        "suite": suite_name,
        "code_commit": _git_commit(repository),
        "horizon": int(horizon),
        "eval_batch_envs": int(eval_batch_envs),
        "expected_rows": len(rows),
        "device": device,
        "rows": [{**asdict(row), "input_path": str(row.input_path), "checkpoint_path": str(row.checkpoint_path)} for row in rows],
    }
    (output_root / "suite-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    runs_root = output_root / "runs"
    runs_root.mkdir()
    for index, row in enumerate(rows, start=1):
        command = [
            sys.executable,
            "-m",
            "experiments.runners.run_s1_closed_loop_audit",
            "row",
            "--input",
            str(row.input_path),
            "--checkpoint",
            str(row.checkpoint_path),
            "--mode",
            row.mode,
            "--seed",
            str(row.seed),
            "--topology",
            row.topology,
            "--horizon",
            str(horizon),
            "--eval-batch-envs",
            str(eval_batch_envs),
            "--device",
            device,
            "--repository",
            str(repository),
            "--output-dir",
            str(runs_root / row.row_id),
        ]
        print(f"[S1 {index}/{len(rows)}] {row.row_id}", flush=True)
        environment = os.environ.copy()
        source_root = str((repository / "src").resolve())
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        completed = subprocess.run(
            command,
            cwd=repository,
            env=environment,
            text=True,
            capture_output=True,
        )
        (runs_root / f"{row.row_id}.launch.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (runs_root / f"{row.row_id}.launch.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"S1 row failed ({row.row_id}, exit={completed.returncode}); "
                f"see {runs_root / f'{row.row_id}.launch.stderr.log'}"
            )
    return render_suite(output_root)


def run_pilot_suite(
    *,
    output_root: Path,
    checkpoint_root: Path,
    input_root: Path,
    repository: Path,
    horizon: int,
    eval_batch_envs: int,
    device: str,
) -> dict[str, Any]:
    return _run_suite(
        suite_name="S1-pilot",
        rows=pilot_rows(checkpoint_root=checkpoint_root, input_root=input_root),
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        repository=repository,
        horizon=horizon,
        eval_batch_envs=eval_batch_envs,
        device=device,
    )


def run_formal_suite(
    *,
    output_root: Path,
    checkpoint_root: Path,
    input_root: Path,
    repository: Path,
    horizon: int,
    eval_batch_envs: int,
    device: str,
) -> dict[str, Any]:
    manifest_path = input_root.expanduser().resolve() / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing S1 formal input manifest: {manifest_path}")
    rows = formal_rows(checkpoint_root=checkpoint_root, input_root=input_root)
    missing = [str(row.input_path) for row in rows if not row.input_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S1 formal inputs: {sorted(set(missing))}")
    return _run_suite(
        suite_name="S1-formal",
        rows=rows,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        repository=repository,
        horizon=horizon,
        eval_batch_envs=eval_batch_envs,
        device=device,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    row = subparsers.add_parser("row", help="run one isolated S1 row")
    row.add_argument("--input", required=True, type=Path)
    row.add_argument("--checkpoint", required=True, type=Path)
    row.add_argument("--mode", required=True, choices=MODES)
    row.add_argument("--seed", required=True, type=int, choices=SEEDS)
    row.add_argument("--topology", required=True, choices=PILOT_TOPOLOGIES)
    row.add_argument("--horizon", type=int, default=256)
    row.add_argument("--eval-batch-envs", type=int, default=16)
    row.add_argument("--device", default="cuda:0")
    row.add_argument("--repository", type=Path, default=Path.cwd())
    row.add_argument("--output-dir", required=True, type=Path)

    pilot = subparsers.add_parser("pilot", help="audit checkpoints and run the pilot")
    pilot.add_argument("--output-root", required=True, type=Path)
    pilot.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    pilot.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    pilot.add_argument("--repository", type=Path, default=Path.cwd())
    pilot.add_argument("--horizon", type=int, default=256)
    pilot.add_argument("--eval-batch-envs", type=int, default=16)
    pilot.add_argument("--device", default="cuda:0")

    prepare = subparsers.add_parser(
        "prepare-formal", help="freeze the 3 x 3 x 128 formal input suite"
    )
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_FORMAL_INPUT_ROOT)

    formal = subparsers.add_parser("formal", help="run the 81-row matched S1 suite")
    formal.add_argument("--output-root", required=True, type=Path)
    formal.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    formal.add_argument("--input-root", type=Path, default=DEFAULT_FORMAL_INPUT_ROOT)
    formal.add_argument("--repository", type=Path, default=Path.cwd())
    formal.add_argument("--horizon", type=int, default=256)
    formal.add_argument("--eval-batch-envs", type=int, default=16)
    formal.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "row":
        try:
            report, details = run_row(
                input_path=args.input,
                checkpoint=args.checkpoint,
                mode=args.mode,
                seed=args.seed,
                topology=args.topology,
                horizon=args.horizon,
                eval_batch_envs=args.eval_batch_envs,
                device=args.device,
                repository=args.repository.expanduser().resolve(),
            )
            _write_row(args.output_dir, report, details)
            print((args.output_dir / "stdout.log").read_text(encoding="utf-8").strip())
            return 0
        except Exception as error:
            payload = {
                "schema_version": 1,
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "mode": args.mode,
                "seed": args.seed,
                "topology": args.topology,
            }
            _write_failure(args.output_dir, payload)
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
            return 2
    if args.command == "prepare-formal":
        manifest = prepare_formal_inputs(args.output_root)
        print(json.dumps({"entries": len(manifest["entries"]), "status": "ok"}))
        return 0
    suite_runner = run_pilot_suite if args.command == "pilot" else run_formal_suite
    aggregate = suite_runner(
            output_root=args.output_root,
            checkpoint_root=args.checkpoint_root,
            input_root=args.input_root,
            repository=args.repository.expanduser().resolve(),
            horizon=args.horizon,
            eval_batch_envs=args.eval_batch_envs,
            device=args.device,
        )
    print(json.dumps(aggregate["gate"], sort_keys=True))
    return 0 if aggregate["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODES",
    "PILOT_AGENT_COUNTS",
    "PILOT_TOPOLOGIES",
    "SEEDS",
    "audit_checkpoint",
    "audit_matched_checkpoints",
    "formal_rows",
    "pilot_rows",
    "prepare_formal_inputs",
    "render_suite",
    "run_formal_suite",
    "run_row",
]
