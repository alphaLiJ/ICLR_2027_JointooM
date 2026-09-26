"""Render accepted S3 memory runs without importing CUDA or model code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROW_ORDER = ("S3-M1", "S3-G1", "S3-M2B", "S3-M2F", "S3-T1", "S3-M3", "S3-C1")
PERSISTENT_CHECKPOINT = {
    "S3-M1": "persistent_allocated",
    "S3-M2F": "persistent_allocated",
    "S3-M2B": "persistent_allocated",
    "S3-M3": "persistent_allocated",
    "S3-G1": "persistent_allocated",
    "S3-T1": "training_persistent",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs = []
    for result_path in sorted(runs_root.rglob("result.json")):
        result = _read_json(result_path)
        if result.get("status") != "complete":
            continue
        acceptance = result.get("acceptance")
        if acceptance is not None and not acceptance.get("complete", False):
            continue
        run_dir = result_path.parent
        runs.append(
            {
                "run_dir": run_dir,
                "result": result,
                "config": _read_json(run_dir / "config.json"),
                "capacity": _read_json(run_dir / "capacity-model.json"),
            }
        )
    return runs


def memory_component_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        row_id = str(run["result"]["row_id"])
        checkpoint = PERSISTENT_CHECKPOINT.get(row_id)
        if checkpoint is None:
            continue
        by_owner: defaultdict[str, int] = defaultdict(int)
        with (run["run_dir"] / "tensor-inventory.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            for item in csv.DictReader(stream):
                if item["checkpoint"] != checkpoint:
                    continue
                if item["counted_unique_storage"].lower() != "true":
                    continue
                by_owner[item["owner"]] += int(item["storage_bytes"])
        for owner, storage_bytes in sorted(by_owner.items()):
            rows.append(
                {
                    "row_id": row_id,
                    "repetition": int(run["result"]["repetition"]),
                    "checkpoint": checkpoint,
                    "owner": owner,
                    "storage_bytes": storage_bytes,
                }
            )
    return rows


def allocator_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        path = run["run_dir"] / "allocator-checkpoints.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["row_id"] = run["result"]["row_id"]
            item["repetition"] = int(run["result"]["repetition"])
            rows.append(item)
    return rows


def capacity_rows(runs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        estimate = run["capacity"]["estimate"]
        preflight = run["capacity"]["preflight"]
        for component, value in estimate["components"].items():
            rows.append(
                {
                    "row_id": run["result"]["row_id"],
                    "repetition": int(run["result"]["repetition"]),
                    "component": component,
                    "bytes": int(value),
                    "mandatory_known_bytes": int(estimate["mandatory_known_bytes"]),
                    "free_bytes": int(preflight["free_bytes"]),
                    "total_bytes": int(preflight["total_bytes"]),
                    "allowed": bool(preflight["allowed"]),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _median_by_key(
    rows: Iterable[Mapping[str, Any]], *, keys: Sequence[str], value: str
) -> dict[tuple[Any, ...], float]:
    grouped: defaultdict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row[value]))
    return {key: statistics.median(values) for key, values in grouped.items()}


def _plot_persistent(components: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    medians = _median_by_key(
        components, keys=("row_id", "owner"), value="storage_bytes"
    )
    row_ids = [row for row in ROW_ORDER if row in {key[0] for key in medians}]
    owners = sorted({key[1] for key in medians})
    bottoms = [0.0] * len(row_ids)
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for owner in owners:
        values = [medians.get((row_id, owner), 0.0) / 1024**3 for row_id in row_ids]
        axis.bar(row_ids, values, bottom=bottoms, label=owner)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_ylabel("Deduplicated persistent storage (GiB)")
    axis.set_xlabel("S3 row")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _plot_microbatch(allocators: Sequence[Mapping[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in allocators
        if row["row_id"] in {"S3-M2F", "S3-M2B"}
        and row["checkpoint"] == "inference_peak"
    ]
    medians = _median_by_key(
        selected, keys=("row_id",), value="peak_allocated_bytes"
    )
    row_ids = [row for row in ("S3-M2F", "S3-M2B") if (row,) in medians]
    values = [medians[(row,)] / 1024**3 for row in row_ids]
    fig, axis = plt.subplots(figsize=(5.0, 4.2))
    axis.bar(row_ids, values, color=("#4c78a8", "#f58518")[: len(row_ids)])
    axis.set_ylabel("Peak allocated memory (GiB)")
    axis.set_xlabel("Matched full / microbatch row")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def render_results_markdown(
    runs: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    allocators: Sequence[Mapping[str, Any]],
) -> str:
    component_totals: defaultdict[tuple[str, int], int] = defaultdict(int)
    for row in components:
        component_totals[(row["row_id"], int(row["repetition"]))] += int(
            row["storage_bytes"]
        )
    peak_rows = [row for row in allocators if row["checkpoint"] in {"inference_peak", "optimizer_step_peak"}]
    peak_medians = _median_by_key(
        peak_rows, keys=("row_id",), value="peak_allocated_bytes"
    )
    component_medians = _median_by_key(
        components, keys=("row_id", "owner"), value="storage_bytes"
    )
    lines = [
        "# S3 Memory Decomposition Results",
        "",
        "All byte values below come from PyTorch allocator counters or a deduplicated tensor-storage census. NVML was not used. Wall time is diagnostic only.",
        "",
        "| Row | Complete reps | Persistent tensor GiB (median) | Peak allocated GiB (median) |",
        "|---|---:|---:|---:|",
    ]
    for row_id in ROW_ORDER:
        row_runs = [run for run in runs if run["result"]["row_id"] == row_id]
        if not row_runs:
            continue
        persistent_values = [
            value / 1024**3
            for (key_row, _), value in component_totals.items()
            if key_row == row_id
        ]
        persistent = (
            f"{statistics.median(persistent_values):.3f}" if persistent_values else "n/a"
        )
        peak = (
            f"{peak_medians[(row_id,)] / 1024**3:.3f}"
            if (row_id,) in peak_medians
            else "n/a"
        )
        lines.append(f"| {row_id} | {len(row_runs)} | {persistent} | {peak} |")

    lines.extend(
        [
            "",
            "## Persistent ownership breakdown",
            "",
            "Host-pinned transport is listed explicitly; all other rows are CUDA tensor storage.",
            "",
            "| Row | Owner | Median MiB |",
            "|---|---|---:|",
        ]
    )
    for row_id in ROW_ORDER:
        for (component_row, owner), value in sorted(component_medians.items()):
            if component_row == row_id:
                lines.append(f"| {row_id} | {owner} | {value / 1024**2:.3f} |")

    full_peak = peak_medians.get(("S3-M2F",))
    micro_peak = peak_medians.get(("S3-M2B",))
    if full_peak is not None and micro_peak is not None:
        full_persistent = [
            value
            for (row_id, _), value in component_totals.items()
            if row_id == "S3-M2F"
        ]
        micro_persistent = [
            value
            for (row_id, _), value in component_totals.items()
            if row_id == "S3-M2B"
        ]
        lines.extend(
            [
                "",
                "## Matched full-batch versus microbatch inference",
                "",
                f"- Full-batch peak: {full_peak / 1024**3:.3f} GiB.",
                f"- 16-environment microbatch peak: {micro_peak / 1024**3:.3f} GiB.",
                f"- Peak reduction: {(full_peak - micro_peak) / 1024**3:.3f} GiB ({full_peak / micro_peak:.3f}x full/micro ratio).",
                f"- Persistent storage medians: {statistics.median(full_persistent) / 1024**3:.3f} GiB and {statistics.median(micro_persistent) / 1024**3:.3f} GiB; the matched inventories are unchanged.",
            ]
        )

    training_runs = [run for run in runs if run["result"]["row_id"] == "S3-T1"]
    if training_runs:
        operation_rows = [run["result"]["operation"] for run in training_runs]
        ring_bytes = statistics.median(
            float(row["ring_raw_payload_bytes"]) for row in operation_rows
        )
        stage_bytes = statistics.median(
            float(row["gpu_stage_bytes"]) for row in operation_rows
        )
        library_bytes = statistics.median(
            float(row["allocator_reconciliation_after_cleanup"]["library_workspace_floor_bytes"])
            for row in operation_rows
        )
        adjusted_ratio = statistics.median(
            float(row["persistent_reconciliation_adjusted_residual_ratio"])
            for row in operation_rows
        )
        lines.extend(
            [
                "",
                "## Training transport and allocator reconciliation",
                "",
                f"- CPU pinned frontier ring payload (depth 128, E4, A256): {ring_bytes / 1024**2:.3f} MiB.",
                f"- GPU compact staging payload (1024 rows): {stage_bytes / 1024:.3f} KiB.",
                f"- Post-cleanup CUDA library workspace floor: {library_bytes / 1024**2:.3f} MiB.",
                f"- Tensor-versus-owned-allocator residual after naming that floor: {adjusted_ratio * 100:.3f}%.",
            ]
        )

    capacity_runs = [run for run in runs if run["result"]["row_id"] == "S3-C1"]
    if capacity_runs:
        capacity = capacity_runs[0]["capacity"]
        estimate = capacity["estimate"]
        preflight = capacity["preflight"]
        lines.extend(
            [
                "",
                "## A512/E1024 safe capacity preflight",
                "",
                "| Analytical component | GiB |",
                "|---|---:|",
            ]
        )
        for name, value in sorted(estimate["components"].items()):
            lines.append(f"| {name} | {int(value) / 1024**3:.3f} |")
        lines.extend(
            [
                f"| **Mandatory known total** | **{int(estimate['mandatory_known_bytes']) / 1024**3:.3f}** |",
                "",
                f"The runtime exposed {int(preflight['free_bytes']) / 1024**3:.3f} GiB free. With a {int(preflight['safety_reserve_bytes']) / 1024**3:.3f} GiB safety reserve, the predicted request was short by {int(preflight['shortfall_bytes']) / 1024**3:.3f} GiB, so the destructive allocation was not attempted.",
            ]
        )

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "| Row | Code commit(s) | Input SHA-256 | Checkpoint SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for row_id in ROW_ORDER:
        row_runs = [run for run in runs if run["result"]["row_id"] == row_id]
        if not row_runs:
            continue
        commits = sorted({run["result"]["provenance"]["code_commit"] for run in row_runs})
        input_hashes = sorted({run["result"]["provenance"]["input_sha256"] for run in row_runs})
        checkpoint_hashes = sorted(
            {
                run["result"]["provenance"]["checkpoint_sha256"]
                for run in row_runs
                if run["result"]["provenance"]["checkpoint_sha256"] is not None
            }
        )
        lines.append(
            f"| {row_id} | {', '.join(commit[:12] for commit in commits)} | "
            f"{', '.join(value[:12] for value in input_hashes)} | "
            f"{', '.join(value[:12] for value in checkpoint_hashes) if checkpoint_hashes else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "S3 explains memory ownership and capacity. It does not provide a throughput, validation-accuracy, or closed-loop-success comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def render(runs_root: Path, output_dir: Path) -> None:
    runs = discover_runs(runs_root)
    if not runs:
        raise RuntimeError(f"no completed S3 runs below {runs_root}")
    output_dir.mkdir(parents=True, exist_ok=False)
    components = memory_component_rows(runs)
    allocators = allocator_rows(runs)
    capacities = capacity_rows(runs)
    _write_csv(output_dir / "memory_components.csv", components)
    _write_csv(output_dir / "allocator_checkpoints.csv", allocators)
    _write_csv(output_dir / "capacity_model.csv", capacities)
    (output_dir / "RESULTS.md").write_text(
        render_results_markdown(runs, components, allocators), encoding="utf-8"
    )
    _plot_persistent(components, output_dir / "persistent_memory.pdf")
    _plot_microbatch(allocators, output_dir / "microbatch_peak.pdf")
    checksum_lines = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        checksum_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render(args.runs_root.expanduser().resolve(), args.output_dir.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
