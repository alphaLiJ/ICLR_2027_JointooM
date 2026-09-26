"""Aggregate the lightweight E5 MAGAT matched-training experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MODE_ORDER = ("strong_online", "compact_sync", "proposed_async")
MODE_LABELS = {
    "strong_online": "Strong online",
    "compact_sync": "Compact synchronous",
    "proposed_async": "Proposed asynchronous",
}
MODE_COLORS = {
    "strong_online": "#dc2626",
    "compact_sync": "#f59e0b",
    "proposed_async": "#2563eb",
}
THRESHOLDS = (0.75, 0.80, 0.85)


def _median(rows: Iterable[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("experiment") == "E5-MAGAT"
            and result.get("mode") in MODE_ORDER
            and result.get("status") == "ok"
        ):
            result["_source"] = str(path.parent)
            rows.append(result)
    return rows


def _first_threshold(
    checkpoints: Sequence[dict[str, Any]], threshold: float
) -> tuple[int | None, float | None]:
    for checkpoint in checkpoints:
        if float(checkpoint["val_overall_accuracy"]) >= threshold:
            return (
                int(checkpoint["optimizer_step"]),
                float(checkpoint["checkpoint_elapsed_s"]),
            )
    return None, None


def _curve_svg(
    validation_rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title: str,
) -> str:
    width, height = 900, 500
    left, right, top, bottom = 85, 30, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in validation_rows:
        grouped[(str(row["mode"]), int(row[x_key]))].append(
            float(row["val_overall_accuracy"])
        )
    x_values = sorted({key[1] for key in grouped})
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = 0.5, 0.9

    def xpos(value: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def ypos(value: float) -> float:
        bounded = min(y_max, max(y_min, value))
        return top + (y_max - bounded) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.tick{font-size:13px}.axis{font-size:15px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}.line{fill:none;stroke-width:3}</style>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">{title}</text>',
    ]
    for y_value in (0.5, 0.6, 0.7, 0.8, 0.9):
        y = ypos(y_value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left-10}" y="{y+5}" text-anchor="end">{y_value:.1f}</text>'
        )
    for mode in MODE_ORDER:
        points = []
        for x_value in x_values:
            values = grouped.get((mode, x_value))
            if values:
                points.append(
                    f"{xpos(x_value):.2f},{ypos(statistics.median(values)):.2f}"
                )
        if points:
            parts.append(
                f'<polyline class="line" stroke="{MODE_COLORS[mode]}" points="{" ".join(points)}"/>'
            )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = x_min + fraction * (x_max - x_min)
        x = xpos(value)
        parts.append(
            f'<text class="tick" x="{x}" y="{height-bottom+25}" text-anchor="middle">{value:g}</text>'
        )
    legend_x = 125
    for index, mode in enumerate(MODE_ORDER):
        x = legend_x + index * 235
        parts.append(
            f'<line x1="{x}" y1="{height-27}" x2="{x+25}" y2="{height-27}" '
            f'stroke="{MODE_COLORS[mode]}" stroke-width="4"/>'
        )
        parts.append(
            f'<text class="tick" x="{x+33}" y="{height-22}">{MODE_LABELS[mode]}</text>'
        )
    parts.append(
        f'<text class="axis" x="{left+plot_w/2}" y="{height-42}" text-anchor="middle">{x_label}</text>'
    )
    parts.append(
        f'<text class="axis" transform="translate(24 {top+plot_h/2}) rotate(-90)" '
        'text-anchor="middle">Validation accuracy</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(root: Path, output_dir: Path, *, allow_partial: bool = False) -> None:
    results = _load_results(root)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[str(result["mode"])].append(result)
    missing = [mode for mode in MODE_ORDER if len(groups[mode]) != 3]
    if missing and not allow_partial:
        counts = {mode: len(groups[mode]) for mode in MODE_ORDER}
        raise ValueError(f"E5-MAGAT requires three seeds per mode, got {counts}")
    if not results:
        raise ValueError(f"no E5-MAGAT results found under {root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for result in results:
        checkpoints = list(result["checkpoint_history"])
        best = max(checkpoints, key=lambda row: float(row["val_overall_accuracy"]))
        final = checkpoints[-1]
        phases = result["phase_metrics"]
        row = {
            "mode": result["mode"],
            "seed": result["seed"],
            "optimizer_steps": result["num_optimizer_steps"],
            "samples_processed": result["samples_processed"],
            "total_wall_s": result["total_wall_s"],
            "samples_s": result["samples_s"],
            "optimizer_steps_s": result["optimizer_steps_s"],
            "mean_training_loss": result["mean_loss"],
            "final_training_loss": result["final_loss"],
            "best_validation_accuracy": best["val_overall_accuracy"],
            "best_validation_step": best["optimizer_step"],
            "final_validation_accuracy": final["val_overall_accuracy"],
            "h2d_bytes": result["h2d_bytes"],
            "d2h_bytes": result["d2h_bytes"],
            "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
            "peak_nvml_gpu_memory_bytes": result.get(
                "peak_nvml_gpu_memory_bytes"
            ),
            "peak_host_memory_bytes": result["peak_host_memory_bytes"],
            "wait_s": phases["producer_or_consumer_wait_s"],
            "dma_or_host_builder_s": phases["dma_or_host_builder_s"],
            "gpu_builder_s": phases["gpu_builder_s"],
            "trainer_s": phases["trainer_s"],
            "protocol_sha256": result["protocol_sha256"],
            "source": result["_source"],
        }
        raw_rows.append(row)
        for checkpoint in checkpoints:
            validation_rows.append(
                {
                    "mode": result["mode"],
                    "seed": result["seed"],
                    "optimizer_step": checkpoint["optimizer_step"],
                    "checkpoint_elapsed_s": checkpoint["checkpoint_elapsed_s"],
                    "training_loss": checkpoint["training_loss"],
                    "val_overall_accuracy": checkpoint["val_overall_accuracy"],
                    "val_nonstay_accuracy": checkpoint["val_nonstay_accuracy"],
                    "validation_elapsed_s": checkpoint["validation_elapsed_s"],
                }
            )
        losses = list(result.get("loss_history", []))
        for start in range(0, len(losses), 1000):
            window = losses[start : start + 1000]
            loss_rows.append(
                {
                    "mode": result["mode"],
                    "seed": result["seed"],
                    "window_end_step": start + len(window),
                    "mean_training_loss": statistics.fmean(float(v) for v in window),
                    "final_training_loss": window[-1],
                }
            )
        for threshold in THRESHOLDS:
            step, elapsed = _first_threshold(checkpoints, threshold)
            threshold_rows.append(
                {
                    "mode": result["mode"],
                    "seed": result["seed"],
                    "validation_threshold": threshold,
                    "first_optimizer_step": step,
                    "time_to_threshold_s": elapsed,
                    "reached": step is not None,
                }
            )

    _write_csv(output_dir / "e5_magat_runs.csv", raw_rows)
    _write_csv(output_dir / "e5_magat_validation_curve.csv", validation_rows)
    _write_csv(output_dir / "e5_magat_loss_windows.csv", loss_rows)
    _write_csv(output_dir / "e5_magat_time_to_threshold.csv", threshold_rows)

    summary_rows: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        rows = [row for row in raw_rows if row["mode"] == mode]
        if not rows:
            continue
        summary_rows.append(
            {
                "mode": mode,
                "runs": len(rows),
                "median_wall_s": _median(rows, "total_wall_s"),
                "min_wall_s": min(float(row["total_wall_s"]) for row in rows),
                "max_wall_s": max(float(row["total_wall_s"]) for row in rows),
                "median_samples_s": _median(rows, "samples_s"),
                "median_optimizer_steps_s": _median(rows, "optimizer_steps_s"),
                "median_best_validation_accuracy": _median(
                    rows, "best_validation_accuracy"
                ),
                "min_best_validation_accuracy": min(
                    float(row["best_validation_accuracy"]) for row in rows
                ),
                "max_best_validation_accuracy": max(
                    float(row["best_validation_accuracy"]) for row in rows
                ),
                "median_final_validation_accuracy": _median(
                    rows, "final_validation_accuracy"
                ),
                "min_final_validation_accuracy": min(
                    float(row["final_validation_accuracy"]) for row in rows
                ),
                "max_final_validation_accuracy": max(
                    float(row["final_validation_accuracy"]) for row in rows
                ),
                "median_peak_gpu_memory_bytes": _median(
                    rows, "peak_gpu_memory_bytes"
                ),
                "median_peak_host_memory_bytes": _median(
                    rows, "peak_host_memory_bytes"
                ),
                "median_h2d_bytes": _median(rows, "h2d_bytes"),
                "median_d2h_bytes": _median(rows, "d2h_bytes"),
                "median_wait_s": _median(rows, "wait_s"),
                "median_dma_or_host_builder_s": _median(
                    rows, "dma_or_host_builder_s"
                ),
                "median_gpu_builder_s": _median(rows, "gpu_builder_s"),
                "median_trainer_s": _median(rows, "trainer_s"),
            }
        )
    _write_csv(output_dir / "e5_magat_summary.csv", summary_rows)

    proposed = next(
        (row for row in summary_rows if row["mode"] == "proposed_async"), None
    )
    speedup_rows = []
    if proposed is not None:
        for row in summary_rows:
            if row["mode"] == "proposed_async":
                continue
            speedup_rows.append(
                {
                    "baseline_mode": row["mode"],
                    "proposed_mode": "proposed_async",
                    "wall_time_speedup": float(row["median_wall_s"])
                    / float(proposed["median_wall_s"]),
                    "throughput_speedup": float(proposed["median_samples_s"])
                    / float(row["median_samples_s"]),
                }
            )
    _write_csv(output_dir / "e5_magat_speedups.csv", speedup_rows)

    common_budget_s = min(float(row["median_wall_s"]) for row in summary_rows)
    equal_wall_rows = []
    for result in results:
        eligible = [
            checkpoint
            for checkpoint in result["checkpoint_history"]
            if float(checkpoint["checkpoint_elapsed_s"]) <= common_budget_s
        ]
        checkpoint = eligible[-1] if eligible else None
        equal_wall_rows.append(
            {
                "wall_budget_s": common_budget_s,
                "mode": result["mode"],
                "seed": result["seed"],
                "observed_checkpoint_step": (
                    checkpoint["optimizer_step"] if checkpoint else None
                ),
                "observed_validation_accuracy": (
                    checkpoint["val_overall_accuracy"] if checkpoint else None
                ),
                "status": "observed" if checkpoint else "below_1000_step_resolution",
            }
        )
    _write_csv(output_dir / "e5_magat_equal_wall.csv", equal_wall_rows)

    threshold_summary = []
    for mode in MODE_ORDER:
        if not groups[mode]:
            continue
        for threshold in THRESHOLDS:
            rows = [
                row
                for row in threshold_rows
                if row["mode"] == mode
                and float(row["validation_threshold"]) == threshold
                and row["reached"]
            ]
            threshold_summary.append(
                {
                    "mode": mode,
                    "validation_threshold": threshold,
                    "runs_total": len(groups[mode]),
                    "runs_reached": len(rows),
                    "median_first_optimizer_step": (
                        statistics.median(
                            int(row["first_optimizer_step"]) for row in rows
                        )
                        if rows
                        else None
                    ),
                    "median_time_to_threshold_s": (
                        statistics.median(
                            float(row["time_to_threshold_s"]) for row in rows
                        )
                        if rows
                        else None
                    ),
                }
            )
    _write_csv(
        output_dir / "e5_magat_time_to_threshold_summary.csv", threshold_summary
    )

    (output_dir / "validation_vs_steps.svg").write_text(
        _curve_svg(
            validation_rows,
            x_key="optimizer_step",
            x_label="Optimizer updates",
            title="E5 MAGAT equal-sample validation",
        ),
        encoding="utf-8",
    )

    protocols_without_seed = []
    for result in results:
        protocol = dict(result["protocol"])
        protocol.pop("seed", None)
        protocols_without_seed.append(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        )
    protocol_matched = len(set(protocols_without_seed)) == 1
    complete = all(len(groups[mode]) == 3 for mode in MODE_ORDER)
    summary_by_mode = {row["mode"]: row for row in summary_rows}
    lines = [
        "# E5-MAGAT matched training comparison",
        "",
        f"- Status: **{'complete' if complete else 'partial'}**",
        f"- Source root: `{root}`",
        f"- Protocol matched after excluding seed: **{protocol_matched}**",
        "- Workload: 256 agents, 4 expert environments, batch 1024, 10k updates.",
        "- Checkpoint selection: frozen validation accuracy every 1000 updates.",
        "",
        "## Performance and equal-sample quality",
        "",
        "| Path | Runs | Median wall | Samples/s | Best val. acc. | Final val. acc. | Peak GPU | Peak host |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODE_ORDER:
        row = summary_by_mode.get(mode)
        if row is None:
            continue
        lines.append(
            f"| {MODE_LABELS[mode]} | {row['runs']} | "
            f"{float(row['median_wall_s']):.2f} s | "
            f"{float(row['median_samples_s']):,.0f} | "
            f"{float(row['median_best_validation_accuracy']):.4f} "
            f"[{float(row['min_best_validation_accuracy']):.4f}, "
            f"{float(row['max_best_validation_accuracy']):.4f}] | "
            f"{float(row['median_final_validation_accuracy']):.4f} "
            f"[{float(row['min_final_validation_accuracy']):.4f}, "
            f"{float(row['max_final_validation_accuracy']):.4f}] | "
            f"{float(row['median_peak_gpu_memory_bytes']) / 2**20:.1f} MiB | "
            f"{float(row['median_peak_host_memory_bytes']) / 2**20:.1f} MiB |"
        )
    lines.extend(["", "## Speedup", ""])
    if speedup_rows:
        lines.extend(
            [
                "| Baseline | Proposed wall-time speedup | Proposed throughput speedup |",
                "|---|---:|---:|",
            ]
        )
        for row in speedup_rows:
            lines.append(
                f"| {MODE_LABELS[str(row['baseline_mode'])]} | "
                f"{float(row['wall_time_speedup']):.2f}× | "
                f"{float(row['throughput_speedup']):.2f}× |"
            )
    else:
        lines.append("Speedup is pending until proposed and baseline rows coexist.")
    lines.extend(
        [
            "",
            "## Host-observed stage and transfer accounting",
            "",
            "| Path | Wait / wall | DMA or host build / wall | GPU build / wall | Train / wall | H2D | D2H |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODE_ORDER:
        row = summary_by_mode.get(mode)
        if row is None:
            continue
        wall = float(row["median_wall_s"])
        lines.append(
            f"| {MODE_LABELS[mode]} | "
            f"{float(row['median_wait_s']) / wall:.1%} | "
            f"{float(row['median_dma_or_host_builder_s']) / wall:.1%} | "
            f"{float(row['median_gpu_builder_s']) / wall:.1%} | "
            f"{float(row['median_trainer_s']) / wall:.1%} | "
            f"{float(row['median_h2d_bytes']) / 2**20:.1f} MiB | "
            f"{float(row['median_d2h_bytes']) / 2**10:.1f} KiB |"
        )
    lines.extend(
        [
            "",
            "These ratios are host-observed instrumentation, not mutually exclusive "
            "GPU kernel shares. The synchronous path includes an explicit CUDA "
            "synchronization after reconstruction and training, whereas the asynchronous "
            "path records host dispatch time and allows queued CUDA work to overlap. "
            "Consequently, phase ratios should only diagnose wait/dispatch behavior within "
            "a path; the cross-path performance claim uses end-to-end wall time. Ratios "
            "also exclude validation, setup/checkpoint work, and overlapped producer work.",
            "",
            "## Time to fixed validation accuracy",
            "",
            "| Path | Threshold | Runs reached | Median first step | Median time |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in threshold_summary:
        step = row["median_first_optimizer_step"]
        elapsed = row["median_time_to_threshold_s"]
        lines.append(
            f"| {MODE_LABELS[str(row['mode'])]} | "
            f"{float(row['validation_threshold']):.2f} | "
            f"{row['runs_reached']}/{row['runs_total']} | "
            f"{'-' if step is None else f'{float(step):.0f}'} | "
            f"{'-' if elapsed is None else f'{float(elapsed):.2f} s'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Equal-sample quality is the validation distribution after the same 10k optimizer updates; it is not a claim that individual runs converge identically.",
            f"- Equal-wall-time uses the fastest path's median terminal time ({common_budget_s:.2f} s). "
            "A missing value means that path did not reach the first 1000-step validation checkpoint within this budget; values are not interpolated.",
            "- The frozen raw compact-stage hashes matched in the diagnostic run. Remaining per-run learning-curve variation is therefore treated as CUDA training nondeterminism and is reported across three seeds.",
            "- `training_loss` in the validation-curve CSV is the online batch loss at checkpoint time. The current frozen evaluator emits validation accuracy, not validation cross entropy.",
            "",
            "## Artifacts",
            "",
            "- `e5_magat_runs.csv`: one row per formal run.",
            "- `e5_magat_summary.csv`: median runtime, throughput, quality, memory, and phase totals.",
            "- `e5_magat_validation_curve.csv`: all 1000-step validation checkpoints.",
            "- `e5_magat_loss_windows.csv`: 1000-update online-loss windows.",
            "- `e5_magat_equal_wall.csv`: observed validation checkpoint under a common wall budget.",
            "- `e5_magat_time_to_threshold.csv`: first observed crossing of 0.75/0.80/0.85.",
            "- `e5_magat_time_to_threshold_summary.csv`: threshold reach counts and median time.",
            "- `validation_vs_steps.svg`: median validation curve over seeds.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    render(args.root, args.output_dir, allow_partial=args.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
