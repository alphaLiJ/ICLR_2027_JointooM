"""Render the paired core-component ablation used by the paper."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Sequence


SEEDS = (42, 43, 44)


def _load_cells(
    roots: Sequence[Path],
    cells: Sequence[str],
    *,
    num_steps: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped = {cell: [] for cell in cells}
    seen: set[tuple[str, int]] = set()
    for root in roots:
        for path in sorted(root.glob("**/result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            cell = str(row.get("cell"))
            if (
                cell not in grouped
                or row.get("experiment") != "CORE-ABLATION"
                or row.get("status") != "ok"
                or not row.get("accepted")
                or int(row.get("num_steps", 0)) != num_steps
            ):
                continue
            key = (cell, int(row["seed"]))
            if key in seen:
                raise ValueError(f"duplicate paired-ablation row: {key}")
            seen.add(key)
            grouped[cell].append(row)
    expected = {(cell, seed) for cell in cells for seed in SEEDS}
    if seen != expected:
        raise ValueError(
            f"paired-ablation mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    return grouped


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _summarize(
    short_rows: dict[str, list[dict[str, Any]]],
    long_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    specifications = (
        ("late_materialization", "full_one_stage", "s0_full_one_stage", short_rows),
        (
            "late_materialization",
            "compact_one_stage",
            "s1_compact_one_stage",
            short_rows,
        ),
        ("ring_decoupling", "compact_one_stage", "s1_compact_one_stage", long_rows),
        ("ring_decoupling", "compact_deep_ring", "s3_compact_deep_ring", long_rows),
    )
    summary = []
    for comparison, variant, cell, source in specifications:
        rows = source[cell]
        wall = _median(rows, "total_wall_s")
        wait = _median(rows, "consumer_wait_s")
        summary.append(
            {
                "comparison": comparison,
                "variant": variant,
                "cell": cell,
                "num_steps": int(rows[0]["num_steps"]),
                "samples": int(rows[0]["samples_processed"]),
                "repetitions": len(rows),
                "median_samples_s": _median(rows, "samples_s"),
                "median_wall_s": wall,
                "median_h2d_bytes": _median(rows, "h2d_bytes"),
                "median_consumer_wait_s": wait,
                "median_consumer_wait_share": wait / wall,
                "median_train_s": _median(rows, "consumer_train_step_s"),
                "median_peak_gpu_bytes": _median(rows, "peak_gpu_memory_bytes"),
                "median_peak_host_bytes": _median(rows, "peak_host_memory_bytes"),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    lookup = {(row["comparison"], row["variant"]): row for row in summary}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    panels = (
        (
            "Late materialization (100 updates)",
            "late_materialization",
            ("full_one_stage", "compact_one_stage"),
            ("Full host", "Compact + replay"),
        ),
        (
            "Ring decoupling (1,000 updates)",
            "ring_decoupling",
            ("compact_one_stage", "compact_deep_ring"),
            ("One stage", "Deep ring"),
        ),
    )
    for ax, (title, comparison, variants, labels) in zip(axes, panels):
        values = [
            lookup[(comparison, variant)]["median_samples_s"] / 1000.0
            for variant in variants
        ]
        bars = ax.bar(labels, values, color=("#9ca3af", "#2563eb"))
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.025,
                f"{value:.1f}k",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Training throughput (k samples/s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(path: Path, summary: list[dict[str, Any]]) -> None:
    lookup = {(row["comparison"], row["variant"]): row for row in summary}
    full = lookup[("late_materialization", "full_one_stage")]
    compact_short = lookup[("late_materialization", "compact_one_stage")]
    compact_long = lookup[("ring_decoupling", "compact_one_stage")]
    deep = lookup[("ring_decoupling", "compact_deep_ring")]
    late_gain = compact_short["median_samples_s"] / full["median_samples_s"]
    payload_gain = full["median_h2d_bytes"] / compact_short["median_h2d_bytes"]
    ring_gain = deep["median_samples_s"] / compact_long["median_samples_s"]
    lines = [
        "# Core Training-System Component Ablation",
        "",
        "Two internally matched comparisons isolate the actual method "
        "components. Both use 256 agents, four maps/expert processes, batch "
        "1024, and three paired seeds. The expensive full-materialization pair "
        "uses 100 updates (102,400 samples); the compact ring pair uses 1,000 "
        "updates (1,024,000 samples) to expose steady-state producer/consumer "
        "overlap. Results are compared only within a pair.",
        "",
        "## Compact transfer + CUDA late materialization",
        "",
        "| Variant | Updates | Throughput | Wall | H2D |",
        "|---|---:|---:|---:|---:|",
        f"| Host materialization + full transfer | 100 | "
        f"{full['median_samples_s'] / 1000.0:.1f}k samples/s | "
        f"{full['median_wall_s']:.2f} s | "
        f"{full['median_h2d_bytes'] / 2**20:.2f} MiB |",
        f"| Compact transfer + CUDA replay | 100 | "
        f"{compact_short['median_samples_s'] / 1000.0:.1f}k samples/s | "
        f"{compact_short['median_wall_s']:.2f} s | "
        f"{compact_short['median_h2d_bytes'] / 2**20:.2f} MiB |",
        "",
        f"Late materialization improves throughput by **{late_gain:.2f}x** and "
        f"reduces measured H2D payload by **{payload_gain:.1f}x**.",
        "",
        "## Bounded-ring producer/consumer decoupling",
        "",
        "| Variant | Updates | Throughput | Wall | Consumer wait |",
        "|---|---:|---:|---:|---:|",
        f"| One-stage backpressure | 1,000 | "
        f"{compact_long['median_samples_s'] / 1000.0:.1f}k samples/s | "
        f"{compact_long['median_wall_s']:.2f} s | "
        f"{compact_long['median_consumer_wait_share'] * 100.0:.1f}% |",
        f"| Deep ring (128 stages) | 1,000 | "
        f"{deep['median_samples_s'] / 1000.0:.1f}k samples/s | "
        f"{deep['median_wall_s']:.2f} s | "
        f"{deep['median_consumer_wait_share'] * 100.0:.1f}% |",
        "",
        f"Ring decoupling improves compact-path throughput by **{ring_gain:.2f}x** "
        f"and reduces consumer wait from "
        f"**{compact_long['median_consumer_wait_share'] * 100.0:.1f}%** to "
        f"**{deep['median_consumer_wait_share'] * 100.0:.1f}%**.",
        "",
        "Validation, monitoring, checkpoint I/O, initialization, and teardown "
        "are outside the timed interval. Stage hashes match across all four "
        "implementation paths in the separate two-frontier audit.",
        "",
        "Producer count, ring depth, batch size, and topology skew are reported "
        "separately as sensitivity and mechanism characterization, not as "
        "method-component ablations. The DMA copy-trigger diagnostic is not "
        "used as paper evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    *,
    short_root: Path,
    long_roots: Sequence[Path],
    output_dir: Path,
) -> None:
    short = _load_cells(
        [short_root],
        ("s0_full_one_stage", "s1_compact_one_stage"),
        num_steps=100,
    )
    long = _load_cells(
        long_roots,
        ("s1_compact_one_stage", "s3_compact_deep_ring"),
        num_steps=1000,
    )
    summary = _summarize(short, long)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "core_component_ablation.csv", summary)
    _render(output_dir / "core_component_ablation.pdf", summary)
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-root", required=True, type=Path)
    parser.add_argument("--long-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(
        short_root=args.short_root,
        long_roots=args.long_root,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
