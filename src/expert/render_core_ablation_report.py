"""Render the clean 2x2 MAGAT component-ablation report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from expert.minimal_core_ablation_runner import CELLS


SEEDS = (42, 43, 44)


def _load_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("experiment") == "CORE-ABLATION"
            and row.get("status") == "ok"
            and row.get("accepted")
        ):
            row["_path"] = str(path)
            rows.append(row)
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row["cell"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"duplicate core-ablation row: {key}")
        seen.add(key)
        grouped[key[0]].append(row)
    expected = {(cell, seed) for cell in CELLS for seed in SEEDS}
    if seen != expected:
        raise ValueError(
            f"core-ablation matrix mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    summary = []
    for cell, (representation, depth) in CELLS.items():
        group = grouped[cell]
        wall_s = _median(group, "total_wall_s")
        wait_s = _median(group, "consumer_wait_s")
        host_s = _median(group, "consumer_host_builder_and_h2d_s") if (
            representation == "full_host_materialized"
        ) else 0.0
        compact_h2d_s = _median(group, "consumer_dma_sync_s") if (
            representation == "compact_cuda_replay"
        ) else 0.0
        replay_s = _median(group, "consumer_gpu_builder_s") if (
            representation == "compact_cuda_replay"
        ) else 0.0
        train_s = _median(group, "consumer_train_step_s")
        summary.append(
            {
                "cell": cell,
                "representation": representation,
                "ring_buffer_steps": depth,
                "repetitions": len(group),
                "median_samples_s": _median(group, "samples_s"),
                "median_wall_s": wall_s,
                "median_h2d_bytes": _median(group, "h2d_bytes"),
                "median_consumer_wait_s": wait_s,
                "median_wait_share": wait_s / wall_s,
                "median_host_materialization_h2d_s": host_s,
                "median_compact_h2d_s": compact_h2d_s,
                "median_cuda_replay_s": replay_s,
                "median_train_s": train_s,
                "median_peak_gpu_bytes": _median(group, "peak_gpu_memory_bytes"),
                "median_peak_host_bytes": _median(group, "peak_host_memory_bytes"),
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

    lookup = {row["cell"]: row for row in summary}
    labels = ["S0\nFull, D1", "S1\nCompact, D1", "S2\nFull, D128", "S3\nCompact, D128"]
    values = [
        lookup[cell]["median_samples_s"] / 1000.0
        for cell in CELLS
    ]
    colors = ["#9ca3af", "#2563eb", "#6b7280", "#16a34a"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{value:.1f}k",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Training throughput (k samples/s)")
    ax.set_ylim(0, max(values) * 1.16)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(path: Path, summary: list[dict[str, Any]]) -> None:
    lookup = {row["cell"]: row for row in summary}
    s0 = lookup["s0_full_one_stage"]
    s1 = lookup["s1_compact_one_stage"]
    s2 = lookup["s2_full_deep_ring"]
    s3 = lookup["s3_compact_deep_ring"]
    late_one = s1["median_samples_s"] / s0["median_samples_s"]
    late_deep = s3["median_samples_s"] / s2["median_samples_s"]
    ring_full = s2["median_samples_s"] / s0["median_samples_s"]
    ring_compact = s3["median_samples_s"] / s1["median_samples_s"]
    full_system = s3["median_samples_s"] / s0["median_samples_s"]
    interaction = ring_compact / ring_full
    h2d_reduction = s0["median_h2d_bytes"] / s1["median_h2d_bytes"]
    lines = [
        "# Core MAGAT Component Ablation",
        "",
        "Clean 2x2 design: host-expanded/full-transfer versus compact transfer "
        "with CUDA replay, crossed with one-stage backpressure versus a "
        "128-stage producer ring. Every cell uses 256 agents, four maps and "
        "expert processes, batch 1024, 100 optimizer updates, 102,400 samples, "
        "and three paired seeds. Validation, monitoring, checkpoint I/O, "
        "initialization, and teardown are outside the timed interval.",
        "",
        "| Cell | Data path | Ring | Throughput | Wall | H2D | Wait |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "full_host_materialized": "Host materialization + full transfer",
        "compact_cuda_replay": "Compact transfer + CUDA replay",
    }
    for cell in CELLS:
        row = lookup[cell]
        lines.append(
            f"| {cell.split('_', 1)[0].upper()} | "
            f"{labels[row['representation']]} | "
            f"{row['ring_buffer_steps']} | "
            f"{row['median_samples_s'] / 1000.0:.1f}k samples/s | "
            f"{row['median_wall_s']:.2f} s | "
            f"{row['median_h2d_bytes'] / 2**20:.2f} MiB | "
            f"{row['median_wait_share'] * 100.0:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Factor effects",
            "",
            f"- Compact transfer + CUDA replay at depth 1: **{late_one:.2f}x**.",
            f"- Compact transfer + CUDA replay at depth 128: **{late_deep:.2f}x**.",
            f"- Deep-ring gain with full materialization: **{ring_full:.3f}x**.",
            f"- Deep-ring gain with compact/CUDA replay: **{ring_compact:.3f}x**.",
            f"- Complete S3/S0 gain: **{full_system:.2f}x**.",
            f"- Multiplicative interaction: **{interaction:.3f}x**.",
            f"- Full/compact H2D payload ratio: **{h2d_reduction:.1f}x**.",
            "",
            "The dominant component is compact transfer with CUDA late "
            "materialization. In the full path, CPU graph/cost-to-go "
            "materialization consumes nearly the entire wall time, so buffering "
            "expert outputs cannot hide it. After late materialization removes "
            "that bottleneck, the deep ring provides a smaller but repeatable "
            "increment by overlapping expert production with GPU consumption.",
            "",
            "This table is the component ablation. Producer-count, ring-depth, "
            "batch-size, and topology-skew scans are sensitivity or mechanism "
            "characterization and must not be presented as independent method "
            "components.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(*, input_root: Path, output_dir: Path) -> None:
    summary = _summarize(_load_rows(input_root))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "core_ablation_summary.csv", summary)
    _render(output_dir / "core_ablation.pdf", summary)
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(input_root=args.input_root, output_dir=args.output_dir)
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
