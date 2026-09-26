"""Render the F3 bounded-ring depth sensitivity report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


RING_DEPTHS = (2, 4, 8, 32, 128)
SEEDS = (42, 43, 44)
FRONTIER_ROWS = 2048
FEATURE_DIM = 8


def _load_rows(f3_root: Path, f2_roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(f3_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("experiment") != "F3":
            continue
        if row.get("status") != "ok" or not row.get("accepted"):
            continue
        row["_path"] = str(path)
        rows.append(row)

    for root in f2_roots:
        for path in sorted(root.glob("**/result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("experiment") != "F2":
                continue
            if row.get("status") != "ok" or not row.get("accepted"):
                continue
            if int(row.get("num_agents", 0)) != 256:
                continue
            if int(row.get("num_producer_processes", 0)) != 8:
                continue
            if int(row.get("async_ring_buffer_steps", 0)) != 128:
                continue
            row = dict(row)
            row["ring_depth"] = 128
            row["ring_payload_bytes"] = (
                FRONTIER_ROWS * 128 * FEATURE_DIM * 2
            )
            row["_path"] = str(path)
            rows.append(row)
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for row in rows:
        depth = int(row["ring_depth"])
        seed = int(row["seed"])
        key = (depth, seed)
        if key in seen:
            raise ValueError(f"duplicate F3 row depth={depth}, seed={seed}")
        seen.add(key)
        grouped[depth].append(row)

    expected = {(depth, seed) for depth in RING_DEPTHS for seed in SEEDS}
    if seen != expected:
        raise ValueError(
            f"F3 matrix mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )

    summary: list[dict[str, Any]] = []
    for depth in RING_DEPTHS:
        group = grouped[depth]
        wall_s = _median(group, "total_wall_s")
        wait_s = _median(group, "consumer_wait_s")
        summary.append(
            {
                "ring_depth": depth,
                "ring_payload_bytes": FRONTIER_ROWS
                * depth
                * FEATURE_DIM
                * 2,
                "repetitions": len(group),
                "median_wall_s": wall_s,
                "median_samples_s": _median(group, "samples_s"),
                "median_consumer_wait_s": wait_s,
                "median_consumer_wait_share": wait_s / wall_s,
                "median_ring_write_s_max": _median(
                    group, "worker_ringbuffer_write_s_max"
                ),
                "min_samples_s": min(float(row["samples_s"]) for row in group),
                "max_samples_s": max(float(row["samples_s"]) for row in group),
            }
        )
    reference = float(summary[-1]["median_samples_s"])
    for row in summary:
        row["throughput_fraction_of_depth128"] = (
            float(row["median_samples_s"]) / reference
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    depths = [row["ring_depth"] for row in summary]
    throughput = [row["median_samples_s"] / 1000.0 for row in summary]
    wait = [row["median_consumer_wait_share"] * 100.0 for row in summary]
    write = [row["median_ring_write_s_max"] for row in summary]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].plot(depths, throughput, marker="o", linewidth=2, color="#2563eb")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(depths, [str(value) for value in depths])
    axes[0].set_xlabel("Ring depth (complete frontiers)")
    axes[0].set_ylabel("Training throughput (k samples/s)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        depths,
        wait,
        marker="o",
        linewidth=2,
        color="#dc2626",
        label="Consumer wait share",
    )
    second = axes[1].twinx()
    second.plot(
        depths,
        write,
        marker="s",
        linewidth=2,
        color="#f59e0b",
        label="Max worker ring-write time",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(depths, [str(value) for value in depths])
    axes[1].set_xlabel("Ring depth (complete frontiers)")
    axes[1].set_ylabel("Consumer wait share (%)")
    second.set_ylabel("Accumulated blocked write time (s)")
    axes[1].grid(alpha=0.25)
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axes[1].legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# F3 Bounded-Ring Depth Sensitivity",
        "",
        "Fixed configuration: 256 agents, eight logical environments, eight "
        "producer processes, batch size 1024, 1,024,000 samples per row, and "
        "three paired seeds. Depth is measured in complete eight-environment "
        "frontiers. The depth-128 endpoint reuses the protocol-identical F2 rows.",
        "",
        "| Depth | Raw ring payload | Throughput | vs depth 128 | "
        "Consumer wait | Max worker blocked write |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        payload_kib = row["ring_payload_bytes"] / 1024.0
        lines.append(
            f"| {row['ring_depth']} | {payload_kib:.0f} KiB | "
            f"{row['median_samples_s'] / 1000.0:.1f}k samples/s | "
            f"{row['throughput_fraction_of_depth128'] * 100.0:.1f}% | "
            f"{row['median_consumer_wait_share'] * 100.0:.1f}% | "
            f"{row['median_ring_write_s_max']:.2f} s |"
        )
    lines.extend(
        [
            "",
            "Depths 2--8 are too shallow for eight heterogeneous expert "
            "streams: producers repeatedly block on occupied slots while the "
            "consumer also waits for the slowest environment needed to complete "
            "the next aligned frontier. Depth 32 absorbs most of this skew and "
            "reaches about 90% of depth-128 throughput with only 1 MiB of raw "
            "ring payload. Depth 128 adds another modest gain and costs 4 MiB.",
            "",
            "This experiment isolates scheduling capacity, not data semantics: "
            "every row still publishes and consumes complete same-timestep "
            "environment blocks. The result supports a bounded but nontrivial "
            "ring rather than either an unbounded queue or a one-slot-per-worker "
            "design.",
            "",
            "Validation, checkpoint I/O, initialization, teardown, and NVML "
            "sampling are outside the timed region. CUDA compute remained "
            "operational despite the host's NVML driver/library mismatch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    *, f3_root: Path, f2_roots: Sequence[Path], output_dir: Path
) -> None:
    rows = _load_rows(f3_root, f2_roots)
    summary = _summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "f3_summary.csv", summary)
    _render(output_dir / "f3_ring_depth.svg", summary)
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-root", required=True, type=Path)
    parser.add_argument("--f2-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(
        f3_root=args.f3_root,
        f2_roots=args.f2_root,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
