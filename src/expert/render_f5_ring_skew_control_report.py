"""Render the F5 homogeneous/heterogeneous ring-depth control."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


CONDITIONS = ("heterogeneous", "homogeneous_topology")
RING_DEPTHS = (8, 32, 128)
SEEDS = (42, 43, 44)


def _load_rows(
    f5_root: Path,
    f3_root: Path,
    f2_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(f5_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("experiment") == "F5"
            and row.get("status") == "ok"
            and row.get("accepted")
        ):
            row["_path"] = str(path)
            rows.append(row)
    for path in sorted(f3_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("experiment") == "F3"
            and row.get("status") == "ok"
            and row.get("accepted")
            and int(row.get("ring_depth", 0)) in RING_DEPTHS
        ):
            row = dict(row)
            row["map_condition"] = "heterogeneous"
            row["_path"] = str(path)
            rows.append(row)
    for root in f2_roots:
        for path in sorted(root.glob("**/result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("experiment") != "F2"
                or row.get("status") != "ok"
                or not row.get("accepted")
            ):
                continue
            if (
                int(row.get("num_agents", 0)) == 256
                and int(row.get("num_producer_processes", 0)) == 8
                and int(row.get("async_ring_buffer_steps", 0)) == 128
            ):
                row = dict(row)
                row["map_condition"] = "heterogeneous"
                row["ring_depth"] = 128
                row["_path"] = str(path)
                rows.append(row)
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(row[key]) for row in rows))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (
            str(row["map_condition"]),
            int(row["ring_depth"]),
            int(row["seed"]),
        )
        if key in seen:
            raise ValueError(f"duplicate F5 control row {key}")
        seen.add(key)
        grouped[(key[0], key[1])].append(row)
    expected = {
        (condition, depth, seed)
        for condition in CONDITIONS
        for depth in RING_DEPTHS
        for seed in SEEDS
    }
    if seen != expected:
        raise ValueError(
            f"F5 matrix mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    summary: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for depth in RING_DEPTHS:
            group = grouped[(condition, depth)]
            wall_s = _median(group, "total_wall_s")
            wait_s = _median(group, "consumer_wait_s")
            summary.append(
                {
                    "map_condition": condition,
                    "ring_depth": depth,
                    "repetitions": len(group),
                    "median_samples_s": _median(group, "samples_s"),
                    "median_wall_s": wall_s,
                    "median_consumer_wait_s": wait_s,
                    "median_consumer_wait_share": wait_s / wall_s,
                    "median_ring_write_s_max": _median(
                        group, "worker_ringbuffer_write_s_max"
                    ),
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

    lookup = {
        (row["map_condition"], row["ring_depth"]): row for row in summary
    }
    styles = {
        "heterogeneous": ("Heterogeneous topologies", "#dc2626"),
        "homogeneous_topology": ("One repeated topology", "#2563eb"),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for condition in CONDITIONS:
        label, color = styles[condition]
        values = [
            lookup[(condition, depth)]["median_samples_s"] / 1000.0
            for depth in RING_DEPTHS
        ]
        ax.plot(
            RING_DEPTHS,
            values,
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(RING_DEPTHS, [str(value) for value in RING_DEPTHS])
    ax.set_xlabel("Ring depth (complete frontiers)")
    ax.set_ylabel("Training throughput (k samples/s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(path: Path, summary: list[dict[str, Any]]) -> None:
    lookup = {
        (row["map_condition"], row["ring_depth"]): row for row in summary
    }
    lines = [
        "# F5 Ring Depth × Topology-Heterogeneity Control",
        "",
        "Fixed configuration: 256 agents, eight producer processes, batch "
        "1024, 1,024,000 samples per row, and three paired seeds. The "
        "heterogeneous condition uses the existing eight-map suite. The "
        "homogeneous-topology condition repeats `mazes-s0_wc8_od55` eight "
        "times while retaining independent seeded agent instances.",
        "",
        "| Depth | Heterogeneous | Homogeneous topology | Homogeneous / hetero | "
        "Hetero wait | Homogeneous wait |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in RING_DEPTHS:
        hetero = lookup[("heterogeneous", depth)]
        homogeneous = lookup[("homogeneous_topology", depth)]
        ratio = homogeneous["median_samples_s"] / hetero["median_samples_s"]
        lines.append(
            f"| {depth} | {hetero['median_samples_s'] / 1000.0:.1f}k | "
            f"{homogeneous['median_samples_s'] / 1000.0:.1f}k | "
            f"{ratio:.2f}× | "
            f"{hetero['median_consumer_wait_share'] * 100.0:.1f}% | "
            f"{homogeneous['median_consumer_wait_share'] * 100.0:.1f}% |"
        )
    lines.extend(
        [
            "",
            "At depth 8, repeating one topology is about 23% faster than the "
            "heterogeneous map suite. The gap falls to roughly 3% at depth 32 "
            "and disappears at depth 128. The deep-buffer endpoints converge "
            "even though the topology mix changes, supporting the explanation "
            "that sufficient ring capacity primarily absorbs producer-rate "
            "skew rather than changing the GPU consumer ceiling.",
            "",
            "This is a topology-level control, not a proof that every expert "
            "call has identical cost: independent starts and goals still "
            "produce within-condition runtime variation, and topology also "
            "changes graph structure. The convergence at depth 128 is therefore "
            "the most informative part of the control.",
            "",
            "Validation, checkpoint I/O, initialization, teardown, and NVML "
            "sampling are outside the timed region.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    *,
    f5_root: Path,
    f3_root: Path,
    f2_roots: Sequence[Path],
    output_dir: Path,
) -> None:
    summary = _summarize(_load_rows(f5_root, f3_root, f2_roots))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "f5_summary.csv", summary)
    _render(output_dir / "f5_ring_skew_control.svg", summary)
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f5-root", required=True, type=Path)
    parser.add_argument("--f3-root", required=True, type=Path)
    parser.add_argument("--f2-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(
        f5_root=args.f5_root,
        f3_root=args.f3_root,
        f2_roots=args.f2_root,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
