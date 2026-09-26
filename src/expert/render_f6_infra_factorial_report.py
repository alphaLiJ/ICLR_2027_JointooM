"""Render the F6 ring-depth by transfer-mode infrastructure factorial."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


TRANSFER_MODES = ("sync", "async")
RING_DEPTHS = (2, 128)
SEEDS = (42, 43, 44)


def _load_rows(
    f6_root: Path,
    f3_root: Path,
    f2_roots: Sequence[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(f6_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("experiment") == "F6"
            and row.get("status") == "ok"
            and row.get("accepted")
            and row.get("transfer_mode") == "sync"
            and int(row.get("ring_depth", 0)) in RING_DEPTHS
        ):
            row["_path"] = str(path)
            rows.append(row)
    for path in sorted(f3_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("experiment") == "F3"
            and row.get("status") == "ok"
            and row.get("accepted")
            and int(row.get("ring_depth", 0)) == 2
        ):
            row = dict(row)
            row["transfer_mode"] = "async"
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
                row["ring_depth"] = 128
                row["transfer_mode"] = "async"
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
            str(row["transfer_mode"]),
            int(row["ring_depth"]),
            int(row["seed"]),
        )
        if key in seen:
            raise ValueError(f"duplicate F6 factorial row {key}")
        seen.add(key)
        grouped[(key[0], key[1])].append(row)
    expected = {
        (mode, depth, seed)
        for mode in TRANSFER_MODES
        for depth in RING_DEPTHS
        for seed in SEEDS
    }
    if seen != expected:
        raise ValueError(
            f"F6 matrix mismatch: missing={sorted(expected - seen)}, "
            f"extra={sorted(seen - expected)}"
        )
    summary: list[dict[str, Any]] = []
    for mode in TRANSFER_MODES:
        for depth in RING_DEPTHS:
            group = grouped[(mode, depth)]
            wall_s = _median(group, "total_wall_s")
            wait_s = _median(group, "consumer_wait_s")
            summary.append(
                {
                    "transfer_mode": mode,
                    "ring_depth": depth,
                    "repetitions": len(group),
                    "median_samples_s": _median(group, "samples_s"),
                    "median_wall_s": wall_s,
                    "median_consumer_wait_s": wait_s,
                    "median_consumer_wait_share": wait_s / wall_s,
                    "median_dma_sync_s": _median(group, "consumer_dma_sync_s"),
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
        (row["transfer_mode"], row["ring_depth"]): row for row in summary
    }
    styles = {
        "sync": ("Consumer-triggered synchronous H2D", "#dc2626"),
        "async": ("Background copy-stream prefetch", "#2563eb"),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for mode in TRANSFER_MODES:
        label, color = styles[mode]
        ax.plot(
            RING_DEPTHS,
            [
                lookup[(mode, depth)]["median_samples_s"] / 1000.0
                for depth in RING_DEPTHS
            ],
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
        (row["transfer_mode"], row["ring_depth"]): row for row in summary
    }
    sync_shallow = lookup[("sync", 2)]
    sync_deep = lookup[("sync", 128)]
    async_shallow = lookup[("async", 2)]
    async_deep = lookup[("async", 128)]
    ring_gain_sync = sync_deep["median_samples_s"] / sync_shallow["median_samples_s"]
    ring_gain_async = (
        async_deep["median_samples_s"] / async_shallow["median_samples_s"]
    )
    async_gain_shallow = (
        async_shallow["median_samples_s"] / sync_shallow["median_samples_s"]
    )
    async_gain_deep = async_deep["median_samples_s"] / sync_deep["median_samples_s"]
    interaction = ring_gain_async / ring_gain_sync
    lines = [
        "# F6 Ring Depth × Transfer Mode Infrastructure Factorial",
        "",
        "Fixed configuration: 256 agents, eight logical environments and "
        "producer processes, batch 1024, 1,024,000 samples per row, and three "
        "paired seeds. Existing accepted asynchronous endpoints from F2/F3 "
        "are reused; only the six synchronous cells were newly measured.",
        "",
        "| Transfer mode | Depth | Throughput | Wall time | Consumer wait | "
        "DMA/event wait |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "sync": "Consumer-triggered synchronous H2D",
        "async": "Background copy-stream prefetch",
    }
    for mode in TRANSFER_MODES:
        for depth in RING_DEPTHS:
            row = lookup[(mode, depth)]
            lines.append(
                f"| {labels[mode]} | {depth} | "
                f"{row['median_samples_s'] / 1000.0:.1f}k samples/s | "
                f"{row['median_wall_s']:.2f} s | "
                f"{row['median_consumer_wait_share'] * 100.0:.1f}% | "
                f"{row['median_dma_sync_s']:.3f} s |"
            )
    lines.extend(
        [
            "",
            "## Factor effects",
            "",
            f"- Deep ring gain under synchronous transfer: **{ring_gain_sync:.2f}×**.",
            f"- Deep ring gain under asynchronous prefetch: **{ring_gain_async:.2f}×**.",
            f"- Async/sync at depth 2: **{async_gain_shallow:.3f}×**.",
            f"- Async/sync at depth 128: **{async_gain_deep:.3f}×**.",
            f"- Multiplicative interaction: **{interaction:.3f}×**.",
            "",
            "The causal result is that ring capacity, not H2D overlap, explains "
            "the measured improvement in this workload. Deep buffering removes "
            "most consumer starvation in both transfer modes. The transferred "
            "frontier is small enough that background copy-stream prefetch adds "
            "no measurable throughput benefit and is slightly slower at the "
            "deep endpoint. This does not imply that asynchronous DMA is "
            "universally useless; it shows that it is not the active bottleneck "
            "for this MAPF configuration.",
            "",
            "The synchronous cell retains the same shared-memory producer ring "
            "and differs only in who triggers the H2D stage. It is therefore a "
            "targeted infrastructure ablation, not a comparison against the "
            "older compact-sync end-to-end pipeline.",
            "",
            "Validation, checkpoint I/O, initialization, teardown, and NVML "
            "sampling are outside the timed region.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(
    *,
    f6_root: Path,
    f3_root: Path,
    f2_roots: Sequence[Path],
    output_dir: Path,
) -> None:
    summary = _summarize(_load_rows(f6_root, f3_root, f2_roots))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "f6_summary.csv", summary)
    _render(output_dir / "f6_infra_factorial.svg", summary)
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f6-root", required=True, type=Path)
    parser.add_argument("--f3-root", required=True, type=Path)
    parser.add_argument("--f2-root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(
        f6_root=args.f6_root,
        f3_root=args.f3_root,
        f2_roots=args.f2_root,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
