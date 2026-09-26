"""Render the F2 agent-count by producer-count scaling report."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


AGENT_COUNTS = (128, 256, 512)
PRODUCER_COUNTS = (1, 2, 4, 8)


def _load_rows(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for root in roots:
        for path in sorted(root.glob("**/result.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("experiment") != "F2":
                continue
            if row.get("status") != "ok" or not row.get("accepted"):
                continue
            key = (
                int(row["num_agents"]),
                int(row["num_producer_processes"]),
                int(row["seed"]),
            )
            if key in seen:
                raise ValueError(f"duplicate accepted F2 row {key}: {path}")
            seen.add(key)
            row["_path"] = str(path)
            rows.append(row)
    return rows


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(statistics.median(values))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["num_agents"]), int(row["num_producer_processes"]))
        ].append(row)

    expected = {(a, p) for a in AGENT_COUNTS for p in PRODUCER_COUNTS}
    missing = expected - set(grouped)
    if missing:
        raise ValueError(f"missing F2 cells: {sorted(missing)}")

    summary: list[dict[str, Any]] = []
    for num_agents in AGENT_COUNTS:
        for num_producers in PRODUCER_COUNTS:
            group = grouped[(num_agents, num_producers)]
            if len(group) != 3:
                raise ValueError(
                    f"expected three repetitions for A={num_agents}, "
                    f"P={num_producers}; got {len(group)}"
                )
            wall_s = _median(group, "total_wall_s")
            wait_s = _median(group, "consumer_wait_s")
            assert wall_s is not None and wait_s is not None
            summary.append(
                {
                    "num_agents": num_agents,
                    "train_batch_size": int(group[0]["train_batch_size"]),
                    "num_producers": num_producers,
                    "repetitions": len(group),
                    "median_wall_s": wall_s,
                    "median_samples_s": _median(group, "samples_s"),
                    "median_optimizer_steps_s": _median(
                        group, "optimizer_steps_s"
                    ),
                    "median_consumer_wait_s": wait_s,
                    "median_consumer_wait_share": wait_s / wall_s,
                    "median_consumer_train_s": _median(
                        group, "consumer_train_step_s"
                    ),
                    "median_consumer_dma_s": _median(
                        group, "consumer_dma_sync_s"
                    ),
                    "median_consumer_refresh_s": _median(
                        group, "consumer_gpu_builder_s"
                    ),
                    "median_ring_write_s_max": _median(
                        group, "worker_ringbuffer_write_s_max"
                    ),
                    "median_peak_gpu_memory_bytes": _median(
                        group, "peak_gpu_memory_bytes"
                    ),
                    "min_samples_s": min(float(row["samples_s"]) for row in group),
                    "max_samples_s": max(float(row["samples_s"]) for row in group),
                }
            )

    lookup = {
        (row["num_agents"], row["num_producers"]): row for row in summary
    }
    for row in summary:
        p1 = lookup[(row["num_agents"], 1)]
        row["speedup_vs_p1"] = (
            float(row["median_samples_s"]) / float(p1["median_samples_s"])
        )
        if row["num_producers"] == 1:
            row["incremental_speedup"] = 1.0
        else:
            previous = lookup[
                (row["num_agents"], row["num_producers"] // 2)
            ]
            row["incremental_speedup"] = (
                float(row["median_samples_s"])
                / float(previous["median_samples_s"])
            )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_scaling(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    lookup = {
        (row["num_agents"], row["num_producers"]): row for row in summary
    }
    colors = {128: "#2563eb", 256: "#16a34a", 512: "#dc2626"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for num_agents in AGENT_COUNTS:
        values = [
            lookup[(num_agents, p)]["median_samples_s"] / 1000.0
            for p in PRODUCER_COUNTS
        ]
        ax.plot(
            PRODUCER_COUNTS,
            values,
            marker="o",
            linewidth=2,
            color=colors[num_agents],
            label=f"{num_agents} agents (batch={num_agents * 4})",
        )
    ax.set_xticks(PRODUCER_COUNTS)
    ax.set_xlabel("Expert producer processes")
    ax.set_ylabel("Training throughput (k samples/s)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_heatmap(
    path: Path,
    summary: list[dict[str, Any]],
    *,
    key: str,
    title: str,
    annotation,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    lookup = {
        (row["num_agents"], row["num_producers"]): row for row in summary
    }
    values = np.asarray(
        [
            [float(lookup[(a, p)][key]) for p in PRODUCER_COUNTS]
            for a in AGENT_COUNTS
        ]
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    for row_index, num_agents in enumerate(AGENT_COUNTS):
        for col_index, num_producers in enumerate(PRODUCER_COUNTS):
            value = values[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                annotation(value),
                ha="center",
                va="center",
                color="white" if value > values.mean() else "black",
                fontsize=9,
            )
    ax.set_xticks(range(len(PRODUCER_COUNTS)), PRODUCER_COUNTS)
    ax.set_yticks(range(len(AGENT_COUNTS)), AGENT_COUNTS)
    ax.set_xlabel("Expert producer processes")
    ax.set_ylabel("Agents per environment")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(path: Path, summary: list[dict[str, Any]]) -> None:
    lookup = {
        (row["num_agents"], row["num_producers"]): row for row in summary
    }
    lines = [
        "# F2 Producer/Consumer Scaling",
        "",
        "Protocol: eight logical MAPF environments, batch size `4 × agents`, "
        "two optimizer updates per full frontier, 1,024,000 samples per row, "
        "three paired seeds. Validation, checkpoint I/O, initialization, "
        "teardown, and NVML sampling are outside the timed region.",
        "",
        "| Agents | Batch | P=1 | P=2 | P=4 | P=8 | P=8 / P=1 | P=8 / P=4 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for num_agents in AGENT_COUNTS:
        cells = [lookup[(num_agents, p)] for p in PRODUCER_COUNTS]
        rates = [float(cell["median_samples_s"]) / 1000.0 for cell in cells]
        lines.append(
            f"| {num_agents} | {num_agents * 4} | "
            f"{rates[0]:.1f}k | {rates[1]:.1f}k | {rates[2]:.1f}k | "
            f"{rates[3]:.1f}k | {rates[3] / rates[0]:.2f}× | "
            f"{rates[3] / rates[2]:.2f}× |"
        )

    lines.extend(
        [
            "",
            "## Bottleneck transition",
            "",
            "| Agents | P=1 wait share | P=2 wait share | P=4 wait share | "
            "P=8 wait share |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for num_agents in AGENT_COUNTS:
        shares = [
            float(lookup[(num_agents, p)]["median_consumer_wait_share"]) * 100.0
            for p in PRODUCER_COUNTS
        ]
        lines.append(
            f"| {num_agents} | {shares[0]:.1f}% | {shares[1]:.1f}% | "
            f"{shares[2]:.1f}% | {shares[3]:.1f}% |"
        )

    lines.extend(
        [
            "",
            "P=1 is producer-limited at every agent count. Increasing producer "
            "parallelism removes most consumer starvation. P=4 is the practical "
            "knee for 128 agents, while 256 and 512 agents retain a modest P=8 "
            "gain because the scaled batches raise useful GPU work per update.",
            "",
            "The reported `consumer_gpu_builder_s` measures the compact-state "
            "refresh interval only. Graph materialization and batch slicing are "
            "part of the remaining consumer path, so this report does not claim "
            "that all graph-building cost is negligible.",
            "",
            "Some 512-agent seeds trigger the configured first-stage LaCAM "
            "one-second timeout before the existing retry/fallback protocol. "
            "This is reflected in the paired-seed variation; medians, rather "
            "than the fastest run, are used for every paper-facing value.",
            "",
            "Environment note: the loaded NVIDIA kernel module and userspace "
            "NVML versions differ. CUDA compute remained operational; NVML "
            "telemetry was unavailable and was not included in timed regions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(*, roots: Sequence[Path], output_dir: Path) -> None:
    rows = _load_rows(roots)
    summary = _summarize(rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_rows = [
        {
            key: row.get(key)
            for key in (
                "num_agents",
                "train_batch_size",
                "num_producer_processes",
                "seed",
                "samples_processed",
                "total_wall_s",
                "samples_s",
                "optimizer_steps_s",
                "consumer_wait_s",
                "consumer_train_step_s",
                "consumer_dma_sync_s",
                "consumer_gpu_builder_s",
                "worker_ringbuffer_write_s_max",
                "peak_gpu_memory_bytes",
                "_path",
            )
        }
        for row in sorted(
            rows,
            key=lambda item: (
                int(item["num_agents"]),
                int(item["num_producer_processes"]),
                int(item["seed"]),
            ),
        )
    ]
    _write_csv(output_dir / "f2_raw_rows.csv", raw_rows)
    _write_csv(output_dir / "f2_summary.csv", summary)
    _render_scaling(output_dir / "f2_throughput_scaling.svg", summary)
    _render_heatmap(
        output_dir / "f2_throughput_heatmap.svg",
        summary,
        key="median_samples_s",
        title="Median training throughput (samples/s)",
        annotation=lambda value: f"{value / 1000.0:.1f}k",
    )
    _render_heatmap(
        output_dir / "f2_consumer_wait_heatmap.svg",
        summary,
        key="median_consumer_wait_share",
        title="Consumer wait share",
        annotation=lambda value: f"{value * 100.0:.1f}%",
    )
    _write_results(output_dir / "RESULTS.md", summary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_report(roots=args.root, output_dir=args.output_dir)
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
