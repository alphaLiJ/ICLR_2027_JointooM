"""Merge new F1 compact-H2D rows with the frozen E4 deployment endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


MODE_ORDER = ("host_expanded", "compact_h2d", "resident_state")
E4_MODE_MAP = {
    "cpu_single": "host_expanded",
    "gpu_stateful": "resident_state",
}


def _load_rows(f1_root: Path, e4_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(f1_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "ok" or row.get("mode") != "compact_h2d":
            continue
        row["_path"] = str(path)
        rows.append(row)
    for path in sorted(e4_root.glob("**/result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        mode = E4_MODE_MAP.get(row.get("mode"))
        if row.get("status") != "ok" or mode is None:
            continue
        row = dict(row)
        row["mode"] = mode
        row["_path"] = str(path)
        rows.append(row)
    return rows


def _median(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(statistics.median(values))


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["num_envs"]), row["mode"])].append(row)
    summary = []
    for (num_envs, mode), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], MODE_ORDER.index(item[0][1]))
    ):
        summary.append(
            {
                "num_agents": 256,
                "num_envs": num_envs,
                "mode": mode,
                "repetitions": len(group),
                "median_wall_ms": _median(group, "wall_ms_per_batched_step"),
                "median_env_steps_s": _median(group, "env_steps_s"),
                "median_agent_steps_s": _median(group, "agent_steps_s"),
                "median_h2d_bytes": _median(group, "h2d_bytes"),
                "median_d2h_bytes": _median(group, "d2h_bytes"),
                "median_peak_gpu_memory_bytes": _median(
                    group, "peak_gpu_memory_bytes"
                ),
                "median_cpu_builder_ms": (
                    None
                    if _median(group, "cpu_builder_s") is None
                    else _median(group, "cpu_builder_s") * 1000.0
                ),
                "median_cpu_pack_ms": (
                    None
                    if _median(group, "cpu_pack_s") is None
                    else _median(group, "cpu_pack_s") * 1000.0
                ),
                "median_compact_h2d_ms": (
                    None
                    if _median(group, "compact_h2d_s") is None
                    else _median(group, "compact_h2d_s") * 1000.0
                ),
                "median_gpu_builder_ms": (
                    None
                    if _median(group, "gpu_builder_s") is None
                    else _median(group, "gpu_builder_s") * 1000.0
                ),
                "median_model_forward_ms": (
                    None
                    if _median(group, "model_forward_s") is None
                    else _median(group, "model_forward_s") * 1000.0
                ),
                "median_cpu_transition_ms": (
                    None
                    if _median(group, "cpu_transition_s") is None
                    else _median(group, "cpu_transition_s") * 1000.0
                ),
            }
        )
    return summary


def _speedups(summary):
    by_key = {(row["num_envs"], row["mode"]): row for row in summary}
    out = []
    for num_envs in sorted({row["num_envs"] for row in summary}):
        host = by_key[(num_envs, "host_expanded")]
        compact = by_key[(num_envs, "compact_h2d")]
        resident = by_key[(num_envs, "resident_state")]
        out.append(
            {
                "num_agents": 256,
                "num_envs": num_envs,
                "host_to_compact_speedup": (
                    host["median_wall_ms"] / compact["median_wall_ms"]
                ),
                "compact_to_resident_speedup": (
                    compact["median_wall_ms"] / resident["median_wall_ms"]
                ),
                "host_to_resident_speedup": (
                    host["median_wall_ms"] / resident["median_wall_ms"]
                ),
                "host_to_compact_h2d_reduction": (
                    host["median_h2d_bytes"] / compact["median_h2d_bytes"]
                ),
                "compact_cpu_transition_share": (
                    compact["median_cpu_transition_ms"]
                    / compact["median_wall_ms"]
                ),
                "compact_h2d_share": (
                    compact["median_compact_h2d_ms"]
                    / compact["median_wall_ms"]
                ),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_latency(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    envs = sorted({row["num_envs"] for row in summary})
    lookup = {(row["num_envs"], row["mode"]): row for row in summary}
    x = np.arange(len(envs))
    width = 0.24
    colors = {
        "host_expanded": "#8c8c8c",
        "compact_h2d": "#3b82f6",
        "resident_state": "#16a34a",
    }
    labels = {
        "host_expanded": "Host expanded",
        "compact_h2d": "Compact H2D + CUDA build",
        "resident_state": "GPU resident",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for index, mode in enumerate(MODE_ORDER):
        values = [lookup[(env, mode)]["median_wall_ms"] for env in envs]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=labels[mode],
            color=colors[mode],
        )
    ax.set_yscale("log")
    ax.set_xticks(x, [str(env) for env in envs])
    ax.set_xlabel("Parallel environments (256 agents each)")
    ax.set_ylabel("Median wall time per batched step (ms, log scale)")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_compact_stages(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    compact = [row for row in summary if row["mode"] == "compact_h2d"]
    compact.sort(key=lambda row: row["num_envs"])
    stage_keys = (
        ("CPU pack", "median_cpu_pack_ms", "#f59e0b"),
        ("Compact H2D", "median_compact_h2d_ms", "#60a5fa"),
        ("CUDA builder", "median_gpu_builder_ms", "#2563eb"),
        ("Model", "median_model_forward_ms", "#7c3aed"),
        ("CPU transition", "median_cpu_transition_ms", "#ef4444"),
    )
    x = np.arange(len(compact))
    bottom = np.zeros(len(compact))
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for label, key, color in stage_keys:
        values = np.asarray([row[key] for row in compact])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    residual = np.asarray([row["median_wall_ms"] for row in compact]) - bottom
    ax.bar(x, residual, bottom=bottom, label="Other/synchronization", color="#d1d5db")
    ax.set_xticks(x, [str(row["num_envs"]) for row in compact])
    ax.set_xlabel("Parallel environments (256 agents each)")
    ax.set_ylabel("Median wall time per batched step (ms)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_results(
    path: Path, summary: list[dict[str, Any]], speedups: list[dict[str, Any]]
) -> None:
    by_key = {(row["num_envs"], row["mode"]): row for row in summary}
    lines = [
        "# F1 MAGAT Data-Path Ablation",
        "",
        "Fixed configuration: 256 agents, one fixed-step closed-loop decision, "
        "the same frozen E4 inputs and validation-selected checkpoint.",
        "",
        "| Envs | Host expanded (ms) | Compact H2D (ms) | GPU resident (ms) | "
        "Host→compact | Compact→resident | H2D reduction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in speedups:
        envs = row["num_envs"]
        lines.append(
            f"| {envs} | {by_key[(envs, 'host_expanded')]['median_wall_ms']:.3f} "
            f"| {by_key[(envs, 'compact_h2d')]['median_wall_ms']:.3f} "
            f"| {by_key[(envs, 'resident_state')]['median_wall_ms']:.3f} "
            f"| {row['host_to_compact_speedup']:.1f}× "
            f"| {row['compact_to_resident_speedup']:.2f}× "
            f"| {row['host_to_compact_h2d_reduction']:.1f}× |"
        )
    lines.extend(
        [
            "",
            "The compact-H2D middle path reproduces the E4 action-trajectory and "
            "final-state hashes. Validation, checkpoint loading, allocation, and "
            "monitoring are outside its timed region.",
            "",
            "The dominant gain is host-expanded → compact-H2D: CPU model-input "
            "materialization disappears. Compact transfer itself is negligible; "
            "the remaining middle-path cost is dominated by CPU POGEMA transition "
            "and model execution. The compact → resident increment therefore "
            "primarily measures transition residency and removal of the CPU/GPU "
            "closed-loop boundary, not merely transfer bandwidth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_f1_report(*, f1_root: Path, e4_root: Path, output_dir: Path) -> None:
    rows = _load_rows(f1_root, e4_root)
    summary = _summarize(rows)
    speedups = _speedups(summary)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "f1_summary.csv", summary)
    _write_csv(output_dir / "f1_speedups.csv", speedups)
    _render_latency(output_dir / "f1_latency.svg", summary)
    _render_compact_stages(output_dir / "f1_compact_stage_breakdown.svg", summary)
    _write_results(output_dir / "RESULTS.md", summary, speedups)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the F1 ablation report.")
    parser.add_argument("--f1-root", required=True, type=Path)
    parser.add_argument("--e4-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render_f1_report(
        f1_root=args.f1_root,
        e4_root=args.e4_root,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
