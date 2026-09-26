"""Render the dependency-free report for the minimal simulator scaling scan."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence


BACKENDS = (
    ("cuda_stateful", "CUDA stateful", "#2563eb"),
    ("jax_gpu", "JAX GPU", "#dc2626"),
    ("pogema_8", "POGEMA 8-core", "#16a34a"),
    ("pogema_single", "POGEMA single-process", "#9333ea"),
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _fmt_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}k"
    return f"{value:.2f}"


def _render_svg(rows: list[dict[str, str]]) -> str:
    width, height = 960, 600
    left, right, top, bottom = 105, 35, 45, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    envs = sorted({int(row["num_envs"]) for row in rows})
    x_min, x_max = math.log2(min(envs)), math.log2(max(envs))
    y_min, y_max = 3.0, 7.0

    def x_pos(value: int) -> float:
        return left + (math.log2(value) - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        clipped = min(max(math.log10(value), y_min), y_max)
        return top + (y_max - clipped) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.tick{font-size:14px}.label{font-size:17px;font-weight:600}.legend{font-size:14px}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#111827;stroke-width:1.5}</style>',
    ]
    for exponent in range(3, 8):
        y = y_pos(10**exponent)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}"/>')
        parts.append(f'<text class="tick" x="{left-12}" y="{y+5:.2f}" text-anchor="end">10^{exponent}</text>')
    for env in envs:
        x = x_pos(env)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        parts.append(f'<text class="tick" x="{x:.2f}" y="{height-bottom+28}" text-anchor="middle">{env}</text>')
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text class="label" x="{left+plot_w/2:.2f}" y="{height-24}" text-anchor="middle">Number of environments (log2)</text>',
            f'<text class="label" x="{left}" y="27">Throughput: environment steps / second (log10)</text>',
        ]
    )

    by_backend = {
        backend: sorted(
            (row for row in rows if row["backend"] == backend),
            key=lambda row: int(row["num_envs"]),
        )
        for backend, _, _ in BACKENDS
    }
    legend_x, legend_y = 645, 65
    for index, (backend, label, color) in enumerate(BACKENDS):
        backend_rows = by_backend[backend]
        points = " ".join(
            f'{x_pos(int(row["num_envs"])):.2f},{y_pos(float(row["median_env_steps_s"])):.2f}'
            for row in backend_rows
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in backend_rows:
            x = x_pos(int(row["num_envs"]))
            y = y_pos(float(row["median_env_steps_s"]))
            y_low = y_pos(float(row["min_env_steps_s"]))
            y_high = y_pos(float(row["max_env_steps_s"]))
            parts.append(f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        ly = legend_y + index * 25
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+30}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{legend_x+15}" cy="{ly}" r="4" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{legend_x+40}" y="{ly+5}">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def _render_markdown(
    rows: list[dict[str, str]], speedups: list[dict[str, str]]
) -> str:
    lookup = {
        (row["backend"], int(row["num_envs"])): float(row["median_env_steps_s"])
        for row in rows
    }
    envs = sorted({env for _, env in lookup})
    speedup_lookup = {
        (row["baseline"], int(row["num_envs"])): float(row["speedup_x"])
        for row in speedups
    }
    lines = [
        "# Minimal simulator scaling results",
        "",
        "Fixed workload: 256 agents per environment, 256 transition steps, five measured repetitions per point. Values are median nominal environment-steps/s.",
        "",
        "![Simulator scaling](scaling.svg)",
        "",
        "## Throughput",
        "",
        "| Environments | CUDA stateful | JAX GPU | POGEMA 8-core | POGEMA single-process |",
        "|---:|---:|---:|---:|---:|",
    ]
    for env in envs:
        values = [_fmt_rate(lookup[(backend, env)]) for backend, _, _ in BACKENDS]
        lines.append(f"| {env} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## CUDA speedup",
            "",
            "| Environments | vs JAX GPU | vs POGEMA 8-core | vs POGEMA single-process |",
            "|---:|---:|---:|---:|",
        ]
    )
    for env in envs:
        lines.append(
            f"| {env} | {speedup_lookup[('jax_gpu', env)]:.2f}x | "
            f"{speedup_lookup[('pogema_8', env)]:.2f}x | "
            f"{speedup_lookup[('pogema_single', env)]:.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and provenance",
            "",
            "- CUDA throughput grows from 0.416M to 6.960M environment-steps/s as the number of environments increases from 16 to 1024.",
            "- At 1024 environments, CUDA is 13.40x faster than JAX GPU, 451.50x faster than 8-core POGEMA, and 3562.80x faster than single-process POGEMA.",
            "- The scan measures transition throughput only. Correctness/parity checks, dataset construction, policy inference, and host-side trajectory validation are outside the timed region.",
            "- Points for 16, 64, and 256 environments reuse retained A2 measurements; the 1024 CUDA/JAX/POGEMA-8 points use the minimal throughput runner. The `protocol` and `source` columns in `combined_scaling.csv` preserve this distinction.",
            "- Error bars in the SVG show the minimum and maximum of five repetitions, not confidence intervals.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", required=True, type=Path)
    parser.add_argument("--speedups", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = _read_rows(args.combined)
    speedups = _read_rows(args.speedups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scaling.svg").write_text(_render_svg(rows), encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(
        _render_markdown(rows, speedups), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
