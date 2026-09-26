"""Aggregate the lightweight E4 closed-loop deployment measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MODE_ORDER = ("cpu_single", "cpu_mp_gpu_model", "gpu_stateful")
MODE_LABEL = {
    "cpu_single": "POGEMA single",
    "cpu_mp_gpu_model": "POGEMA 8-process",
    "gpu_stateful": "CUDA resident",
}
MODE_COLOR = {
    "cpu_single": "#dc2626",
    "cpu_mp_gpu_model": "#f59e0b",
    "gpu_stateful": "#2563eb",
}


def _median(rows: Iterable[dict[str, Any]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_results(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(root.glob("**/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        if result.get("mode") not in MODE_ORDER:
            continue
        result["_source"] = str(path.parent)
        results.append(result)
    return results


def _latency_svg(summary: list[dict[str, Any]]) -> str:
    width, height = 900, 510
    left, right, top, bottom = 95, 25, 55, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    envs = (16, 64, 256)
    group_w = plot_w / len(envs)
    bar_w = group_w / 4.1
    y_low, y_high = math.log10(0.02), math.log10(600.0)

    def ypos(seconds: float) -> float:
        value = math.log10(max(seconds, 10**y_low))
        return top + (y_high - value) / (y_high - y_low) * plot_h

    lookup = {
        (int(row["num_envs"]), row["mode"]): row for row in summary
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.tick{font-size:13px}.axis{font-size:15px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">E4 complete closed-loop latency (log scale)</text>',
    ]
    for seconds, label in (
        (0.1, "0.1 s"),
        (1.0, "1 s"),
        (10.0, "10 s"),
        (100.0, "100 s"),
    ):
        y = ypos(seconds)
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+5}" text-anchor="end">{label}</text>')
    for env_index, num_envs in enumerate(envs):
        center = left + (env_index + 0.5) * group_w
        for mode_index, mode in enumerate(MODE_ORDER):
            row = lookup[(num_envs, mode)]
            value = float(row["median_wall_ms"]) / 1000.0
            x = center + (mode_index - 1) * bar_w
            y = ypos(value)
            parts.append(
                f'<rect x="{x-bar_w*0.42}" y="{y}" width="{bar_w*0.84}" '
                f'height="{top+plot_h-y}" fill="{MODE_COLOR[mode]}" rx="2"/>'
            )
        parts.append(
            f'<text class="axis" x="{center}" y="{height-bottom+28}" text-anchor="middle">E={num_envs}</text>'
        )
    legend_x = 180
    for index, mode in enumerate(MODE_ORDER):
        x = legend_x + index * 205
        parts.append(f'<rect x="{x}" y="{height-30}" width="14" height="14" fill="{MODE_COLOR[mode]}"/>')
        parts.append(f'<text class="tick" x="{x+20}" y="{height-18}">{MODE_LABEL[mode]}</text>')
    parts.append(
        f'<text class="axis" transform="translate(24 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Wall time per batched step</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _speedup_svg(speedups: list[dict[str, Any]]) -> str:
    width, height = 820, 450
    left, right, top, bottom = 90, 25, 50, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(float(row["speedup_vs_cuda"]) for row in speedups) * 1.1
    envs = (16, 64, 256)
    group_w = plot_w / len(envs)
    bar_w = group_w / 3.1
    colors = {"cpu_single": "#dc2626", "cpu_mp_gpu_model": "#f59e0b"}
    lookup = {(int(row["num_envs"]), row["baseline_mode"]): row for row in speedups}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.tick{font-size:13px}.axis{font-size:15px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">CUDA-resident end-to-end speedup</text>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + (1.0 - fraction) * plot_h
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+5}" text-anchor="end">{maximum*fraction:.0f}×</text>')
    for env_index, num_envs in enumerate(envs):
        center = left + (env_index + 0.5) * group_w
        for mode_index, mode in enumerate(("cpu_single", "cpu_mp_gpu_model")):
            value = float(lookup[(num_envs, mode)]["speedup_vs_cuda"])
            height_now = value / maximum * plot_h
            x = center + (mode_index - 0.5) * bar_w
            parts.append(
                f'<rect x="{x-bar_w*0.42}" y="{top+plot_h-height_now}" '
                f'width="{bar_w*0.84}" height="{height_now}" fill="{colors[mode]}" rx="2"/>'
            )
            parts.append(
                f'<text class="tick" x="{x}" y="{top+plot_h-height_now-6}" text-anchor="middle">{value:.0f}×</text>'
            )
        parts.append(f'<text class="axis" x="{center}" y="{height-bottom+28}" text-anchor="middle">E={num_envs}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(root: Path, output_dir: Path) -> None:
    results = _load_results(root)
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(int(result["num_envs"]), result["mode"])].append(result)
    expected = {(envs, mode) for envs in (16, 64, 256) for mode in MODE_ORDER}
    missing = sorted(expected - set(groups))
    if missing:
        raise ValueError(f"incomplete E4 result matrix: {missing}")
    for key in expected:
        if len(groups[key]) != 3:
            raise ValueError(f"E4 requires three repetitions for {key}, got {len(groups[key])}")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    for result in results:
        raw_rows.append(
            {
                "num_agents": result["num_agents"],
                "num_envs": result["num_envs"],
                "mode": result["mode"],
                "wall_ms": result["wall_ms_per_batched_step"],
                "env_steps_s": result["env_steps_s"],
                "agent_steps_s": result["agent_steps_s"],
                "cpu_utilization_pct": result["cpu_process_utilization_pct"],
                "gpu_utilization_pct": result["gpu_utilization_pct"],
                "h2d_bytes": result["h2d_bytes"],
                "d2h_bytes": result["d2h_bytes"],
                "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
                "peak_nvml_gpu_memory_bytes": result["peak_nvml_gpu_memory_bytes"],
                "peak_host_memory_bytes": result["peak_host_memory_bytes"],
                "worker_peak_host_memory_sum_bytes": result.get(
                    "worker_peak_host_memory_sum_bytes", 0
                ),
                "worker_peak_host_memory_max_bytes": max(
                    result.get("worker_peak_host_memory_bytes", [0])
                ),
                "action_sha256": result["trajectory_action_sha256"],
                "final_state_sha256": result["final_state_sha256"],
                "source": result["_source"],
            }
        )
    _write_csv(output_dir / "e4_deployment_rows.csv", raw_rows)

    summary = []
    for num_envs in (16, 64, 256):
        for mode in MODE_ORDER:
            rows = groups[(num_envs, mode)]
            summary.append(
                {
                    "num_agents": 256,
                    "num_envs": num_envs,
                    "mode": mode,
                    "repetitions": len(rows),
                    "median_wall_ms": _median(rows, "wall_ms_per_batched_step"),
                    "min_wall_ms": min(float(row["wall_ms_per_batched_step"]) for row in rows),
                    "max_wall_ms": max(float(row["wall_ms_per_batched_step"]) for row in rows),
                    "median_env_steps_s": _median(rows, "env_steps_s"),
                    "median_agent_steps_s": _median(rows, "agent_steps_s"),
                    "median_cpu_utilization_pct": _median(rows, "cpu_process_utilization_pct"),
                    "median_gpu_utilization_pct": _median(rows, "gpu_utilization_pct"),
                    "median_h2d_bytes": _median(rows, "h2d_bytes"),
                    "median_d2h_bytes": _median(rows, "d2h_bytes"),
                    "median_peak_gpu_memory_bytes": _median(rows, "peak_gpu_memory_bytes"),
                    "median_peak_host_memory_bytes": _median(rows, "peak_host_memory_bytes"),
                    "median_worker_peak_host_memory_sum_bytes": statistics.median(
                        float(row.get("worker_peak_host_memory_sum_bytes", 0))
                        for row in rows
                    ),
                }
            )
    _write_csv(output_dir / "e4_deployment_summary.csv", summary)

    summary_lookup = {
        (int(row["num_envs"]), row["mode"]): row for row in summary
    }
    speedups = []
    for num_envs in (16, 64, 256):
        cuda_ms = float(summary_lookup[(num_envs, "gpu_stateful")]["median_wall_ms"])
        for baseline in ("cpu_single", "cpu_mp_gpu_model"):
            baseline_ms = float(summary_lookup[(num_envs, baseline)]["median_wall_ms"])
            speedups.append(
                {
                    "num_agents": 256,
                    "num_envs": num_envs,
                    "baseline_mode": baseline,
                    "cuda_mode": "gpu_stateful",
                    "speedup_vs_cuda": baseline_ms / cuda_ms,
                }
            )
    _write_csv(output_dir / "e4_speedups.csv", speedups)

    parity = {}
    for num_envs in (16, 64, 256):
        rows = [
            row
            for mode in MODE_ORDER
            for row in groups[(num_envs, mode)]
        ]
        action_hashes = sorted({row["trajectory_action_sha256"] for row in rows})
        state_hashes = sorted({row["final_state_sha256"] for row in rows})
        parity[str(num_envs)] = {
            "exact_action_parity": len(action_hashes) == 1,
            "exact_final_state_parity": len(state_hashes) == 1,
            "action_hashes": action_hashes,
            "final_state_hashes": state_hashes,
        }
    (output_dir / "e4_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "e4_latency.svg").write_text(_latency_svg(summary), encoding="utf-8")
    (output_dir / "e4_speedup.svg").write_text(_speedup_svg(speedups), encoding="utf-8")

    lines = [
        "# E4: complete MAGAT closed-loop deployment",
        "",
        "Configuration: standard MAPF, 256 agents, E=16/64/256, validation-selected MAGAT checkpoint at optimizer step 95,000. Each point is the median of three independent fresh-process repetitions. The timed region is one complete fixed closed-loop step; checkpoint loading, one-environment parity, and warm-up are outside the timed region.",
        "",
        "![Complete closed-loop latency](e4_latency.svg)",
        "",
        "## End-to-end performance",
        "",
        "| Envs | POGEMA single | POGEMA 8-process | CUDA resident | CUDA speedup vs single | CUDA speedup vs 8-process |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    speed_lookup = {
        (int(row["num_envs"]), row["baseline_mode"]): row["speedup_vs_cuda"]
        for row in speedups
    }
    for num_envs in (16, 64, 256):
        single = summary_lookup[(num_envs, "cpu_single")]
        multi = summary_lookup[(num_envs, "cpu_mp_gpu_model")]
        cuda = summary_lookup[(num_envs, "gpu_stateful")]
        lines.append(
            f"| {num_envs} | {single['median_wall_ms']/1000:.3f} s "
            f"({single['median_env_steps_s']:.2f} env-step/s) | "
            f"{multi['median_wall_ms']/1000:.3f} s "
            f"({multi['median_env_steps_s']:.2f} env-step/s) | "
            f"{cuda['median_wall_ms']:.3f} ms "
            f"({cuda['median_env_steps_s']:.1f} env-step/s) | "
            f"{speed_lookup[(num_envs, 'cpu_single')]:.1f}× | "
            f"{speed_lookup[(num_envs, 'cpu_mp_gpu_model')]:.1f}× |"
        )
    lines.extend(
        [
            "",
            "![End-to-end speedup](e4_speedup.svg)",
            "",
            "## Exact one-step parity",
            "",
            "| Envs | Actions identical across all paths/repetitions | Final state identical |",
            "|---:|:---:|:---:|",
        ]
    )
    for num_envs in (16, 64, 256):
        row = parity[str(num_envs)]
        lines.append(
            f"| {num_envs} | {'yes' if row['exact_action_parity'] else 'no'} | "
            f"{'yes' if row['exact_final_state_parity'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Transfer and memory medians",
            "",
            "| Envs | Path | H2D/step | D2H/step | PyTorch peak GPU | Parent peak host |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for num_envs in (16, 64, 256):
        for mode in MODE_ORDER:
            row = summary_lookup[(num_envs, mode)]
            lines.append(
                f"| {num_envs} | {MODE_LABEL[mode]} | "
                f"{row['median_h2d_bytes']/2**20:.3f} MiB | "
                f"{row['median_d2h_bytes']/2**20:.3f} MiB | "
                f"{row['median_peak_gpu_memory_bytes']/2**30:.2f} GiB | "
                f"{row['median_peak_host_memory_bytes']/2**30:.2f} GiB |"
            )
    lines.extend(
        [
            "",
            "## Interpretation and limits",
            "",
            "- This experiment measures deployment, not only the transition kernel: observation/graph construction, MAGAT forward, argmax/action update, and environment transition are all inside the timed region.",
            "- The CUDA path keeps state and model inputs resident. Its timed loop transfers no bulk H2D data and reads one device control byte per measured step; post-loop diagnostic state readback is excluded.",
            "- CPU paths include CPU observation/graph construction, H2D model inputs, GPU inference, D2H actions, and POGEMA transition.",
            "- One-step CSR/SoC/makespan values are parity diagnostics only. Final policy-quality claims must come from the longer E1/validation evaluation.",
            "- The CUDA measurement includes initial energy-map preparation in the measured step and is therefore conservative for steady-state multi-step deployment.",
            "- To reduce experiment turnaround, the three POGEMA-single repetitions for E16/E64 ran concurrently while each process remained pinned to one distinct CPU core; E256 ran serially because one full model batch uses about 7.1 GiB of allocated GPU memory. Shared-memory-bandwidth contention can make the E16/E64 CPU baseline slightly conservative.",
            "- Per-process PyTorch peak memory is used in the CSV. NVML device-wide peaks from concurrently scheduled CPU baselines should not be interpreted as per-process memory.",
            "",
            f"Raw artifact root: `{root}`",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render(args.root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
