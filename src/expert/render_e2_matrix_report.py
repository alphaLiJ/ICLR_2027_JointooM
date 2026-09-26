"""Merge retained A2 rows with new E2 matrix rows and render the E2 report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


AGENTS = (64, 128, 256, 512)
ENVS = (16, 64, 256, 1024)
BACKENDS = ("cuda", "jax")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0])
        fieldnames.extend(
            key for row in rows for key in row if key not in fieldnames
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _formal_memory(source: str) -> tuple[int | None, str]:
    path = Path(source)
    directory = path if path.is_dir() else path.parent
    health_path = directory / "health.json"
    if not health_path.exists():
        return None, "unavailable"
    payload = json.loads(health_path.read_text(encoding="utf-8"))
    values = [
        (event.get("gpu") or {}).get("memory_used_bytes")
        for event in payload.get("events", [])
    ]
    observed = [int(value) for value in values if value is not None]
    return (max(observed), "formal_nvml_device_used") if observed else (None, "unavailable")


def _new_results(root: Path) -> tuple[dict[tuple[str, int, int], dict], list[dict]]:
    successful: dict[tuple[str, int, int], dict] = {}
    failures: list[dict] = []
    for path in sorted(root.glob("*/*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        backend = str(payload["backend"])
        key = (backend, int(payload["num_agents"]), int(payload["num_envs"]))
        payload["_source"] = str(path.parent)
        if payload.get("status"):
            failures.append(payload)
        elif "median_env_steps_s" in payload:
            successful[key] = payload
    return successful, failures


def collect_rows(
    *,
    agent_scan_csv: Path,
    env_scan_csv: Path,
    new_runs_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix: dict[tuple[str, int, int], dict[str, Any]] = {}

    for row in _read_csv(agent_scan_csv):
        if row["axis"] != "agent" or row["backend"] not in BACKENDS:
            continue
        agents, envs = int(row["num_agents"]), int(row["num_envs"])
        if agents not in AGENTS or envs != 16:
            continue
        memory, memory_source = _formal_memory(row["source"])
        matrix[(row["backend"], agents, envs)] = {
            "backend": row["backend"],
            "num_agents": agents,
            "num_envs": envs,
            "status": "ok",
            "median_env_steps_s": float(row["median_env_steps_s"]),
            "min_env_steps_s": float(row["min_env_steps_s"]),
            "max_env_steps_s": float(row["max_env_steps_s"]),
            "peak_gpu_memory_bytes": memory,
            "memory_metric": memory_source,
            "protocol": "retained_a2_formal",
            "source": row["source"],
        }

    backend_map = {"cuda_stateful": "cuda", "jax_gpu": "jax"}
    for row in _read_csv(env_scan_csv):
        backend = backend_map.get(row["backend"])
        if backend is None:
            continue
        agents, envs = int(row["num_agents"]), int(row["num_envs"])
        if agents != 256 or envs not in ENVS:
            continue
        memory, memory_source = _formal_memory(row["source"])
        matrix[(backend, agents, envs)] = {
            "backend": backend,
            "num_agents": agents,
            "num_envs": envs,
            "status": "ok",
            "median_env_steps_s": float(row["median_env_steps_s"]),
            "min_env_steps_s": float(row["min_env_steps_s"]),
            "max_env_steps_s": float(row["max_env_steps_s"]),
            "peak_gpu_memory_bytes": memory,
            "memory_metric": memory_source,
            "protocol": row["protocol"],
            "source": row["source"],
        }

    new_rows, failures = _new_results(new_runs_root)
    for key, payload in new_rows.items():
        memory = payload.get("memory") or {}
        new_row = {
            "backend": key[0],
            "num_agents": key[1],
            "num_envs": key[2],
            "status": "ok",
            "median_env_steps_s": float(payload["median_env_steps_s"]),
            "min_env_steps_s": float(payload["min_env_steps_s"]),
            "max_env_steps_s": float(payload["max_env_steps_s"]),
            "peak_gpu_memory_bytes": memory.get("max_nvml_process_memory_bytes"),
            "memory_metric": "minimal_nvml_process_used",
            "protocol": "minimal_e2_transition_only",
            "source": payload["_source"],
        }
        if key in matrix:
            # Retain existing throughput, but fill its previously missing memory
            # from the isolated E2 row (currently A256/E1024).
            if matrix[key]["peak_gpu_memory_bytes"] is None:
                matrix[key]["peak_gpu_memory_bytes"] = new_row["peak_gpu_memory_bytes"]
                matrix[key]["memory_metric"] = new_row["memory_metric"]
                matrix[key]["memory_source"] = new_row["source"]
        else:
            matrix[key] = new_row

    failure_rows: list[dict[str, Any]] = []
    for payload in failures:
        if payload.get("status") != "capacity_failure":
            continue
        key = (str(payload["backend"]), int(payload["num_agents"]), int(payload["num_envs"]))
        if key not in matrix:
            matrix[key] = {
                "backend": key[0],
                "num_agents": key[1],
                "num_envs": key[2],
                "status": "capacity_failure",
                "median_env_steps_s": None,
                "min_env_steps_s": None,
                "max_env_steps_s": None,
                "peak_gpu_memory_bytes": None,
                "memory_metric": "unavailable_at_allocation_failure",
                "protocol": "minimal_e2_transition_only",
                "source": payload["_source"],
            }
        failure_rows.append(
            {
                "backend": key[0],
                "num_agents": key[1],
                "num_envs": key[2],
                "status": payload["status"],
                "exception_type": payload["exception_type"],
                "exception_message": payload["exception_message"],
                "source": payload["_source"],
            }
        )

    expected = {(backend, agents, envs) for backend in BACKENDS for agents in AGENTS for envs in ENVS}
    missing = sorted(expected - set(matrix))
    if missing:
        raise ValueError(f"E2 matrix is incomplete: {missing}")
    for row in matrix.values():
        if row["status"] != "ok":
            row["median_trajectory_duration_s"] = None
            row["min_trajectory_duration_s"] = None
            row["max_trajectory_duration_s"] = None
            continue
        nominal_steps = int(row["num_envs"]) * 256
        row["median_trajectory_duration_s"] = nominal_steps / float(
            row["median_env_steps_s"]
        )
        row["min_trajectory_duration_s"] = nominal_steps / float(
            row["max_env_steps_s"]
        )
        row["max_trajectory_duration_s"] = nominal_steps / float(
            row["min_env_steps_s"]
        )
    return [matrix[key] for key in sorted(matrix)], failure_rows


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "OOM"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    return f"{value / 1_000:.2f}k"


def _fmt_memory(value: int | None) -> str:
    return "OOM" if value is None else f"{value / 2**30:.2f} GiB"


def _color(value: float, low: float, high: float) -> str:
    ratio = 0.5 if high <= low else (value - low) / (high - low)
    ratio = min(1.0, max(0.0, ratio))
    red = int(245 - 190 * ratio)
    green = int(247 - 70 * ratio)
    blue = int(255 - 25 * ratio)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _heatmap_svg(title: str, values: dict[tuple[int, int], float | None], formatter) -> str:
    width, height = 760, 470
    left, top, cell_w, cell_h = 120, 90, 145, 72
    finite = [float(value) for value in values.values() if value is not None]
    low, high = min(finite), max(finite)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.axis{font-size:15px;font-weight:600}.cell{font-size:15px}</style>',
        f'<text class="title" x="{width/2}" y="32" text-anchor="middle">{title}</text>',
        f'<text class="axis" x="{left + 2*cell_w}" y="60" text-anchor="middle">Number of environments</text>',
    ]
    for column, envs in enumerate(ENVS):
        parts.append(f'<text class="axis" x="{left+(column+0.5)*cell_w}" y="82" text-anchor="middle">{envs}</text>')
    for row_index, agents in enumerate(AGENTS):
        y = top + row_index * cell_h
        parts.append(f'<text class="axis" x="{left-15}" y="{y+cell_h/2+5}" text-anchor="end">{agents}</text>')
        for column, envs in enumerate(ENVS):
            x = left + column * cell_w
            value = values[(agents, envs)]
            fill = "#fee2e2" if value is None else _color(float(value), low, high)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" fill="{fill}"/>')
            parts.append(f'<text class="cell" x="{x+(cell_w-3)/2}" y="{y+cell_h/2+5}" text-anchor="middle">{formatter(value)}</text>')
    parts.append(f'<text class="axis" transform="translate(25 {top+2*cell_h}) rotate(-90)" text-anchor="middle">Number of agents</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_report(rows: list[dict[str, Any]], failures: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "e2_matrix.csv", rows)
    if failures:
        _write_csv(output_dir / "e2_capacity_failures.csv", failures)

    lookup = {(row["backend"], row["num_agents"], row["num_envs"]): row for row in rows}
    speedups = []
    for agents in AGENTS:
        for envs in ENVS:
            cuda = lookup[("cuda", agents, envs)]["median_env_steps_s"]
            jax = lookup[("jax", agents, envs)]["median_env_steps_s"]
            speedups.append({
                "num_agents": agents,
                "num_envs": envs,
                "cuda_vs_jax_x": None if cuda is None else float(cuda) / float(jax),
            })
    _write_csv(output_dir / "e2_speedups.csv", speedups)

    for backend in BACKENDS:
        throughput = {
            (agents, envs): lookup[(backend, agents, envs)]["median_env_steps_s"]
            for agents in AGENTS for envs in ENVS
        }
        memory = {
            (agents, envs): lookup[(backend, agents, envs)]["peak_gpu_memory_bytes"]
            for agents in AGENTS for envs in ENVS
        }
        (output_dir / f"e2_{backend}_throughput.svg").write_text(
            _heatmap_svg(f"{backend.upper()} throughput (environment-steps/s)", throughput, _fmt_rate),
            encoding="utf-8",
        )
        (output_dir / f"e2_{backend}_memory.svg").write_text(
            _heatmap_svg(f"{backend.upper()} observed peak GPU memory", memory, _fmt_memory),
            encoding="utf-8",
        )
    speedup_values = {(row["num_agents"], row["num_envs"]): row["cuda_vs_jax_x"] for row in speedups}
    (output_dir / "e2_cuda_vs_jax.svg").write_text(
        _heatmap_svg("CUDA / JAX throughput speedup", speedup_values, lambda value: "OOM" if value is None else f"{value:.2f}x"),
        encoding="utf-8",
    )

    lines = [
        "# E2: two-dimensional simulator scaling",
        "",
        "Workload: standard MAPF, random maps at density 0.2, 256 transition steps, five measured repetitions. The timed region contains only the fixed-trajectory transition loop; input construction, warm-up, synchronization setup, correctness replay, and memory sampling are outside it.",
        "",
        "## Throughput",
        "",
    ]
    for backend in BACKENDS:
        lines.extend([
            f"### {backend.upper()}", "",
            "| Agents \\ Environments | 16 | 64 | 256 | 1024 |",
            "|---:|---:|---:|---:|---:|",
        ])
        for agents in AGENTS:
            cells = [_fmt_rate(lookup[(backend, agents, envs)]["median_env_steps_s"]) for envs in ENVS]
            lines.append(f"| {agents} | " + " | ".join(cells) + " |")
        lines.extend(["", f"![{backend} throughput](e2_{backend}_throughput.svg)", ""])

    lines.extend([
        "## CUDA speedup over JAX", "",
        "| Agents \\ Environments | 16 | 64 | 256 | 1024 |",
        "|---:|---:|---:|---:|---:|",
    ])
    speedup_lookup = {(row["num_agents"], row["num_envs"]): row["cuda_vs_jax_x"] for row in speedups}
    for agents in AGENTS:
        cells = ["OOM" if speedup_lookup[(agents, envs)] is None else f"{speedup_lookup[(agents, envs)]:.2f}x" for envs in ENVS]
        lines.append(f"| {agents} | " + " | ".join(cells) + " |")
    lines.extend(["", "![CUDA versus JAX](e2_cuda_vs_jax.svg)", "", "## Observed peak GPU memory", ""])
    for backend in BACKENDS:
        lines.extend([
            f"### {backend.upper()}", "",
            "| Agents \\ Environments | 16 | 64 | 256 | 1024 |",
            "|---:|---:|---:|---:|---:|",
        ])
        for agents in AGENTS:
            cells = [_fmt_memory(lookup[(backend, agents, envs)]["peak_gpu_memory_bytes"]) for envs in ENVS]
            lines.append(f"| {agents} | " + " | ".join(cells) + " |")
        lines.extend(["", f"![{backend} memory](e2_{backend}_memory.svg)", ""])

    lines.extend([
        "## Interpretation", "",
        "- Increasing the number of environments substantially improves CUDA utilization for 64--256 agents. At 1024 environments CUDA reaches 41.56M, 21.86M, and 6.96M environment-steps/s for 64, 128, and 256 agents respectively.",
        "- The interaction is not monotonic in agent count. At 512 agents the current stateful implementation allocates per-agent energy maps and an `A²` edge-capacity buffer; throughput falls and CUDA reaches the 16 GB capacity boundary at 1024 environments.",
        "- CUDA is slightly slower than JAX at A512/E16, but becomes 1.82x faster at E64 and 7.89x faster at E256. This supports the claim that batching environments is essential to expose the CUDA design's advantage.",
        "- `A512/E1024` is reported as a capacity boundary, not omitted or imputed. JAX completes that point at about 116.82k environment-steps/s with 1.53 GiB observed process memory.",
        "- E16 and the retained A256 environment scan reuse prior formal A2 throughput. Newly missing cells use the lightweight E2 runner. Exact protocol and source are retained per row in `e2_matrix.csv`.",
        "- Memory values use the closest available NVML observation. Retained formal rows record whole-device used memory; new isolated rows record this benchmark process. They are suitable for scale/capacity interpretation but should not be presented as byte-identical allocator accounting.",
        "- The simulator currently owns model-oriented state (especially energy maps and graph capacity) even though E2 times transition-only execution. The memory heatmap therefore describes the present end-to-end stateful object, not a hypothetical transition-only allocator.",
        "",
    ])
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the E2 matrix report.")
    parser.add_argument("--agent-scan-csv", required=True, type=Path)
    parser.add_argument("--env-scan-csv", required=True, type=Path)
    parser.add_argument("--new-runs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    rows, failures = collect_rows(
        agent_scan_csv=args.agent_scan_csv,
        env_scan_csv=args.env_scan_csv,
        new_runs_root=args.new_runs_root,
    )
    render_report(rows, failures, args.output_dir)
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
