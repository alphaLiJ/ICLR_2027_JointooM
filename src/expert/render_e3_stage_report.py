"""Render the lightweight E3 resident-stage decomposition report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence


PLOT_STAGES = (
    "transition",
    "derived_state_steady",
    "node_construction",
    "graph_finalization",
    "edge_materialization",
    "model_forward",
    "argmax_action_update",
    "closed_loop",
)
STAGE_LABELS = {
    "transition": "Transition",
    "derived_state_steady": "Derived",
    "node_construction": "Nodes",
    "graph_finalization": "Graph",
    "edge_materialization": "Model-ready",
    "model_forward": "Model",
    "argmax_action_update": "Argmax/update",
    "closed_loop": "Closed loop",
}
COLORS = {16: "#2563eb", 64: "#16a34a", 256: "#dc2626"}


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"E3 result is not successful: {path}")
    return payload


def _stage_lookup(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["stage"]: row for row in result["stages"]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    fields.extend(key for row in rows for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _stage_latency_svg(results: list[dict[str, Any]]) -> str:
    width, height = 1160, 620
    left, right, top, bottom = 95, 25, 60, 125
    plot_w, plot_h = width - left - right, height - top - bottom
    group_w = plot_w / len(PLOT_STAGES)
    bar_w = group_w / 4.2
    y_low, y_high = -2.0, math.log10(250.0)

    def ypos(value: float) -> float:
        log_value = math.log10(max(value, 10**y_low))
        return top + (y_high - log_value) / (y_high - y_low) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:21px;font-weight:600}.tick{font-size:13px}.label{font-size:15px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="30" text-anchor="middle">E3 resident-stage wall latency (log scale)</text>',
    ]
    for exponent in (-2, -1, 0, 1, 2):
        y = ypos(10**exponent)
        parts.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width-right}" y2="{y}"/>')
        parts.append(f'<text class="tick" x="{left-10}" y="{y+5}" text-anchor="end">{10**exponent:g} ms</text>')
    for stage_index, stage in enumerate(PLOT_STAGES):
        center = left + (stage_index + 0.5) * group_w
        for result_index, result in enumerate(results):
            envs = int(result["num_envs"])
            value = float(_stage_lookup(result)[stage]["median_wall_ms_per_step"])
            x = center + (result_index - 1) * bar_w
            y = ypos(value)
            parts.append(
                f'<rect x="{x-bar_w*0.42}" y="{y}" width="{bar_w*0.84}" '
                f'height="{top+plot_h-y}" fill="{COLORS[envs]}" rx="2"/>'
            )
        parts.append(
            f'<text class="tick" transform="translate({center+5} {height-bottom+18}) rotate(42)" '
            f'text-anchor="start">{STAGE_LABELS[stage]}</text>'
        )
    legend_x = width - 275
    for index, result in enumerate(results):
        envs = int(result["num_envs"])
        x = legend_x + index * 82
        parts.append(f'<rect x="{x}" y="43" width="14" height="14" fill="{COLORS[envs]}"/>')
        parts.append(f'<text class="tick" x="{x+20}" y="55">E={envs}</text>')
    parts.append(
        f'<text class="label" transform="translate(25 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Wall time per environment step</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _share_svg(results: list[dict[str, Any]]) -> str:
    width, height = 920, 350
    left, right, top = 115, 40, 70
    bar_w = width - left - right
    components = (
        ("model_forward", "Model", "#dc2626"),
        ("node_construction", "Nodes", "#f59e0b"),
        ("edge_materialization", "Model-ready", "#16a34a"),
        ("transition", "Transition", "#2563eb"),
        ("derived_state_steady", "Derived", "#8b5cf6"),
        ("graph_finalization", "Graph", "#06b6d4"),
        ("argmax_action_update", "Argmax", "#6b7280"),
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}.title{font-size:20px;font-weight:600}.label{font-size:14px}.axis{font-size:15px;font-weight:600}</style>',
        f'<text class="title" x="{width/2}" y="30" text-anchor="middle">Exclusive-stage time share (diagnostic normalization)</text>',
    ]
    for row_index, result in enumerate(results):
        lookup = _stage_lookup(result)
        values = [float(lookup[key]["median_wall_ms_per_step"]) for key, _, _ in components]
        total = sum(values)
        y = top + row_index * 62
        x = left
        parts.append(f'<text class="axis" x="{left-15}" y="{y+25}" text-anchor="end">E={result["num_envs"]}</text>')
        for value, (_, _, color) in zip(values, components):
            width_now = bar_w * value / total
            parts.append(f'<rect x="{x}" y="{y}" width="{width_now}" height="36" fill="{color}"/>')
            x += width_now
    legend_y = 275
    x = left
    for _, label, color in components:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="13" height="13" fill="{color}"/>')
        parts.append(f'<text class="label" x="{x+18}" y="{legend_y+12}">{label}</text>')
        x += 102
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render(
    *,
    result_paths: list[Path],
    microbatch_result_path: Path | None,
    fullbatch_capacity_paths: list[Path],
    output_dir: Path,
) -> None:
    results = sorted((_load(path) for path in result_paths), key=lambda row: row["num_envs"])
    if [result["num_envs"] for result in results] != [16, 64, 256]:
        raise ValueError("E3 report requires E16, E64, and E256")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for result, source in zip(results, sorted(result_paths, key=lambda path: _load(path)["num_envs"])):
        for stage in result["stages"]:
            summary_rows.append(
                {
                    "num_agents": result["num_agents"],
                    "num_envs": result["num_envs"],
                    "model_microbatch_envs": result["model_microbatch_envs"],
                    "model_microbatch_count": result["model_microbatch_count"],
                    "graph_edges": result["graph_edges"],
                    "average_graph_degree": result["average_graph_degree"],
                    **stage,
                    "source": str(source.parent),
                }
            )
    _write_csv(output_dir / "e3_stage_summary.csv", summary_rows)

    capacity_rows = []
    for path in fullbatch_capacity_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        capacity_rows.append(
            {
                "status": payload["status"],
                "scan_cell": payload["scan_cell"],
                "exception_type": payload["exception_type"],
                "exception_message": payload["exception_message"],
                "source": str(path.parent),
            }
        )
    if capacity_rows:
        _write_csv(output_dir / "e3_diagnostic_failures.csv", capacity_rows)

    microbatch = _load(microbatch_result_path) if microbatch_result_path else None
    if microbatch is not None:
        comparisons = []
        full = results[-1]
        for label, payload in (("full_batch", full), ("mb16", microbatch)):
            stages = _stage_lookup(payload)
            comparisons.append(
                {
                    "mode": label,
                    "model_microbatch_envs": payload["model_microbatch_envs"],
                    "model_microbatch_count": payload["model_microbatch_count"],
                    "model_forward_ms": stages["model_forward"]["median_wall_ms_per_step"],
                    "closed_loop_ms": stages["closed_loop"]["median_wall_ms_per_step"],
                    "closed_loop_env_steps_s": stages["closed_loop"]["median_env_steps_s"],
                    "closed_loop_launches": stages["closed_loop"]["cuda_launch_count_per_step"],
                    "peak_nvml_process_memory_bytes": payload["memory"]["max_nvml_process_memory_bytes"],
                }
            )
        _write_csv(output_dir / "e3_e256_batching_comparison.csv", comparisons)

    (output_dir / "e3_stage_latency.svg").write_text(
        _stage_latency_svg(results), encoding="utf-8"
    )
    (output_dir / "e3_stage_share.svg").write_text(
        _share_svg(results), encoding="utf-8"
    )

    lookups = [_stage_lookup(result) for result in results]
    lines = [
        "# E3: GPU-resident MAGAT stage decomposition",
        "",
        "Configuration: standard MAPF, 256 agents, random maps at density 0.2, validation-selected MAGAT checkpoint at optimizer step 95,000. E16/E64 use one full model batch; E256 also completes as one full 65,536-agent model batch. Each formal point uses two warm-up steps and five repetitions of five measured steps.",
        "",
        "![Stage latency](e3_stage_latency.svg)",
        "",
        "## Median wall latency per environment step",
        "",
        "| Stage | E16 | E64 | E256 |",
        "|---|---:|---:|---:|",
    ]
    for stage in ("derived_state_initial_energy_map", *PLOT_STAGES):
        values = [lookup[stage]["median_wall_ms_per_step"] for lookup in lookups]
        lines.append(
            f"| {STAGE_LABELS.get(stage, 'Initial energy map')} | "
            + " | ".join(f"{value:.4f} ms" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Closed-loop performance and memory",
            "",
            "| Environments | Closed-loop env-steps/s | Agent-steps/s | Model / closed-loop | Observed peak GPU memory | Edges | Mean degree |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result, lookup in zip(results, lookups):
        closed = lookup["closed_loop"]
        share = lookup["model_forward"]["median_wall_ms_per_step"] / closed["median_wall_ms_per_step"]
        lines.append(
            f"| {result['num_envs']} | {closed['median_env_steps_s']:.2f} | "
            f"{closed['median_agent_steps_s'] / 1e3:.2f}k | {share*100:.1f}% | "
            f"{result['memory']['max_nvml_process_memory_bytes'] / 2**30:.2f} GiB | "
            f"{result['graph_edges']} | {result['average_graph_degree']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Exclusive-stage share](e3_stage_share.svg)",
            "",
            "## Main findings",
            "",
            "- The resident transition kernel is no longer the end-to-end bottleneck: it remains around 0.053--0.063 ms, while the MAGAT forward grows from 9.34 ms at E16 to 151.76 ms at E256.",
            "- MAGAT forward accounts for 95--97% of directly measured closed-loop latency. The CUDA simulator/builder acceleration therefore does extend to model-ready inputs, but full deployment throughput is bounded by the policy network.",
            "- Closed-loop throughput stays near 1.6k environment-steps/s (about 0.42M agent-steps/s) as environments increase. More environments improve simulator utilization but do not improve a model whose work scales nearly linearly with the number of agents.",
            "- Reset-time energy-map construction is a one-off cost and scales from roughly 4.06 ms at E16 to 62.57 ms at E256. Steady standard-MAPF derived-state maintenance remains around 0.15 ms because goals do not refresh.",
            "- Mean graph degree is stable at about 2.25. Node construction, rather than edge finalization, is the largest builder cost at E256 (about 4.92 ms).",
            "- Timed H2D and D2H traffic is zero: state, model inputs, logits, argmax actions, and simulator actions remain GPU-resident.",
            "- CUDA launch count rises from 226 per closed-loop E16 step to 246 for full-batch E256. This is dominated by the MAGAT forward, not the simulator transition.",
        ]
    )
    if microbatch is not None:
        full_lookup = lookups[-1]
        mb_lookup = _stage_lookup(microbatch)
        lines.extend(
            [
                "",
                "## E256 model-batching diagnostic",
                "",
                "| Mode | Model batches | Model forward | Closed loop | Peak GPU memory | Launches/step |",
                "|---|---:|---:|---:|---:|---:|",
                f"| Full 256-env batch | 1 | {full_lookup['model_forward']['median_wall_ms_per_step']:.2f} ms | {full_lookup['closed_loop']['median_wall_ms_per_step']:.2f} ms | {results[-1]['memory']['max_nvml_process_memory_bytes']/2**30:.2f} GiB | {full_lookup['closed_loop']['cuda_launch_count_per_step']:.0f} |",
                f"| 16-env microbatches | {microbatch['model_microbatch_count']} | {mb_lookup['model_forward']['median_wall_ms_per_step']:.2f} ms | {mb_lookup['closed_loop']['median_wall_ms_per_step']:.2f} ms | {microbatch['memory']['max_nvml_process_memory_bytes']/2**30:.2f} GiB | {mb_lookup['closed_loop']['cuda_launch_count_per_step']:.0f} |",
                "",
                "The env-aligned microbatch path cuts observed memory sharply while preserving latency, but increases launch count substantially. This is a useful deployment knob, not the primary E3 protocol.",
            ]
        )
    lines.extend(
        [
            "",
            "## Measurement notes",
            "",
            "- Exclusive stages are measured independently and therefore are diagnostic rather than perfectly additive; allocator reuse and stream overlap make their sum differ slightly from the directly measured closed loop.",
            "- CUDA launch counts come from a separate one-step PyTorch profiler pass and do not contaminate the five timing repetitions.",
            "- Two pre-fix diagnostic OOM attempts are retained in `e3_diagnostic_failures.csv`. They were caused by the profiling runner retaining materialized graph/model tensors across stages; the corrected full-batch E256 protocol succeeds.",
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the E3 stage report.")
    parser.add_argument("--result", action="append", required=True, type=Path)
    parser.add_argument("--microbatch-result", type=Path)
    parser.add_argument("--diagnostic-failure", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    render(
        result_paths=args.result,
        microbatch_result_path=args.microbatch_result,
        fullbatch_capacity_paths=args.diagnostic_failure,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
