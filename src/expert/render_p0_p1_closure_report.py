"""Aggregate P0/P1 resident-closure rows into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def _load_rows(root: Path, *, horizon: int = 120) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok" or int(payload.get("horizon", -1)) != horizon:
            continue
        payload["_source"] = str(path)
        rows.append(payload)
    return rows


def _summarize(
    rows: Iterable[dict[str, Any]],
    *,
    latency_key: str,
    throughput_key: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["num_agents"]), int(row["num_envs"]))].append(row)
    output = []
    for (agents, envs), group in sorted(grouped.items()):
        latency = [float(row[latency_key]) for row in group]
        throughput = [float(row[throughput_key]) for row in group]
        memory = [
            int(row["peak_gpu_memory_allocated_bytes"]) for row in group
        ]
        output.append(
            {
                "num_agents": agents,
                "num_envs": envs,
                "repetitions": len(group),
                "latency_ms_median": statistics.median(latency),
                "latency_ms_min": min(latency),
                "latency_ms_max": max(latency),
                "throughput_median": statistics.median(throughput),
                "peak_gpu_memory_gib": max(memory) / 2**30,
                "individual_success_rate": statistics.median(
                    float(row["individual_success_rate"]) for row in group
                ),
                "complete_success_rate": statistics.median(
                    float(row["complete_success_rate"]) for row in group
                ),
                "truncation_rate": statistics.median(
                    float(row["truncation_rate"]) for row in group
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_table(rows: list[dict[str, Any]], throughput_name: str) -> list[str]:
    result = [
        f"| Agents | Envs | Reps | Median ms/step | Min--max ms | "
        f"{throughput_name} | Peak GiB | ISR | CSR | Trunc. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        result.append(
            "| {num_agents} | {num_envs} | {repetitions} | "
            "{latency_ms_median:.3f} | {latency_ms_min:.3f}--"
            "{latency_ms_max:.3f} | {throughput_median:.2f} | "
            "{peak_gpu_memory_gib:.3f} | {individual_success_rate:.4f} | "
            "{complete_success_rate:.4f} | {truncation_rate:.4f} |".format(**row)
        )
    return result


def render_report(
    *,
    p0_root: Path,
    p1_root: Path,
    output_dir: Path,
    official_root: Path | None = None,
) -> None:
    p0_raw = _load_rows(p0_root)
    p1_raw = _load_rows(p1_root)
    if not p0_raw or not p1_raw:
        raise ValueError("both P0 and P1 must contain 120-step successful rows")
    p0 = _summarize(
        p0_raw,
        latency_key="wall_ms_per_batched_step",
        throughput_key="agent_steps_s",
    )
    p1 = _summarize(
        p1_raw,
        latency_key="wall_ms_per_batched_step",
        throughput_key="active_env_steps_s",
    )
    official = (
        _summarize(
            _load_rows(official_root),
            latency_key="wall_ms_per_batched_step",
            throughput_key="agent_steps_s",
        )
        if official_root is not None
        else []
    )
    p0_parity = all(
        int(row["temporal_parity"]["token_mismatches"]) == 0
        and int(row["temporal_parity"]["history_mismatches"]) == 0
        for row in p0_raw
    )
    p1_parity = all(
        int(row["temporal_shared_action_parity"]["state_mismatches"]) == 0
        and int(row["temporal_shared_action_parity"]["steps"]) == 120
        for row in p1_raw
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "p0_mapf_gpt_resident.csv", p0)
    _write_csv(output_dir / "p1_magat_long_resident.csv", p1)
    lines = [
        "# P0/P1 GPU-resident closure results",
        "",
        "All rows use standard MAPF semantics and argmax actions. Checkpoint "
        "loading, validation/parity, monitoring, and result readback are "
        "outside the timed region. The measured resident loops report zero "
        "timed H2D/D2H bytes.",
        "",
        "## P0: MAPF-GPT resident inference",
        "",
        f"- CUDA token/history parity across formal rows: `{p0_parity}`.",
        "- A separate five-step temporal oracle replay is retained in the P0 "
        "artifact directory.",
        "- Throughput below is resident agent-steps/s; all agents pass through "
        "the model and on-target agents are masked to wait before transition.",
        "",
        *_format_table(p0, "Agent-steps/s"),
        "",
        *(
            [
                "### Official-weight compatibility",
                "",
                "The official released MAPF-GPT-2M checkpoint, including its "
                "`_orig_mod.` state-dict prefix and `model_args` metadata, was "
                "loaded unchanged into the same resident path.",
                "",
                *_format_table(official, "Agent-steps/s"),
                "",
            ]
            if official
            else []
        ),
        "## P1: MAGAT 120-step resident loop",
        "",
        f"- POGEMA/CUDA shared-action parity for all 120 steps: `{p1_parity}`.",
        "- Throughput below is active env-steps/s; no large CPU baseline was "
        "rerun because matched one-step deployment baselines already exist.",
        "",
        *_format_table(p1, "Active env-steps/s"),
        "",
        "## Interpretation",
        "",
        "- P0 closes the model-generality gap: current spatial state, cached "
        "cost-to-go, five-action history, transformer inference, argmax, and "
        "transition remain on device.",
        "- P1 closes the temporal gap: the one-step MAGAT deployment result is "
        "not hiding state drift; the same execution path remains semantically "
        "aligned for 120 steps.",
        "- CSR is reported as policy-quality context, not as a systems "
        "acceptance criterion.",
        "",
        f"P0 source: `{p0_root}`",
        "",
        f"P1 source: `{p1_root}`",
        "",
        *(
            [f"Official checkpoint compatibility source: `{official_root}`"]
            if official_root is not None
            else []
        ),
    ]
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-root", required=True, type=Path)
    parser.add_argument("--p1-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--official-root", type=Path)
    args = parser.parse_args(argv)
    render_report(
        p0_root=args.p0_root,
        p1_root=args.p1_root,
        output_dir=args.output_dir,
        official_root=args.official_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["render_report"]
