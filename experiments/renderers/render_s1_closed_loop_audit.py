"""Derive seed-clustered paired statistics from an accepted S1 result matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPARISONS = (
    ("proposed_async", "strong_online"),
    ("proposed_async", "compact_sync"),
    ("compact_sync", "strong_online"),
)
METRICS = ("normalized_arrival_gain", "incremental_arrival_auc")


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _bootstrap_seed_means(
    values: Sequence[float], *, draws: int = 20_000, seed: int = 20_260_810
) -> tuple[float, float]:
    if not values:
        raise ValueError("seed-cluster bootstrap requires at least one seed")
    rng = random.Random(seed)
    samples = sorted(
        statistics.mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    )
    return samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def paired_seed_summary(
    results: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    left_mode: str,
    right_mode: str,
) -> dict[str, Any]:
    selected = [row for row in results if row["mode"] in {left_mode, right_mode}]
    seeds = sorted({int(row["seed"]) for row in selected})
    per_seed = []
    for seed in seeds:
        left = {
            (int(row["num_agents"]), row["topology"]): float(row[metric])
            for row in selected
            if row["mode"] == left_mode and int(row["seed"]) == seed
        }
        right = {
            (int(row["num_agents"]), row["topology"]): float(row[metric])
            for row in selected
            if row["mode"] == right_mode and int(row["seed"]) == seed
        }
        if not left or left.keys() != right.keys():
            raise ValueError(
                f"unpaired S1 cells for seed={seed}, modes={left_mode}/{right_mode}"
            )
        differences = [left[cell] - right[cell] for cell in sorted(left)]
        per_seed.append(
            {
                "seed": seed,
                "mean_paired_difference": statistics.mean(differences),
                "median_paired_difference": statistics.median(differences),
                "left_wins": sum(value > 0.0 for value in differences),
                "ties": sum(value == 0.0 for value in differences),
                "cell_count": len(differences),
                "cell_differences": differences,
            }
        )
    seed_means = [float(row["mean_paired_difference"]) for row in per_seed]
    low, high = _bootstrap_seed_means(seed_means)
    return {
        "metric": metric,
        "left_mode": left_mode,
        "right_mode": right_mode,
        "mean_paired_difference": statistics.mean(seed_means),
        "median_seed_mean_difference": statistics.median(seed_means),
        "seed_cluster_bootstrap_95_interval": [low, high],
        "all_seed_means_positive": all(value > 0.0 for value in seed_means),
        "seed_count": len(seed_means),
        "per_seed": per_seed,
        "uncertainty_note": (
            "The bootstrap resamples training seeds as clusters; topology/agent "
            "cells within one seed are paired observations, not independent seeds."
        ),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return None if denominator == 0.0 else numerator / denominator


def derive_analysis(payload: Mapping[str, Any]) -> dict[str, Any]:
    results = [row for row in payload["results"] if row.get("status") == "ok"]
    if len(results) != int(payload["row_count"]):
        raise ValueError("S1 paired analysis requires all rows to be successful")
    paired = [
        paired_seed_summary(results, metric=metric, left_mode=left, right_mode=right)
        for metric in METRICS
        for left, right in COMPARISONS
    ]

    checkpoint_rows = []
    for mode in sorted({row["mode"] for row in results}):
        seeds = sorted({int(row["seed"]) for row in results if row["mode"] == mode})
        for seed in seeds:
            rows = [
                row
                for row in results
                if row["mode"] == mode and int(row["seed"]) == seed
            ]
            checkpoint_rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "validation_overall_accuracy": float(
                        rows[0]["checkpoint"]["validation_overall_accuracy"]
                    ),
                    "mean_normalized_arrival_gain": statistics.mean(
                        float(row["normalized_arrival_gain"]) for row in rows
                    ),
                    "mean_incremental_arrival_auc": statistics.mean(
                        float(row["incremental_arrival_auc"]) for row in rows
                    ),
                }
            )
    correlation = _pearson(
        [row["validation_overall_accuracy"] for row in checkpoint_rows],
        [row["mean_normalized_arrival_gain"] for row in checkpoint_rows],
    )
    return {
        "schema_version": 1,
        "experiment": "S1-formal-paired-analysis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": payload["experiment"],
        "source_code_commit": payload["suite_config"]["code_commit"],
        "row_count": len(results),
        "paired_comparisons": paired,
        "checkpoint_level": checkpoint_rows,
        "validation_accuracy_vs_arrival_gain_pearson": correlation,
        "interpretation_limits": [
            "Only three independent training seeds are available.",
            "Complete success rate is zero in every formal row.",
            "Correlation across nine checkpoints is descriptive and not causal.",
            "S1 audits retained checkpoints; it does not reproduce solver-quality training.",
        ],
    }


def render_analysis(*, source: Path, output_dir: Path, repository: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"S1 analysis output already exists: {output_dir}")
    payload = json.loads(source.expanduser().resolve().read_text(encoding="utf-8"))
    analysis = derive_analysis(payload)
    analysis["analysis_code_commit"] = _git_commit(repository)
    analysis["source_path"] = str(source.expanduser().resolve())
    output_dir.mkdir(parents=True)
    (output_dir / "paired-analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# S1 formal paired analysis",
        "",
        "Differences are `left - right`. The interval resamples the three training "
        "seeds as clusters; cells within a seed stay paired.",
        "",
        "| Metric | Left | Right | Mean difference | Seed-cluster bootstrap 95% interval | Per-seed mean differences | All seeds positive? |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in analysis["paired_comparisons"]:
        interval = row["seed_cluster_bootstrap_95_interval"]
        seed_values = ", ".join(
            f"s{value['seed']}={value['mean_paired_difference']:.4f}"
            for value in row["per_seed"]
        )
        lines.append(
            f"| {row['metric']} | {row['left_mode']} | {row['right_mode']} | "
            f"{row['mean_paired_difference']:.4f} | [{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{seed_values} | {row['all_seed_means_positive']} |"
        )
    lines.extend(
        [
            "",
            f"Across the nine retained checkpoints, validation accuracy and mean "
            f"arrival gain have descriptive Pearson correlation "
            f"{analysis['validation_accuracy_vs_arrival_gain_pearson']:.3f}.",
            "",
            "All 81 formal rows have CSR=0. The defensible S1 claim is therefore "
            "about partial closed-loop progress and preservation, not episode solve rate.",
            "",
            "With only three seeds, intervals are descriptive. In particular, proposed "
            "is consistently above strong_online in seed-mean progress, but its comparison "
            "with compact_sync changes sign across seeds.",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    analysis = render_analysis(
        source=args.source,
        output_dir=args.output_dir,
        repository=args.repository.expanduser().resolve(),
    )
    print(json.dumps({"status": "ok", "row_count": analysis["row_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["derive_analysis", "paired_seed_summary", "render_analysis"]
