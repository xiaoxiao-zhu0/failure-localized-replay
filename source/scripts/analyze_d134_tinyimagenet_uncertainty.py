#!/usr/bin/env python3
"""Add per-seed uncertainty reporting to the frozen D134 summary."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev


T_CRITICAL_95 = {1: 12.7062047364, 2: 4.30265272975, 3: 3.18244630528,
                 4: 2.7764451052, 5: 2.5705818356, 6: 2.4469118511,
                 7: 2.364624251, 8: 2.306004135, 9: 2.262157163,
                 10: 2.228138852, 11: 2.20098516, 12: 2.17881283}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def summary(values: list[float], *, bootstrap_seed: int = 134) -> dict:
    n = len(values)
    avg = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = T_CRITICAL_95.get(n - 1, 1.96)
    rng = random.Random(bootstrap_seed)
    boot = [mean(rng.choice(values) for _ in values) for _ in range(10000)]
    return {
        "n": n,
        "per_seed": values,
        "mean": avg,
        "sample_std": sd,
        "standard_error": se,
        "student_t_95_ci": [avg - t * se, avg + t * se],
        "bootstrap_percentile_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "positive_count": sum(v > 0 for v in values),
        "zero_count": sum(v == 0 for v in values),
        "negative_count": sum(v < 0 for v in values),
    }


def paired_sign_flip(values: list[float]) -> float:
    observed = abs(mean(values))
    signs = itertools.product((-1.0, 1.0), repeat=len(values))
    null = [abs(mean(sign * value for sign, value in zip(signs_row, values)))
            for signs_row in signs]
    return sum(v >= observed - 1e-15 for v in null) / len(null)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in source["seeds"]]
    methods = source["methods"]
    absolute = source["absolute"]
    contrasts = source["full_vs_layer2"]["per_seed"]

    absolute_metrics = {}
    for method in methods:
        rows = [absolute[method][str(seed)] for seed in seeds]
        values = {
            "mean_accuracy": [100 * row["mean_accuracy"] for row in rows],
            "worst_forgetting": [100 * row["worst_forgetting"] for row in rows],
            "pss": [row["pss"] for row in rows],
            "hard_accuracy": [100 * row["hard"]["final_validation_accuracy"] for row in rows],
            "blurry_accuracy": [100 * row["blurry"]["final_validation_accuracy"] for row in rows],
            "hard_forgetting": [100 * row["hard"]["final_validation_forgetting"] for row in rows],
            "blurry_forgetting": [100 * row["blurry"]["final_validation_forgetting"] for row in rows],
        }
        absolute_metrics[method] = {key: summary(value) for key, value in values.items()}

    contrast_values = {
        "accuracy_delta_pp": [100 * contrasts[str(seed)]["accuracy_delta"] for seed in seeds],
        "mean_forgetting_improvement_pp": [100 * contrasts[str(seed)]["mean_forgetting_improvement"] for seed in seeds],
        "worst_forgetting_improvement_pp": [100 * contrasts[str(seed)]["worst_forgetting_improvement"] for seed in seeds],
        "pss_relative_change_percent": [100 * contrasts[str(seed)]["pss_relative_change"] for seed in seeds],
        "hard_accuracy_delta_pp": [100 * contrasts[str(seed)]["stream_delta"]["hard"]["accuracy_delta"] for seed in seeds],
        "blurry_accuracy_delta_pp": [100 * contrasts[str(seed)]["stream_delta"]["blurry"]["accuracy_delta"] for seed in seeds],
    }
    paired = {key: {**summary(values), "exact_sign_flip_p_value": paired_sign_flip(values)}
              for key, values in contrast_values.items()}

    output = {
        "source": str(args.input),
        "protocol": {"stage": source["stage"], "seeds": seeds, "methods": methods},
        "absolute": absolute_metrics,
        "full_vs_layer2": paired,
        "per_seed_contrasts": {str(seed): contrasts[str(seed)] for seed in seeds},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "d134_uncertainty_analysis.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# D134 Tiny ImageNet uncertainty analysis",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. All statistics use paired seed identities.",
        "",
        "## Absolute results",
        "",
        "| Method | Metric | Mean | SD | 95% Student-t CI | Per-seed values |",
        "|---|---|---:|---:|---|---|",
    ]
    display_metrics = ("mean_accuracy", "worst_forgetting", "pss", "hard_accuracy", "blurry_accuracy")
    for method in methods:
        for key in display_metrics:
            item = absolute_metrics[method][key]
            lines.append(
                f"| {method} | {key} | {item['mean']:.4f} | {item['sample_std']:.4f} | "
                f"[{item['student_t_95_ci'][0]:.4f}, {item['student_t_95_ci'][1]:.4f}] | "
                f"{', '.join(f'{v:.4f}' for v in item['per_seed'])} |"
            )
    lines += [
        "", "## Full minus Layer 2 paired effects", "",
        "| Effect | Mean | SD | 95% Student-t CI | Bootstrap 95% CI | Direction | Exact sign-flip p | Per-seed values |",
        "|---|---:|---:|---|---|---|---:|---|",
    ]
    for key, item in paired.items():
        direction = f"{item['positive_count']}/{item['n']} positive"
        lines.append(
            f"| {key} | {item['mean']:.4f} | {item['sample_std']:.4f} | "
            f"[{item['student_t_95_ci'][0]:.4f}, {item['student_t_95_ci'][1]:.4f}] | "
            f"[{item['bootstrap_percentile_95_ci'][0]:.4f}, {item['bootstrap_percentile_95_ci'][1]:.4f}] | "
            f"{direction} | {item['exact_sign_flip_p_value']:.4f} | "
            f"{', '.join(f'{v:.4f}' for v in item['per_seed'])} |"
        )
    lines += [
        "", "Interpretation boundary: five seeds support uncertainty and directional reporting, "
        "not broad universal significance claims. The primary paper text should report the "
        "per-seed effects and intervals, while the full JSON preserves exact values."
    ]
    (args.output_dir / "d134_uncertainty_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
