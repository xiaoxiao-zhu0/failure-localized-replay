#!/usr/bin/env python3
"""Aggregate matched hard/blurry CORe50 evidence for the A/B/C/D chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import deployment_stream_metrics


METHODS = {
    "A": "causal_er_ace",
    "B": "semantic_proto_hybrid_75_25",
    "C": "persistent_srrd_selective_swap_1",
    "D": "persistent_srrd_prequential_arbitration_1",
}


def load(root: Path, benchmark: str, seed: int, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def standard_metrics(payload: dict) -> dict[str, float]:
    history = payload["classwise_audit"]["consequence_history"]
    classes = history[-1]["classes"]
    return {
        "final_validation_accuracy": mean(
            float(item["accuracy"]) for item in classes.values()
        ),
        "final_validation_forgetting": mean(
            float(item["forgetting"]) for item in classes.values()
        ),
    }


def metrics(payload: dict, method: str) -> dict[str, float]:
    if method == METHODS["D"]:
        return deployment_stream_metrics(payload)
    return standard_metrics(payload)


def protocol(payload: dict, benchmark: str, seed: int) -> dict[str, bool]:
    metadata = payload.get("run_metadata", {})
    checks = {
        "benchmark": metadata.get("benchmark") == benchmark,
        "seed": int(metadata.get("seed", -1)) == seed,
        "experiences": int(metadata.get("n_experiences", -1)) == 9,
        "epochs": int(metadata.get("train_epochs", -1)) == 5,
        "memory": int(metadata.get("mem_size", -1)) == 100,
        "learning_rate": math.isclose(float(metadata.get("lr", -1)), 0.01),
    }
    return checks


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--blurry-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    hard_name = "core50"
    blurry_name = "equal_exposure_blurry_core50"
    rows: dict[str, dict[str, dict]] = {label: {} for label in METHODS}
    protocol_checks = []
    for seed in args.seeds:
        for label, method in METHODS.items():
            hard = load(args.hard_root, hard_name, seed, method)
            blurry = load(args.blurry_root, blurry_name, seed, method)
            hard_metrics = metrics(hard, method)
            blurry_metrics = metrics(blurry, method)
            gaps = {
                key: blurry_metrics[key] - hard_metrics[key]
                for key in hard_metrics
            }
            rows[label][str(seed)] = {
                "seed": seed,
                "method": method,
                "hard": hard_metrics,
                "blurry": blurry_metrics,
                "blurry_minus_hard": gaps,
                "pss": sum(abs(value) for value in gaps.values()),
                "mean_accuracy": mean(
                    [
                        hard_metrics["final_validation_accuracy"],
                        blurry_metrics["final_validation_accuracy"],
                    ]
                ),
                "mean_forgetting": mean(
                    [
                        hard_metrics["final_validation_forgetting"],
                        blurry_metrics["final_validation_forgetting"],
                    ]
                ),
                "worst_forgetting": max(
                    hard_metrics["final_validation_forgetting"],
                    blurry_metrics["final_validation_forgetting"],
                ),
            }
            protocol_checks.append(
                {
                    "seed": seed,
                    "method": method,
                    "hard": protocol(hard, hard_name, seed),
                    "blurry": protocol(blurry, blurry_name, seed),
                }
            )

    aggregate = {}
    for label, method in METHODS.items():
        method_rows = [rows[label][str(seed)] for seed in args.seeds]
        aggregate[label] = {
            "method": method,
            "accuracy": summarize([row["mean_accuracy"] for row in method_rows]),
            "mean_forgetting": summarize(
                [row["mean_forgetting"] for row in method_rows]
            ),
            "pss": summarize([row["pss"] for row in method_rows]),
            "worst_forgetting": summarize(
                [row["worst_forgetting"] for row in method_rows]
            ),
        }

    comparisons = {}
    for child, parent in (("B", "A"), ("C", "B"), ("D", "C"), ("D", "A")):
        per_seed = []
        for seed in args.seeds:
            left = rows[child][str(seed)]
            right = rows[parent][str(seed)]
            per_seed.append(
                {
                    "seed": seed,
                    "accuracy_delta_pp": 100.0
                    * (left["mean_accuracy"] - right["mean_accuracy"]),
                    "forgetting_improvement_pp": 100.0
                    * (right["mean_forgetting"] - left["mean_forgetting"]),
                    "pss_delta_pp": 100.0 * (left["pss"] - right["pss"]),
                }
            )
        comparisons[f"{child}-{parent}"] = {
            "per_seed": per_seed,
            "accuracy_delta_pp": summarize(
                [row["accuracy_delta_pp"] for row in per_seed]
            ),
            "forgetting_improvement_pp": summarize(
                [row["forgetting_improvement_pp"] for row in per_seed]
            ),
            "pss_delta_pp": summarize([row["pss_delta_pp"] for row in per_seed]),
            "accuracy_improves_count": sum(
                row["accuracy_delta_pp"] > 0 for row in per_seed
            ),
            "forgetting_improves_count": sum(
                row["forgetting_improvement_pp"] > 0 for row in per_seed
            ),
        }

    all_protocol_pass = all(
        all(checks.values())
        for row in protocol_checks
        for checks in (row["hard"], row["blurry"])
    )
    report = {
        "stage": "D135 matched CORe50 hard/blurry cross-stream confirmation",
        "status": "completed" if all_protocol_pass else "protocol_mismatch",
        "benchmarks": {"hard": hard_name, "blurry": blurry_name},
        "seeds": args.seeds,
        "methods": METHODS,
        "protocol": {
            "learning_rate": 0.01,
            "train_epochs": 5,
            "memory_size": 100,
            "paired_seed_policy": True,
        },
        "protocol_checks": protocol_checks,
        "all_protocol_checks_pass": all_protocol_pass,
        "per_method": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "interpretation_note": (
            "PSS is an auxiliary paired hard/blurry sensitivity measure; lower is better. "
            "All effects are descriptive with three paired seeds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# D135 matched CORe50 hard/blurry cross-stream confirmation",
        "",
        f"Protocol checks pass: {all_protocol_pass}",
        "",
        "| Method | Mean accuracy (%) | Mean forgetting (%) | PSS (pp) |",
        "|:--|--:|--:|--:|",
    ]
    for label, method in METHODS.items():
        item = aggregate[label]
        lines.append(
            f"| {label}: {method} | "
            f"{100 * item['accuracy']['mean']:.3f} +/- "
            f"{100 * item['accuracy']['sample_std']:.3f} | "
            f"{100 * item['mean_forgetting']['mean']:.3f} +/- "
            f"{100 * item['mean_forgetting']['sample_std']:.3f} | "
            f"{100 * item['pss']['mean']:.3f} +/- "
            f"{100 * item['pss']['sample_std']:.3f} |"
        )
    lines += ["", "| Contrast | Accuracy delta (pp) | Forgetting improvement (pp) | PSS delta (pp) |", "|:--|--:|--:|--:|"]
    for contrast, item in comparisons.items():
        lines.append(
            f"| {contrast} | {item['accuracy_delta_pp']['mean']:+.3f} | "
            f"{item['forgetting_improvement_pp']['mean']:+.3f} | "
            f"{item['pss_delta_pp']['mean']:+.3f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
