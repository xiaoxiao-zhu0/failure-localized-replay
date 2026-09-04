#!/usr/bin/env python3
"""Compare matched CORe50 OBC results with the frozen A/B/C/D matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import deployment_stream_metrics


METHODS = {
    "A: CE-ACE": "causal_er_ace",
    "B: Layer 1": "semantic_proto_hybrid_75_25",
    "C: Layer 2": "persistent_srrd_selective_swap_1",
    "D: PRBA": "persistent_srrd_prequential_arbitration_1",
    "OBC": "persistent_srrd_obc_1",
}


def load(root: Path, benchmark: str, seed: int, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def standard_metrics(payload: dict) -> dict[str, float]:
    classes = payload["classwise_audit"]["consequence_history"][-1]["classes"]
    return {
        "final_validation_accuracy": mean(
            float(row["accuracy"]) for row in classes.values()
        ),
        "final_validation_forgetting": mean(
            float(row["forgetting"]) for row in classes.values()
        ),
    }


def metrics(payload: dict, method: str) -> dict[str, float]:
    if method in {
        METHODS["D: PRBA"],
        METHODS["OBC"],
    }:
        return deployment_stream_metrics(payload)
    return standard_metrics(payload)


def protocol(payload: dict, benchmark: str, seed: int) -> dict[str, bool]:
    metadata = payload.get("run_metadata", {})
    return {
        "benchmark": metadata.get("benchmark") == benchmark,
        "seed": int(metadata.get("seed", -1)) == seed,
        "experiences": int(metadata.get("n_experiences", -1)) == 9,
        "epochs": int(metadata.get("train_epochs", -1)) == 5,
        "memory": int(metadata.get("mem_size", -1)) == 100,
        "learning_rate": math.isclose(float(metadata.get("lr", -1)), 0.01),
    }


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--blurry-root", type=Path, required=True)
    parser.add_argument("--obc-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    rows: dict[str, dict[str, dict]] = {label: {} for label in METHODS}
    checks = []
    for seed in args.seeds:
        for label, method in METHODS.items():
            hard_root = args.obc_root if label == "OBC" else args.hard_root
            blurry_root = args.obc_root if label == "OBC" else args.blurry_root
            hard_name = "core50"
            blurry_name = "equal_exposure_blurry_core50"
            hard = load(hard_root, hard_name, seed, method)
            blurry = load(blurry_root, blurry_name, seed, method)
            hard_metrics = metrics(hard, method)
            blurry_metrics = metrics(blurry, method)
            gaps = {key: blurry_metrics[key] - hard_metrics[key] for key in hard_metrics}
            rows[label][str(seed)] = {
                "seed": seed,
                "method": method,
                "hard": hard_metrics,
                "blurry": blurry_metrics,
                "blurry_minus_hard": gaps,
                "pss": sum(abs(value) for value in gaps.values()),
                "mean_accuracy": mean([hard_metrics["final_validation_accuracy"], blurry_metrics["final_validation_accuracy"]]),
                "mean_forgetting": mean([hard_metrics["final_validation_forgetting"], blurry_metrics["final_validation_forgetting"]]),
            }
            checks.append({
                "seed": seed,
                "method": method,
                "hard": protocol(hard, hard_name, seed),
                "blurry": protocol(blurry, blurry_name, seed),
            })

    aggregate = {}
    for label, method in METHODS.items():
        values = [rows[label][str(seed)] for seed in args.seeds]
        aggregate[label] = {
            "method": method,
            "accuracy": summary([row["mean_accuracy"] for row in values]),
            "mean_forgetting": summary([row["mean_forgetting"] for row in values]),
            "pss": summary([row["pss"] for row in values]),
        }

    comparisons = {}
    for left, right in (("OBC", "D: PRBA"), ("OBC", "C: Layer 2"), ("OBC", "A: CE-ACE")):
        deltas = []
        for seed in args.seeds:
            child = rows[left][str(seed)]
            parent = rows[right][str(seed)]
            deltas.append({
                "seed": seed,
                "accuracy_delta_pp": 100 * (child["mean_accuracy"] - parent["mean_accuracy"]),
                "forgetting_improvement_pp": 100 * (parent["mean_forgetting"] - child["mean_forgetting"]),
                "pss_delta_pp": 100 * (child["pss"] - parent["pss"]),
            })
        comparisons[f"{left} - {right}"] = {
            "per_seed": deltas,
            "accuracy_delta_pp": summary([row["accuracy_delta_pp"] for row in deltas]),
            "forgetting_improvement_pp": summary([row["forgetting_improvement_pp"] for row in deltas]),
            "pss_delta_pp": summary([row["pss_delta_pp"] for row in deltas]),
            "accuracy_improves_count": sum(row["accuracy_delta_pp"] > 0 for row in deltas),
            "forgetting_improves_count": sum(row["forgetting_improvement_pp"] > 0 for row in deltas),
            "pss_improves_count": sum(row["pss_delta_pp"] < 0 for row in deltas),
        }

    all_protocol_checks_pass = all(
        all(axis.values()) for check in checks for axis in (check["hard"], check["blurry"])
    )
    report = {
        "stage": "D135 matched CORe50 OBC comparison",
        "status": "completed" if all_protocol_checks_pass else "protocol_mismatch",
        "seeds": args.seeds,
        "protocol": {"learning_rate": 0.01, "train_epochs": 5, "memory_size": 100, "paired_seed_policy": True},
        "protocol_checks": checks,
        "all_protocol_checks_pass": all_protocol_checks_pass,
        "aggregate": aggregate,
        "per_method": rows,
        "comparisons": comparisons,
        "interpretation_note": "PSS is lower-is-better paired hard/blurry sensitivity; all effects are descriptive with three paired seeds.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# D135 matched CORe50 OBC comparison",
        "",
        f"Protocol checks pass: {all_protocol_checks_pass}",
        "",
        "| Method | Mean accuracy (%) | Mean forgetting (%) | PSS (pp) |",
        "|:--|--:|--:|--:|",
    ]
    for label in METHODS:
        item = aggregate[label]
        lines.append(
            f"| {label} | {100 * item['accuracy']['mean']:.3f} +/- {100 * item['accuracy']['sample_std']:.3f} | "
            f"{100 * item['mean_forgetting']['mean']:.3f} +/- {100 * item['mean_forgetting']['sample_std']:.3f} | "
            f"{100 * item['pss']['mean']:.3f} +/- {100 * item['pss']['sample_std']:.3f} |"
        )
    lines += ["", "| Contrast | Accuracy delta (pp) | Forgetting improvement (pp) | PSS delta (pp) |", "|:--|--:|--:|--:|"]
    for name, item in comparisons.items():
        lines.append(
            f"| {name} | {item['accuracy_delta_pp']['mean']:+.3f} | "
            f"{item['forgetting_improvement_pp']['mean']:+.3f} | {item['pss_delta_pp']['mean']:+.3f} |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
