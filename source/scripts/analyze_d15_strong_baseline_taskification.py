"""Analyze taskification sensitivity of canonical strong OCL baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "budget_stats" in payload:
        history = payload["budget_stats"]["consequence_history"]
    else:
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


def pair(
    root: Path,
    benchmark_hard: str,
    benchmark_blurry: str,
    seed: int,
    method: str,
) -> dict:
    hard = metrics(root / benchmark_hard / f"seed_{seed}" / method / "summary.json")
    blurry = metrics(
        root / benchmark_blurry / f"seed_{seed}" / method / "summary.json"
    )
    gaps = {key: blurry[key] - hard[key] for key in hard}
    return {
        "hard": hard,
        "blurry": blurry,
        "blurry_minus_hard": gaps,
        "pss": sum(abs(value) for value in gaps.values()),
        "mean_accuracy": (
            hard["final_validation_accuracy"]
            + blurry["final_validation_accuracy"]
        )
        / 2.0,
        "worst_forgetting": max(
            hard["final_validation_forgetting"],
            blurry["final_validation_forgetting"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["replay", "derpp", "er_ace", "mir"]
    )
    parser.add_argument("--hard-benchmark", default="split_cifar100")
    parser.add_argument(
        "--blurry-benchmark", default="equal_exposure_blurry_cifar100"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {
        method: pair(
            args.root,
            args.hard_benchmark,
            args.blurry_benchmark,
            args.seed,
            method,
        )
        for method in args.methods
    }
    er = rows["replay"]
    comparisons = {}
    for method, row in rows.items():
        pss_reduction = 1.0 - row["pss"] / er["pss"]
        mean_accuracy_delta = row["mean_accuracy"] - er["mean_accuracy"]
        worst_forgetting_delta = (
            row["worst_forgetting"] - er["worst_forgetting"]
        )
        checks = {
            "pss_reduction_at_least_30pct": pss_reduction >= 0.30,
            "mean_accuracy_guard": mean_accuracy_delta >= -0.01,
            "retention_guard": worst_forgetting_delta <= 0.03,
        }
        comparisons[method] = {
            "pss_reduction_fraction_vs_er": pss_reduction,
            "mean_accuracy_delta_vs_er": mean_accuracy_delta,
            "worst_forgetting_delta_vs_er": worst_forgetting_delta,
            "checks": checks,
            "is_taskification_robust_candidate": (
                method != "replay" and all(checks.values())
            ),
        }

    robust = [
        method
        for method, row in comparisons.items()
        if row["is_taskification_robust_candidate"]
    ]
    report = {
        "stage": "D15 strong-baseline taskification audit",
        "seed": args.seed,
        "methods": args.methods,
        "per_method": rows,
        "comparisons_against_er": comparisons,
        "robust_candidate_methods": robust,
        "any_strong_baseline_solves_screen": bool(robust),
        "problem_persists_across_methods": (
            sum(row["pss"] >= 0.05 for row in rows.values()) >= 3
        ),
        "locked_rule": (
            "A strong baseline is a taskification-robust candidate only if it "
            "reduces ER PSS by >=30%, keeps mean accuracy within -0.01, and "
            "does not worsen worst-stream forgetting by more than 0.03."
        ),
        "next_action": (
            "Analyze the residual of the best robust strong baseline before "
            "claiming a new mechanism."
            if robust
            else "Proceed to the pre-specified non-invasive target-repair audit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
