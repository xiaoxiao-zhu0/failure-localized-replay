#!/usr/bin/env python3
"""Build the corrected four-method D134 Tiny ImageNet matrix."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import deployment_stream_metrics
from analyze_d15_strong_baseline_taskification import pair


HARD = "split_tinyimagenet"
BLURRY = "equal_exposure_blurry_tinyimagenet"
CEACE = "causal_er_ace"
LAYER2 = "persistent_srrd_selective_swap_1"
PRBA = "persistent_srrd_prequential_arbitration_1"
OBC = "persistent_srrd_obc_1"
METHODS = (CEACE, LAYER2, PRBA, OBC)
DEPLOYMENT_HEAD_METHODS = {PRBA, OBC}
LABELS = {
    CEACE: "CE-ACE",
    LAYER2: "Layer 2 / C",
    PRBA: "PRBA",
    OBC: "OBC",
}
EXPECTED_MEAN = [0.4914, 0.4822, 0.4465]
EXPECTED_STD = [0.2023, 0.1994, 0.2010]
T_CRITICAL_95 = {
    1: 12.7062047364,
    2: 4.30265272975,
    3: 3.18244630528,
    4: 2.7764451052,
    5: 2.5705818356,
    6: 2.4469118511,
    7: 2.364624251,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
    11: 2.20098516,
    12: 2.17881283,
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize(values: list[float], *, bootstrap_seed: int = 134) -> dict:
    n = len(values)
    avg = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    critical = T_CRITICAL_95.get(n - 1, 1.96)
    rng = random.Random(bootstrap_seed)
    bootstrap = [mean(rng.choice(values) for _ in values) for _ in range(10000)]
    return {
        "n": n,
        "per_seed": values,
        "mean": avg,
        "sample_std": sd,
        "standard_error": se,
        "student_t_95_ci": [avg - critical * se, avg + critical * se],
        "bootstrap_percentile_95_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "positive_count": sum(value > 0.0 for value in values),
        "zero_count": sum(value == 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
    }


def paired_sign_flip(values: list[float]) -> float:
    observed = abs(mean(values))
    null = (
        abs(mean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    )
    return sum(value >= observed - 1e-15 for value in null) / (2 ** len(values))


def load_summary(root: Path, benchmark: str, seed: int, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def stream_pair(root: Path, seed: int, method: str) -> dict:
    hard_payload = load_summary(root, HARD, seed, method)
    blurry_payload = load_summary(root, BLURRY, seed, method)
    if method in DEPLOYMENT_HEAD_METHODS:
        hard = deployment_stream_metrics(hard_payload)
        blurry = deployment_stream_metrics(blurry_payload)
        gaps = {key: blurry[key] - hard[key] for key in hard}
        row = {
            "hard": hard,
            "blurry": blurry,
            "blurry_minus_hard": gaps,
            "pss": sum(abs(value) for value in gaps.values()),
            "mean_accuracy": mean(
                stream["final_validation_accuracy"] for stream in (hard, blurry)
            ),
            "worst_forgetting": max(
                hard["final_validation_forgetting"],
                blurry["final_validation_forgetting"],
            ),
            "evaluation_source": "deployment_head_per_class_audit_history",
        }
    else:
        row = pair(root, HARD, BLURRY, seed, method)
        row["evaluation_source"] = "training_head_classwise_audit_history"
    row["mean_forgetting"] = mean(
        row[stream]["final_validation_forgetting"] for stream in ("hard", "blurry")
    )
    return row


def close_list(actual: object, expected: list[float]) -> bool:
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    return all(math.isclose(float(left), right, abs_tol=1e-12) for left, right in zip(actual, expected))


def protocol_checks(payload: dict, benchmark: str, seed: int, method: str) -> dict:
    metadata = payload.get("run_metadata", {})
    checks = {
        "benchmark": metadata.get("benchmark") == benchmark,
        "method": metadata.get("strategy") == method,
        "seed": int(metadata.get("seed", -1)) == seed,
        "model": metadata.get("model") == "slim_resnet18",
        "experiences": int(metadata.get("n_experiences", -1)) == 10,
        "epochs": int(metadata.get("train_epochs", -1)) == 5,
        "memory": int(metadata.get("mem_size", -1)) == 100,
        "current_minibatch": int(metadata.get("train_mb_size", -1)) == 64,
        "replay_minibatch": int(metadata.get("replay_mb_size", -1)) == 64,
        "learning_rate": math.isclose(float(metadata.get("lr", -1)), 0.05),
        "momentum": math.isclose(float(metadata.get("momentum", -1)), 0.9),
        "validation_fraction": math.isclose(
            float(metadata.get("validation_fraction", -1)), 0.0
        ),
        "historical_reference": metadata.get("historical_reference") == "test_stream",
        "deterministic": metadata.get("deterministic") is True,
    }
    if method != CEACE:
        memory = payload.get("strategy_audit", {}).get(
            "semantic_representative_memory", {}
        )
        checks.update(
            {
                "dataset_family": memory.get("dataset_family") == "tinyimagenet",
                "corrected_normalization_mean": close_list(
                    memory.get("input_normalization_mean"), EXPECTED_MEAN
                ),
                "corrected_normalization_std": close_list(
                    memory.get("input_normalization_std"), EXPECTED_STD
                ),
            }
        )
    return checks


def parent_path_checks(root: Path, seed: int, child_method: str) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load_summary(root, benchmark, seed, LAYER2)
        child = load_summary(root, benchmark, seed, child_method)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        child_hash = child["strategy_audit"]["memory_trace_determinism"]
        checks = {
            "training_model_exact": parent_hash["final_model_hash"]
            == child_hash["final_model_hash"],
            "memory_exact": parent_hash["final_memory_hash"]
            == child_hash["final_memory_hash"],
            "replay_indices_exact": parent_hash["replay_index_hash"]
            == child_hash["replay_index_hash"],
        }
        if child_method == PRBA:
            arbitration = child["strategy_audit"]["risk_budgeted_head_arbitration"]
            checks.update(
                {
                    "prequential_order": arbitration["prequential_test_then_train"],
                    "no_same_update_reuse": not arbitration[
                        "post_update_replay_reuse_leakage"
                    ],
                    "no_extra_replay_or_backbone": (
                        arbitration["additional_replay_draws"] == 0
                        and arbitration["additional_backbone_forwards"] == 0
                    ),
                }
            )
        elif child_method == OBC:
            obc = child["strategy_audit"]["online_bias_correction"]
            checks.update(
                {
                    "canonical_second_draw_present": obc["second_memory_draws"] > 0,
                    "canonical_extra_backbone_forward_present": (
                        obc["additional_backbone_forwards"]
                        == obc["second_memory_draws"]
                    ),
                }
            )
        streams[stream] = {"checks": checks, "passes": all(checks.values())}
    return {"streams": streams, "passes": all(row["passes"] for row in streams.values())}


def paired_effect(parent: dict, child: dict) -> dict[str, float]:
    return {
        "accuracy_delta_pp": 100.0
        * (float(child["mean_accuracy"]) - float(parent["mean_accuracy"])),
        "mean_forgetting_improvement_pp": 100.0
        * (float(parent["mean_forgetting"]) - float(child["mean_forgetting"])),
        "worst_forgetting_improvement_pp": 100.0
        * (float(parent["worst_forgetting"]) - float(child["worst_forgetting"])),
        "pss_delta_pp": 100.0 * (float(child["pss"]) - float(parent["pss"])),
        "pss_relative_change_percent": 100.0
        * (
            (float(child["pss"]) - float(parent["pss"])) / float(parent["pss"])
            if float(parent["pss"]) != 0.0
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = {method: {} for method in METHODS}
    protocol = []
    for method in METHODS:
        for seed in args.seeds:
            rows[method][str(seed)] = stream_pair(args.root, seed, method)
            for benchmark in (HARD, BLURRY):
                payload = load_summary(args.root, benchmark, seed, method)
                checks = protocol_checks(payload, benchmark, seed, method)
                protocol.append(
                    {
                        "method": method,
                        "seed": seed,
                        "benchmark": benchmark,
                        "checks": checks,
                        "passes": all(checks.values()),
                    }
                )

    aggregate = {}
    for method in METHODS:
        method_rows = [rows[method][str(seed)] for seed in args.seeds]
        aggregate[method] = {
            "accuracy_percent": summarize(
                [100.0 * row["mean_accuracy"] for row in method_rows]
            ),
            "mean_forgetting_percent": summarize(
                [100.0 * row["mean_forgetting"] for row in method_rows]
            ),
            "worst_forgetting_percent": summarize(
                [100.0 * row["worst_forgetting"] for row in method_rows]
            ),
            "pss_points": summarize([100.0 * row["pss"] for row in method_rows]),
        }

    comparisons = {}
    for name, parent_method, child_method in (
        ("prba_vs_layer2", LAYER2, PRBA),
        ("prba_vs_obc", OBC, PRBA),
        ("prba_vs_ceace", CEACE, PRBA),
    ):
        per_seed = {
            str(seed): paired_effect(
                rows[parent_method][str(seed)], rows[child_method][str(seed)]
            )
            for seed in args.seeds
        }
        effects = {}
        for key in next(iter(per_seed.values())):
            values = [per_seed[str(seed)][key] for seed in args.seeds]
            effects[key] = {
                **summarize(values),
                "exact_sign_flip_p_value": paired_sign_flip(values),
            }
        comparisons[name] = {
            "parent": parent_method,
            "child": child_method,
            "per_seed": per_seed,
            "aggregate": effects,
        }

    parent_paths = {
        "prba_vs_layer2": {
            str(seed): parent_path_checks(args.root, seed, PRBA)
            for seed in args.seeds
        },
        "obc_vs_layer2": {
            str(seed): parent_path_checks(args.root, seed, OBC)
            for seed in args.seeds
        },
    }
    all_protocol_checks_pass = all(row["passes"] for row in protocol)
    all_parent_path_checks_pass = all(
        audit["passes"]
        for comparison in parent_paths.values()
        for audit in comparison.values()
    )
    report = {
        "stage": "D134 unified corrected Tiny ImageNet matrix",
        "status": (
            "completed_protocol_checked"
            if all_protocol_checks_pass and all_parent_path_checks_pass
            else "audit_failure"
        ),
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "seeds": args.seeds,
        "methods": list(METHODS),
        "expected_semantic_normalization": {
            "mean": EXPECTED_MEAN,
            "std": EXPECTED_STD,
            "note": "Not applicable to CE-ACE, which has no semantic-memory encoder.",
        },
        "all_protocol_checks_pass": all_protocol_checks_pass,
        "all_parent_path_checks_pass": all_parent_path_checks_pass,
        "protocol_checks": protocol,
        "parent_path_checks": parent_paths,
        "per_method": rows,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "interpretation_note": (
            "Five paired seeds support bounded directional and uncertainty reporting. "
            "The shared-GPU runtime files are not used for timing claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# D134 unified corrected Tiny ImageNet matrix",
        "",
        f"Protocol checks pass: {all_protocol_checks_pass}",
        f"Parent-path checks pass: {all_parent_path_checks_pass}",
        "",
        "## Absolute results",
        "",
        "| Method | Accuracy (%) | Mean AF (%) | Worst AF (%) | PSS (pp) |",
        "|:--|--:|--:|--:|--:|",
    ]
    for method in METHODS:
        item = aggregate[method]
        lines.append(
            f"| {LABELS[method]} | "
            f"{item['accuracy_percent']['mean']:.3f} +/- {item['accuracy_percent']['sample_std']:.3f} | "
            f"{item['mean_forgetting_percent']['mean']:.3f} +/- {item['mean_forgetting_percent']['sample_std']:.3f} | "
            f"{item['worst_forgetting_percent']['mean']:.3f} +/- {item['worst_forgetting_percent']['sample_std']:.3f} | "
            f"{item['pss_points']['mean']:.3f} +/- {item['pss_points']['sample_std']:.3f} |"
        )

    lines += [
        "",
        "## Paired PRBA contrasts",
        "",
        "| Contrast | Accuracy delta (pp) | Mean AF improvement (pp) | Worst AF improvement (pp) | PSS delta (pp) |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name in ("prba_vs_layer2", "prba_vs_obc", "prba_vs_ceace"):
        item = comparisons[name]
        aggregate_effect = item["aggregate"]
        label = f"PRBA - {LABELS[item['parent']]}"
        lines.append(
            f"| {label} | {aggregate_effect['accuracy_delta_pp']['mean']:+.3f} | "
            f"{aggregate_effect['mean_forgetting_improvement_pp']['mean']:+.3f} | "
            f"{aggregate_effect['worst_forgetting_improvement_pp']['mean']:+.3f} | "
            f"{aggregate_effect['pss_delta_pp']['mean']:+.3f} |"
        )

    lines += [
        "",
        "The JSON file contains per-seed values, Student-t and bootstrap 95% intervals, "
        "exact sign-flip p-values, protocol checks, and parent-path audits.",
    ]
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
