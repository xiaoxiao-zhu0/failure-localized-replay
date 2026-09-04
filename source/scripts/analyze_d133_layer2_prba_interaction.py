"""Analyze the same-seed 2x2 interaction between Layer 2 and PRBA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import (
    deployment_stream_metrics,
)
from analyze_d15_strong_baseline_taskification import pair


HARD = "split_cifar100"
BLURRY = "equal_exposure_blurry_cifar100"
BASE = "causal_er_ace"
LAYER2 = "persistent_srrd_selective_swap_1"
LAYER3_ONLY = "causal_er_ace_prequential_arbitration_1"
FULL = "persistent_srrd_prequential_arbitration_1"


def load(root: Path, seed: int, benchmark: str, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def deployment_pair(root: Path, seed: int, method: str) -> dict:
    hard = deployment_stream_metrics(load(root, seed, HARD, method))
    blurry = deployment_stream_metrics(load(root, seed, BLURRY, method))
    gaps = {key: blurry[key] - hard[key] for key in hard}
    return {
        "hard": hard,
        "blurry": blurry,
        "blurry_minus_hard": gaps,
        "pss": sum(abs(value) for value in gaps.values()),
        "mean_accuracy": mean(
            row["final_validation_accuracy"] for row in (hard, blurry)
        ),
        "mean_forgetting": mean(
            row["final_validation_forgetting"] for row in (hard, blurry)
        ),
        "worst_forgetting": max(
            hard["final_validation_forgetting"],
            blurry["final_validation_forgetting"],
        ),
        "evaluation_source": "deployment_head_per_class_audit_history",
    }


def standard_pair(root: Path, seed: int, method: str) -> dict:
    row = pair(root, HARD, BLURRY, seed, method)
    row["mean_forgetting"] = mean(
        row[stream]["final_validation_forgetting"]
        for stream in ("hard", "blurry")
    )
    row["evaluation_source"] = "training_head_classwise_history"
    return row


def effect(parent: dict, child: dict) -> dict:
    return {
        "accuracy_delta": child["mean_accuracy"] - parent["mean_accuracy"],
        "forgetting_improvement": (
            parent["mean_forgetting"] - child["mean_forgetting"]
        ),
        "worst_forgetting_improvement": (
            parent["worst_forgetting"] - child["worst_forgetting"]
        ),
        "pss_relative_change": (
            child["pss"] / parent["pss"] - 1.0 if parent["pss"] > 0 else 0.0
        ),
    }


def full_mechanism(root: Path, seed: int) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load(root, seed, benchmark, LAYER2)
        full = load(root, seed, benchmark, FULL)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        audit = full["strategy_audit"]
        full_hash = audit["memory_trace_determinism"]
        calibration = audit["replay_feature_dual_head_calibration"]
        arbitration = audit["risk_budgeted_head_arbitration"]
        checks = {
            "parent_model_exact": parent_hash["final_model_hash"]
            == full_hash["final_model_hash"],
            "parent_memory_exact": parent_hash["final_memory_hash"]
            == full_hash["final_memory_hash"],
            "parent_replay_exact": parent_hash["replay_index_hash"]
            == full_hash["replay_index_hash"],
            "updates_match": calibration["calibration_updates"]
            == calibration["replay_calls"]
            == arbitration["arbitration_batches"]
            and arbitration["arbitration_batches"] > 0,
            "prequential_no_leakage": arbitration["prequential_test_then_train"]
            and not arbitration["post_update_replay_reuse_leakage"],
            "no_extra_replay_or_backbone": arbitration["additional_replay_draws"]
            == 0
            and arbitration["additional_backbone_forwards"] == 0,
            "no_nonfinite": calibration["nonfinite_skips"] == 0
            and arbitration["nonfinite_skips"] == 0,
        }
        streams[stream] = {"checks": checks, "passes": all(checks.values())}
    return {
        "streams": streams,
        "passes": all(row["passes"] for row in streams.values()),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_effect(rows: dict[str, dict]) -> dict:
    return {
        key: summarize([float(row[key]) for row in rows.values()])
        for key in (
            "accuracy_delta",
            "forgetting_improvement",
            "worst_forgetting_improvement",
            "pss_relative_change",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d132-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    d132_report = json.loads(
        (
            args.d132_root
            / "analysis"
            / "d132_causal_parent_prba_transfer.json"
        ).read_text(encoding="utf-8")
    )
    rows = {}
    for seed in args.seeds:
        cells = {
            "base": standard_pair(args.d132_root, seed, BASE),
            "layer2_only": standard_pair(args.root, seed, LAYER2),
            "layer3_only": deployment_pair(args.d132_root, seed, LAYER3_ONLY),
            "full_three_layer": deployment_pair(args.root, seed, FULL),
        }
        effects = {
            "layer2_without_layer3": effect(cells["base"], cells["layer2_only"]),
            "layer3_without_layer2": effect(cells["base"], cells["layer3_only"]),
            "layer2_with_layer3": effect(
                cells["layer3_only"], cells["full_three_layer"]
            ),
            "layer3_with_layer2": effect(
                cells["layer2_only"], cells["full_three_layer"]
            ),
            "full_vs_base": effect(cells["base"], cells["full_three_layer"]),
        }
        without = effects["layer3_without_layer2"]
        with_layer2 = effects["layer3_with_layer2"]
        preservation = (
            with_layer2["forgetting_improvement"]
            / without["forgetting_improvement"]
            if without["forgetting_improvement"] > 0.0
            else 0.0
        )
        interaction = {
            "layer3_accuracy_penalty_mitigation": (
                with_layer2["accuracy_delta"] - without["accuracy_delta"]
            ),
            "layer3_retention_gain_preservation_fraction": preservation,
            "full_pareto_dominates_base": (
                effects["full_vs_base"]["accuracy_delta"] > 0.0
                and effects["full_vs_base"]["forgetting_improvement"] > 0.0
            ),
        }
        rows[str(seed)] = {
            "cells": cells,
            "effects": effects,
            "interaction": interaction,
            "mechanism": {
                "d132_layer3_only_passes": d132_report["per_seed"][str(seed)][
                    "mechanism"
                ]["passes"],
                "full_three_layer": full_mechanism(args.root, seed),
            },
        }

    effect_names = next(iter(rows.values()))["effects"].keys()
    aggregate = {
        name: aggregate_effect(
            {seed: row["effects"][name] for seed, row in rows.items()}
        )
        for name in effect_names
    }
    interaction = {
        key: summarize(
            [float(row["interaction"][key]) for row in rows.values()]
        )
        for key in (
            "layer3_accuracy_penalty_mitigation",
            "layer3_retention_gain_preservation_fraction",
        )
    }
    required = math.ceil(2 * len(args.seeds) / 3)
    full_vs_base = aggregate["full_vs_base"]
    l3_with_l2 = aggregate["layer3_with_layer2"]
    l2_with_l3 = aggregate["layer2_with_layer3"]
    checks = {
        "all_mechanism_audits_pass": all(
            row["mechanism"]["d132_layer3_only_passes"]
            and row["mechanism"]["full_three_layer"]["passes"]
            for row in rows.values()
        ),
        "full_accuracy_improves_vs_base_at_least_1pp": full_vs_base[
            "accuracy_delta"
        ]["mean"]
        >= 0.01,
        "full_forgetting_improves_vs_base_at_least_2pp": full_vs_base[
            "forgetting_improvement"
        ]["mean"]
        >= 0.02,
        "full_pss_improves_vs_base_at_least_10pct": full_vs_base[
            "pss_relative_change"
        ]["mean"]
        <= -0.10,
        "full_pareto_dominates_base_in_two_thirds_seeds": sum(
            row["interaction"]["full_pareto_dominates_base"]
            for row in rows.values()
        )
        >= required,
        "layer3_retention_gain_with_layer2_at_least_2pp": l3_with_l2[
            "forgetting_improvement"
        ]["mean"]
        >= 0.02,
        "layer3_accuracy_cost_with_layer2_at_most_0_75pp": l3_with_l2[
            "accuracy_delta"
        ]["mean"]
        >= -0.0075,
        "layer2_recovers_at_least_1pp_with_layer3": l2_with_l3[
            "accuracy_delta"
        ]["mean"]
        >= 0.01,
        "layer3_accuracy_penalty_mitigated_at_least_0_3pp": interaction[
            "layer3_accuracy_penalty_mitigation"
        ]["mean"]
        >= 0.003,
        "at_least_half_layer3_retention_gain_preserved": interaction[
            "layer3_retention_gain_preservation_fraction"
        ]["mean"]
        >= 0.50,
    }
    report = {
        "stage": "D133 same-seed Layer2 x PRBA interaction confirmation",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "cells": {
            "00_base": BASE,
            "10_layer2_only": LAYER2,
            "01_layer3_only": LAYER3_ONLY,
            "11_full_three_layer": FULL,
        },
        "execution": {
            "server": "server2",
            "physical_gpu": 1,
            "seed_workers": 2,
            "runtime_authoritative": False,
            "reused_same_seed_d132_cells": [BASE, LAYER3_ONLY],
        },
        "per_seed": rows,
        "aggregate_effects": aggregate,
        "aggregate_interaction": interaction,
        "gate": {
            "required_direction_count": required,
            "checks": checks,
            "passes": all(checks.values()),
            "interpretation": (
                "Layer 2 and PRBA form a validated balanced interaction"
                if all(checks.values())
                else "the full model remains empirical, but the pre-specified internal interaction gate is incomplete"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
