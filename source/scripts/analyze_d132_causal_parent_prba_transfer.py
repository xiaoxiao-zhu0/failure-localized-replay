"""Analyze PRBA transfer from Persistent-SRRD to the plain Causal ER-ACE parent."""

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
PARENT = "causal_er_ace"
CANDIDATE = "causal_er_ace_prequential_arbitration_1"


def load_summary(root: Path, seed: int, benchmark: str, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def deployment_pair(root: Path, seed: int) -> dict:
    hard = deployment_stream_metrics(load_summary(root, seed, HARD, CANDIDATE))
    blurry = deployment_stream_metrics(
        load_summary(root, seed, BLURRY, CANDIDATE)
    )
    gaps = {key: blurry[key] - hard[key] for key in hard}
    return {
        "hard": hard,
        "blurry": blurry,
        "blurry_minus_hard": gaps,
        "pss": sum(abs(value) for value in gaps.values()),
        "mean_accuracy": mean(
            row["final_validation_accuracy"] for row in (hard, blurry)
        ),
        "worst_forgetting": max(
            hard["final_validation_forgetting"],
            blurry["final_validation_forgetting"],
        ),
        "evaluation_source": "deployment_head_per_class_audit_history",
    }


def delta(parent: dict, child: dict) -> dict:
    parent_mean_af = mean(
        parent[stream]["final_validation_forgetting"]
        for stream in ("hard", "blurry")
    )
    child_mean_af = mean(
        child[stream]["final_validation_forgetting"]
        for stream in ("hard", "blurry")
    )
    return {
        "mean_accuracy_delta": child["mean_accuracy"] - parent["mean_accuracy"],
        "mean_average_forgetting_improvement": parent_mean_af - child_mean_af,
        "worst_stream_average_forgetting_improvement": (
            parent["worst_forgetting"] - child["worst_forgetting"]
        ),
        "pss_relative_change": (
            child["pss"] / parent["pss"] - 1.0 if parent["pss"] > 0 else 0.0
        ),
        "stream_delta": {
            stream: {
                "accuracy_delta": (
                    child[stream]["final_validation_accuracy"]
                    - parent[stream]["final_validation_accuracy"]
                ),
                "forgetting_delta": (
                    child[stream]["final_validation_forgetting"]
                    - parent[stream]["final_validation_forgetting"]
                ),
            }
            for stream in ("hard", "blurry")
        },
    }


def mechanism(root: Path, seed: int) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load_summary(root, seed, benchmark, PARENT)
        child = load_summary(root, seed, benchmark, CANDIDATE)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        child_audit = child["strategy_audit"]
        child_hash = child_audit["memory_trace_determinism"]
        calibration = child_audit["replay_feature_dual_head_calibration"]
        arbitration = child_audit["risk_budgeted_head_arbitration"]
        history = calibration["evaluation_bias_audit"]["history"]
        partition = sum(
            int(arbitration[key])
            for key in (
                "zero_alpha_batches",
                "interior_alpha_batches",
                "one_alpha_batches",
            )
        )
        alpha = float(arbitration["deployment_alpha"])
        checks = {
            "plain_causal_parent_declared": calibration["parent_method"] == PARENT
            and arbitration["parent_method"] == PARENT,
            "parent_model_exact": parent_hash["final_model_hash"]
            == child_hash["final_model_hash"],
            "parent_memory_exact": parent_hash["final_memory_hash"]
            == child_hash["final_memory_hash"],
            "parent_replay_exact": parent_hash["replay_index_hash"]
            == child_hash["replay_index_hash"],
            "calibration_and_arbitration_active": calibration["enabled"]
            and arbitration["enabled"]
            and calibration["deployment_uses_risk_budgeted_blend"],
            "updates_match": calibration["calibration_updates"]
            == calibration["replay_calls"]
            == arbitration["arbitration_batches"]
            and arbitration["arbitration_batches"] > 0,
            "valid_alpha": 0.0 <= alpha <= 1.0,
            "batch_partition_exact": partition == arbitration["arbitration_batches"],
            "prequential_test_then_train": arbitration["prequential_test_then_train"]
            and arbitration["arbitration_is_recorded_before_main_optimizer_step"]
            and arbitration[
                "arbitration_is_recorded_before_current_calibration_update"
            ],
            "same_update_leakage_absent": arbitration[
                "same_update_replay_fit_and_budget_evidence_are_separated"
            ]
            and not arbitration["post_update_replay_reuse_leakage"],
            "fixed_low_cost_solver": arbitration[
                "bisection_steps_for_numerical_solution"
            ]
            == 8
            and abs(arbitration["numerical_alpha_resolution"] - 2.0**-8)
            < 1e-12,
            "no_fixed_or_tuned_alpha": not arbitration["fixed_or_tuned_alpha"],
            "convex_objective_nonworsening": arbitration[
                "mean_joint_ce_improvement"
            ]
            >= -1e-6,
            "two_head_forwards_per_batch": arbitration[
                "additional_head_only_forwards"
            ]
            == 2 * arbitration["arbitration_batches"],
            "no_extra_replay_or_backbone": arbitration["additional_replay_draws"]
            == 0
            and arbitration["additional_backbone_forwards"] == 0,
            "evaluation_history_observed": calibration["evaluation_bias_audit"][
                "enabled"
            ]
            and bool(history),
            "no_nonfinite_updates": calibration["nonfinite_skips"] == 0
            and arbitration["nonfinite_skips"] == 0,
            "main_training_contract_unchanged": calibration[
                "main_model_parameters_are_not_modified_by_calibration"
            ]
            and calibration["main_optimizer_state_is_not_modified_by_calibration"]
            and calibration["memory_and_replay_identities_are_unchanged"],
            "causal_evidence_only": arbitration[
                "uses_arrived_current_and_replay_batches_only"
            ]
            and not arbitration["uses_validation_pss_future_data_or_task_boundary"],
        }
        streams[stream] = {
            "checks": checks,
            "passes": all(checks.values()),
            "deployment_alpha": alpha,
            "arbitration_batches": arbitration["arbitration_batches"],
        }
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


def aggregate(rows: dict[str, dict]) -> dict:
    metrics = (
        "mean_accuracy_delta",
        "mean_average_forgetting_improvement",
        "worst_stream_average_forgetting_improvement",
        "pss_relative_change",
    )
    result = {
        metric: summarize([float(row[metric]) for row in rows.values()])
        for metric in metrics
    }
    result["stream_delta"] = {
        stream: {
            metric: summarize(
                [float(row["stream_delta"][stream][metric]) for row in rows.values()]
            )
            for metric in ("accuracy_delta", "forgetting_delta")
        }
        for stream in ("hard", "blurry")
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for seed in args.seeds:
        parent = pair(args.root, HARD, BLURRY, seed, PARENT)
        child = deployment_pair(args.root, seed)
        rows[str(seed)] = {
            "parent": parent,
            "child": child,
            "delta": delta(parent, child),
            "mechanism": mechanism(args.root, seed),
        }
    deltas = {seed: row["delta"] for seed, row in rows.items()}
    summary = aggregate(deltas)
    required = math.ceil(2 * len(args.seeds) / 3)
    forgetting_direction_count = sum(
        row["mean_average_forgetting_improvement"] > 0.0
        for row in deltas.values()
    )
    accuracy_direction_count = sum(
        row["mean_accuracy_delta"] > 0.0 for row in deltas.values()
    )
    checks = {
        "all_mechanism_audits_pass": all(
            row["mechanism"]["passes"] for row in rows.values()
        ),
        "mean_accuracy_within_minus_0_5pp": summary["mean_accuracy_delta"][
            "mean"
        ]
        >= -0.005,
        "mean_forgetting_improves_at_least_0_5pp": summary[
            "mean_average_forgetting_improvement"
        ]["mean"]
        >= 0.005,
        "forgetting_improves_in_at_least_two_thirds_seeds": (
            forgetting_direction_count >= required
        ),
        "worst_stream_forgetting_does_not_worsen": summary[
            "worst_stream_average_forgetting_improvement"
        ]["mean"]
        >= 0.0,
        "mean_pss_worsening_at_most_10pct": summary["pss_relative_change"][
            "mean"
        ]
        <= 0.10,
    }
    for stream in ("hard", "blurry"):
        checks[f"{stream}_accuracy_within_minus_0_75pp"] = summary[
            "stream_delta"
        ][stream]["accuracy_delta"]["mean"] >= -0.0075
        checks[f"{stream}_forgetting_worsening_at_most_0_25pp"] = summary[
            "stream_delta"
        ][stream]["forgetting_delta"]["mean"] <= 0.0025

    report = {
        "stage": "D132 causal-parent PRBA transfer confirmation",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "methods": {"parent": PARENT, "candidate": CANDIDATE},
        "execution": {
            "server": "server2",
            "physical_gpu": 1,
            "seed_workers": 2,
            "runtime_authoritative": False,
        },
        "research_question": (
            "Does the same prequential risk-budget arbitration remain useful "
            "when the Persistent-SRRD/Layer-2 parent is removed?"
        ),
        "per_seed": rows,
        "aggregate": summary,
        "gate": {
            "required_direction_count": required,
            "forgetting_positive_seed_count": forgetting_direction_count,
            "accuracy_positive_seed_count": accuracy_direction_count,
            "checks": checks,
            "passes": all(checks.values()),
            "interpretation": (
                "cross-parent transfer supported"
                if all(checks.values())
                else "mechanism or benefit is parent-dependent under this protocol"
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
