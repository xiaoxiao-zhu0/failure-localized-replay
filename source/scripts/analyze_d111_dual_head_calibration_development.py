"""Analyze fresh-seed development of replay-feature dual-head calibration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from analyze_d15_strong_baseline_taskification import pair

from analyze_d102_srrd_loss_transfer_development import (
    BLURRY,
    CEACE,
    HARD,
    LAYER2,
    aggregate_delta,
    catastrophic_reasons,
    load_summary,
    runtime,
)


CANDIDATE = "persistent_srrd_dual_head_calibration_1"


def deployment_stream_metrics(summary: dict) -> dict[str, float]:
    """Reconstruct final deployment-head metrics from audit-only histories.

    The generic classwise audit follows the unchanged surrogate training head.
    A dual-head method must instead be judged from the deployment head that is
    actually returned during evaluation.
    """

    history = summary["strategy_audit"][
        "replay_feature_dual_head_calibration"
    ]["evaluation_bias_audit"]["history"]
    class_accuracy_history: dict[str, list[float]] = {}
    for event in history:
        per_class = event["dual_head_metrics"]["deployment"]["per_class"]
        for label, row in per_class.items():
            accuracy = row["accuracy"]
            if accuracy is not None:
                class_accuracy_history.setdefault(str(label), []).append(
                    float(accuracy)
                )
    if not class_accuracy_history:
        raise ValueError("deployment-head per-class evaluation history is empty")
    final_accuracy = [values[-1] for values in class_accuracy_history.values()]
    final_forgetting = [
        max(values) - values[-1] for values in class_accuracy_history.values()
    ]
    return {
        "final_validation_accuracy": mean(final_accuracy),
        "final_validation_forgetting": mean(final_forgetting),
    }


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


def deployment_method_delta(root: Path, seed: int, parent: str) -> dict:
    parent_row = pair(root, HARD, BLURRY, seed, parent)
    child_row = deployment_pair(root, seed)
    parent_runtime = sum(
        runtime(root, seed, benchmark, parent) for benchmark in (HARD, BLURRY)
    )
    child_runtime = sum(
        runtime(root, seed, benchmark, CANDIDATE)
        for benchmark in (HARD, BLURRY)
    )
    parent_mean_af = mean(
        parent_row[stream]["final_validation_forgetting"]
        for stream in ("hard", "blurry")
    )
    child_mean_af = mean(
        child_row[stream]["final_validation_forgetting"]
        for stream in ("hard", "blurry")
    )
    return {
        "parent": parent_row,
        "child": child_row,
        "evaluation_source": "deployment_head_per_class_audit_history",
        "parent_mean_average_forgetting": parent_mean_af,
        "child_mean_average_forgetting": child_mean_af,
        "mean_average_forgetting_improvement": parent_mean_af - child_mean_af,
        "worst_stream_average_forgetting_improvement": parent_row[
            "worst_forgetting"
        ]
        - child_row["worst_forgetting"],
        "mean_accuracy_delta": child_row["mean_accuracy"]
        - parent_row["mean_accuracy"],
        "pss_relative_change": (
            child_row["pss"] / parent_row["pss"] - 1.0
            if parent_row["pss"] > 0
            else 0.0
        ),
        "runtime_relative_change": child_runtime / parent_runtime - 1.0,
        "stream_delta": {
            stream: {
                "accuracy_delta": child_row[stream]["final_validation_accuracy"]
                - parent_row[stream]["final_validation_accuracy"],
                "forgetting_delta": child_row[stream][
                    "final_validation_forgetting"
                ]
                - parent_row[stream]["final_validation_forgetting"],
            }
            for stream in ("hard", "blurry")
        },
    }


def mechanism_audit(root: Path, seed: int) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load_summary(root, seed, benchmark, LAYER2)
        summary = load_summary(root, seed, benchmark, CANDIDATE)
        replay = summary["strategy_audit"]["persistent_srrd_replay"]
        calibration = summary["strategy_audit"][
            "replay_feature_dual_head_calibration"
        ]
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        candidate_hash = summary["strategy_audit"]["memory_trace_determinism"]
        replay_comparison = calibration["replay_head_comparison"]
        evaluation_audit = calibration["evaluation_bias_audit"]
        evaluation_history = evaluation_audit["history"]
        evaluated_samples = sum(
            int(row["dual_head_metrics"]["sample_count"])
            for row in evaluation_history
        )
        checks = {
            "layer2_one_swap_enabled": replay["enabled"]
            and replay["max_swaps_per_batch"] == 1,
            "replacement_layer3_enabled": calibration["enabled"],
            "fixed_learning_rate_scale": abs(
                calibration["calibration_lr_scale"] - 1.0
            )
            < 1e-12,
            "fixed_label_smoothing": abs(calibration["label_smoothing"] - 0.1)
            < 1e-12,
            "calibration_updates_every_replay_call": calibration[
                "calibration_updates"
            ]
            == calibration["replay_calls"]
            and calibration["calibration_updates"] > 0,
            "deployment_head_is_active": calibration[
                "deployment_uses_calibration_head"
            ],
            "deployment_head_differs_from_training_head": calibration[
                "deployment_head_hash"
            ]
            != calibration["training_head_hash"],
            "class_balanced_memory_only_objective": calibration[
                "calibration_uses_replay_only"
            ]
            and calibration[
                "calibration_loss_is_class_balanced_over_present_classes"
            ],
            "replay_features_are_reused": calibration[
                "reuses_layer2_replay_features"
            ],
            "no_additional_replay_draws": calibration[
                "additional_replay_draws"
            ]
            == 0,
            "no_additional_backbone_forwards": calibration[
                "additional_backbone_forwards"
            ]
            == 0,
            "training_head_audit_matches_calibration_samples": int(
                replay_comparison["sample_count"]
            )
            == int(calibration["calibration_samples"])
            and int(calibration["calibration_samples"]) > 0,
            "training_head_forwards_match_replay_calls": calibration[
                "additional_training_calibration_head_forwards"
            ]
            == calibration["replay_calls"],
            "training_feature_hook_is_active": calibration[
                "training_feature_hook_registered"
            ]
            and calibration["disabled_path_registers_no_training_feature_hook"],
            "evaluation_bias_audit_is_observed": evaluation_audit["enabled"]
            and bool(evaluation_history)
            and evaluated_samples > 0
            and calibration["additional_evaluation_calibration_head_forwards"]
            > 0,
            "evaluation_audit_is_noncontrolling": evaluation_audit[
                "uses_eval_labels_for_audit_only"
            ]
            and evaluation_audit[
                "uses_training_phase_boundary_for_old_new_audit_only"
            ]
            and evaluation_audit[
                "audit_does_not_control_training_sampling_or_prediction"
            ]
            and evaluation_audit[
                "future_unobserved_target_classes_are_excluded"
            ],
            "calibration_cost_accounting_is_present": calibration[
                "head_update_host_dispatch_seconds"
            ]
            >= 0.0
            and calibration[
                "run_level_peak_cuda_memory_is_recorded_in_run_metadata"
            ],
            "no_nonfinite_updates": calibration["nonfinite_skips"] == 0,
            "main_model_hash_exact": parent_hash["final_model_hash"]
            == candidate_hash["final_model_hash"],
            "memory_hash_exact": parent_hash["final_memory_hash"]
            == candidate_hash["final_memory_hash"],
            "replay_index_hash_exact": parent_hash["replay_index_hash"]
            == candidate_hash["replay_index_hash"],
            "main_training_is_unchanged": calibration[
                "main_model_parameters_are_not_modified_by_calibration"
            ]
            and calibration[
                "main_optimizer_state_is_not_modified_by_calibration"
            ]
            and calibration["memory_and_replay_identities_are_unchanged"],
            "causal_inputs_only": not calibration[
                "uses_task_id_boundary_validation_pss_or_future_data"
            ],
        }
        streams[stream] = {
            "persistent_srrd_replay": replay,
            "replay_feature_dual_head_calibration": calibration,
            "checks": checks,
            "passes": all(checks.values()),
        }
    return {
        "streams": streams,
        "passes": all(row["passes"] for row in streams.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for seed in args.seeds:
        candidate_vs_layer2 = deployment_method_delta(args.root, seed, LAYER2)
        candidate_vs_ceace = deployment_method_delta(args.root, seed, CEACE)
        candidate_vs_layer2["mechanism"] = mechanism_audit(args.root, seed)
        candidate_vs_layer2["catastrophic_reasons"] = catastrophic_reasons(
            candidate_vs_layer2
        )
        rows[str(seed)] = {
            "candidate_vs_layer2": candidate_vs_layer2,
            "candidate_vs_ceace": candidate_vs_ceace,
        }

    layer2_rows = {
        seed: row["candidate_vs_layer2"] for seed, row in rows.items()
    }
    ceace_rows = {seed: row["candidate_vs_ceace"] for seed, row in rows.items()}
    layer2_aggregate = aggregate_delta(layer2_rows)
    ceace_aggregate = aggregate_delta(ceace_rows)
    required = math.ceil(2 * len(args.seeds) / 3)
    direction_count = sum(
        row["mean_average_forgetting_improvement"] > 0.0
        for row in layer2_rows.values()
    )
    catastrophic_seed_count = sum(
        bool(row["catastrophic_reasons"]) for row in layer2_rows.values()
    )
    calibrations = [
        row["mechanism"]["streams"][stream][
            "replay_feature_dual_head_calibration"
        ]
        for row in layer2_rows.values()
        for stream in ("hard", "blurry")
    ]
    total_calls = sum(int(row["replay_calls"]) for row in calibrations)
    total_updates = sum(int(row["calibration_updates"]) for row in calibrations)
    total_samples = sum(int(row["calibration_samples"]) for row in calibrations)
    checks = {
        "mean_af_improves_at_least_0_5pp_vs_layer2": layer2_aggregate[
            "mean_average_forgetting_improvement"
        ]
        >= 0.005,
        "worst_stream_af_improves_at_least_0_25pp_vs_layer2": layer2_aggregate[
            "worst_stream_average_forgetting_improvement"
        ]
        >= 0.0025,
        "mean_af_improves_in_at_least_two_thirds_seeds": direction_count
        >= required,
        "mean_accuracy_within_minus_0_5pp_vs_layer2": layer2_aggregate[
            "mean_accuracy_delta"
        ]
        >= -0.005,
        "mean_pss_worsening_at_most_10pct_vs_layer2": layer2_aggregate[
            "pss_relative_change"
        ]
        <= 0.10,
        "runtime_overhead_at_most_20pct_vs_layer2": layer2_aggregate[
            "runtime_relative_change"
        ]
        <= 0.20,
        "mean_accuracy_at_least_plus_1pp_vs_ceace": ceace_aggregate[
            "mean_accuracy_delta"
        ]
        >= 0.01,
        "pss_at_least_15pct_better_than_ceace": ceace_aggregate[
            "pss_relative_change"
        ]
        <= -0.15,
        "mean_af_within_plus_0_2pp_of_ceace": ceace_aggregate[
            "mean_average_forgetting_improvement"
        ]
        >= -0.002,
        "no_catastrophic_seed": catastrophic_seed_count == 0,
        "all_mechanism_audits_pass": all(
            row["mechanism"]["passes"] for row in layer2_rows.values()
        ),
        "calibration_updates_match_replay_calls": total_updates == total_calls,
        "calibration_samples_are_observed": total_samples > 0,
    }
    for stream in ("hard", "blurry"):
        checks[
            f"{stream}_accuracy_within_minus_0_75pp_vs_layer2"
        ] = layer2_aggregate["stream_delta"][stream]["accuracy_delta"] >= -0.0075
        checks[
            f"{stream}_forgetting_worsening_at_most_0_2pp_vs_layer2"
        ] = layer2_aggregate["stream_delta"][stream]["forgetting_delta"] <= 0.002

    passed = all(checks.values())
    report = {
        "stage": "D111 replay-feature dual-head calibration development",
        "status": "fresh_seed_replacement_layer3_development",
        "seeds": args.seeds,
        "methods": {
            "overall_baseline": CEACE,
            "layer2_parent": LAYER2,
            "replacement_layer3_candidate": CANDIDATE,
        },
        "literature_basis": {
            "online_bias_correction": "Chrysakis and Moens, ICLR 2023",
            "classifier_decoupling": "Kang et al., ICLR 2020",
            "rejected_alternatives": [
                "GEM/MGDA-style optimizer geometry repeats the D79-negative route",
                "OCAR changes the full optimizer and adds K-FAC state",
                "NCM/generative classifiers require a different representation objective",
            ],
        },
        "fixed_mechanism": {
            "source_evidence": "D109 accepted only 6 of 5919 same-head repairs; strict same-head feasibility saturated",
            "layer2": "confidence-gated instantaneous one-swap correction",
            "layer3": "separate deployment head updated from reused Layer-2 replay features",
            "calibration_lr_scale": 1.0,
            "label_smoothing": 0.1,
            "class_balanced_present_class_loss": True,
            "additional_replay_draws": 0,
            "additional_backbone_forwards": 0,
            "hyperparameter_search": False,
            "performance_evaluation_source": (
                "deployment_head_per_class_audit_history"
            ),
            "surrogate_classwise_audit_is_not_used_for_candidate_metrics": True,
        },
        "per_seed": rows,
        "aggregate": {
            "candidate_vs_layer2": layer2_aggregate,
            "candidate_vs_ceace": ceace_aggregate,
            "total_replay_calls": total_calls,
            "total_calibration_updates": total_updates,
            "total_calibration_samples": total_samples,
        },
        "gate": {
            "required_direction_count": required,
            "actual_direction_count": direction_count,
            "catastrophic_seed_count": catastrophic_seed_count,
            "checks": checks,
            "passes": passed,
            "selected_branch": (
                "authorize_dual_head_calibration_fresh_confirmation"
                if passed
                else "delete_or_revise_dual_head_calibration"
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
