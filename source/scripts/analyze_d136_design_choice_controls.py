"""Summarize D136 random-arbitration and no-Wilson controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import (
    deployment_stream_metrics,
)
from analyze_d15_strong_baseline_taskification import metrics as training_metrics


HARD = "split_cifar100"
BLURRY = "equal_exposure_blurry_cifar100"
STREAMS = (("hard", HARD), ("blurry", BLURRY))
RANDOM = "persistent_srrd_prequential_random"
NO_WILSON_LAYER2 = "persistent_srrd_selective_swap_no_wilson"
NO_WILSON_PRBA = "persistent_srrd_prequential_no_wilson"
FULL_PRBA = "persistent_srrd_prequential_arbitration_1"
METHODS = (RANDOM, NO_WILSON_LAYER2, NO_WILSON_PRBA)


def load_summary(root: Path, seed: int, benchmark: str, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def confidence(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def pair(root: Path, seed: int, method: str) -> dict[str, object]:
    rows = {}
    for stream, benchmark in STREAMS:
        summary_path = root / benchmark / f"seed_{seed}" / method / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if method == NO_WILSON_LAYER2:
            rows[stream] = training_metrics(summary_path)
        else:
            rows[stream] = deployment_stream_metrics(summary)
    gap = {
        key: rows["blurry"][key] - rows["hard"][key]
        for key in rows["hard"]
    }
    return {
        **rows,
        "blurry_minus_hard": gap,
        "pss": sum(abs(value) for value in gap.values()),
        "mean_accuracy": mean(
            rows[stream]["final_validation_accuracy"]
            for stream in ("hard", "blurry")
        ),
        "mean_forgetting": mean(
            rows[stream]["final_validation_forgetting"]
            for stream in ("hard", "blurry")
        ),
        "worst_forgetting": max(
            rows[stream]["final_validation_forgetting"]
            for stream in ("hard", "blurry")
        ),
    }


def reference_pair(d119: dict, seed: int, child: bool) -> dict[str, object]:
    row = d119["per_method"][FULL_PRBA]["per_seed"][str(seed)]
    source = row["child" if child else "parent"]
    return {
        **source,
        "mean_forgetting": mean(
            source[stream]["final_validation_forgetting"]
            for stream in ("hard", "blurry")
        ),
    }


def delta(child: dict, parent: dict) -> dict[str, object]:
    return {
        "mean_accuracy_delta": child["mean_accuracy"] - parent["mean_accuracy"],
        "mean_forgetting_improvement": (
            parent["mean_forgetting"] - child["mean_forgetting"]
        ),
        "worst_forgetting_improvement": (
            parent["worst_forgetting"] - child["worst_forgetting"]
        ),
        "pss_delta": child["pss"] - parent["pss"],
        "pss_relative_change": (
            child["pss"] / parent["pss"] - 1.0
            if parent["pss"] > 0.0
            else 0.0
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


def aggregate_pairs(rows: dict[str, dict]) -> dict[str, dict]:
    return {
        "accuracy": confidence([row["mean_accuracy"] for row in rows.values()]),
        "mean_af": confidence([row["mean_forgetting"] for row in rows.values()]),
        "worst_af": confidence([row["worst_forgetting"] for row in rows.values()]),
        "pss": confidence([row["pss"] for row in rows.values()]),
    }


def aggregate_deltas(rows: dict[str, dict]) -> dict[str, dict]:
    keys = (
        "mean_accuracy_delta",
        "mean_forgetting_improvement",
        "worst_forgetting_improvement",
        "pss_delta",
        "pss_relative_change",
    )
    return {
        key: confidence([row[key] for row in rows.values()])
        for key in keys
    }


def mechanism_audit(root: Path, seed: int, method: str) -> dict[str, object]:
    streams = {}
    for stream, benchmark in STREAMS:
        summary = load_summary(root, seed, benchmark, method)
        replay = summary["strategy_audit"]["persistent_srrd_replay"]
        checks = {
            "one_replacement_maximum": replay["max_swaps_per_batch"] == 1,
            "lagged_causal_signal": replay["signal_is_read_before_current_batch_update"],
            "no_current_or_future_selection_loss": not replay[
                "uses_current_or_future_loss_for_selection"
            ],
            "no_task_validation_or_future_data": not replay[
                "uses_task_id_gradient_validation_pss_or_future_data"
            ],
            "main_sampler_receives_no_extra_draws": replay[
                "main_sampler_rng_receives_no_extra_draws"
            ],
        }
        if method == RANDOM:
            arbitration = summary["strategy_audit"][
                "risk_budgeted_head_arbitration"
            ]
            checks.update(
                {
                    "wilson_gate_retained": replay["wilson_gate_enabled"],
                    "random_uniform_alpha": arbitration["alpha_source"]
                    == "independent fixed-seed uniform random batch coefficient"
                    and arbitration["random_arbitration_distribution"]
                    == "Uniform(0, 1)",
                    "cumulative_mean_retained": arbitration["aggregation_rule"]
                    == "online cumulative mean of random coefficients",
                    "prequential_order": arbitration["prequential_test_then_train"],
                    "no_same_update_reuse": not arbitration[
                        "post_update_replay_reuse_leakage"
                    ],
                    "no_extra_replay_or_backbone": arbitration[
                        "additional_replay_draws"
                    ]
                    == 0
                    and arbitration["additional_backbone_forwards"] == 0,
                    "no_nonfinite_arbitration": arbitration["nonfinite_skips"] == 0,
                }
            )
        else:
            checks.update(
                {
                    "wilson_gate_removed_only": not replay["wilson_gate_enabled"]
                    and replay["confidence_interval"]
                    == "point_estimate_positive_gap",
                    "residual_controller_retained": replay["controller_mode"]
                    == "confidence_gated_instantaneous_residual"
                    and not replay["debt_carries_across_batches"],
                    "abstention_retained": replay["supports_abstention"],
                }
            )
        if method == NO_WILSON_PRBA:
            parent = load_summary(root, seed, benchmark, NO_WILSON_LAYER2)
            parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
            child_hash = summary["strategy_audit"]["memory_trace_determinism"]
            arbitration = summary["strategy_audit"][
                "risk_budgeted_head_arbitration"
            ]
            calibration = summary["strategy_audit"][
                "replay_feature_dual_head_calibration"
            ]
            checks.update(
                {
                    "no_wilson_parent_model_exact": parent_hash[
                        "final_model_hash"
                    ]
                    == child_hash["final_model_hash"],
                    "no_wilson_parent_memory_exact": parent_hash[
                        "final_memory_hash"
                    ]
                    == child_hash["final_memory_hash"],
                    "no_wilson_parent_replay_exact": parent_hash[
                        "replay_index_hash"
                    ]
                    == child_hash["replay_index_hash"],
                    "deployment_head_active": calibration[
                        "deployment_uses_calibration_head"
                    ],
                    "prequential_order": arbitration["prequential_test_then_train"],
                    "no_same_update_reuse": not arbitration[
                        "post_update_replay_reuse_leakage"
                    ],
                    "no_extra_replay_or_backbone": arbitration[
                        "additional_replay_draws"
                    ]
                    == 0
                    and arbitration["additional_backbone_forwards"] == 0,
                    "no_nonfinite_updates": arbitration["nonfinite_skips"] == 0
                    and calibration["nonfinite_skips"] == 0,
                }
            )
        streams[stream] = {
            "checks": checks,
            "passes": all(checks.values()),
            "swap_count": replay["swap_count"],
            "replay_calls": replay["replay_calls"],
            "confidence_rejected_batches": replay[
                "confidence_rejected_batches"
            ],
        }
    return {
        "streams": streams,
        "passes": all(row["passes"] for row in streams.values()),
    }


def markdown(report: dict) -> str:
    labels = {
        RANDOM: "Random arbitration",
        NO_WILSON_LAYER2: "Layer 2 without Wilson gate",
        NO_WILSON_PRBA: "PRBA without Wilson gate",
    }
    lines = [
        "# D136 Design-Choice Controls",
        "",
        "| Variant | Accuracy | Mean AF | Worst AF | PSS |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["per_method"][method]["aggregate"]
        values = []
        for key in ("accuracy", "mean_af", "worst_af", "pss"):
            values.append(
                f'{100 * row[key]["mean"]:.3f} +/- '
                f'{100 * row[key]["sample_std"]:.3f}'
            )
        lines.append(f'| {labels[method]} | ' + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "All values are mean +/- sample SD over seeds 252--254 after "
            "combining the paired hard and blurry streams by the registered "
            "metric definitions.",
            "",
            "## Audit",
            "",
        ]
    )
    for method in METHODS:
        status = "PASS" if report["per_method"][method]["all_audits_pass"] else "FAIL"
        lines.append(f'- {labels[method]}: {status}')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d119-summary", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    d119 = json.loads(args.d119_summary.read_text(encoding="utf-8"))
    per_method = {}
    for method in METHODS:
        pairs = {str(seed): pair(args.root, seed, method) for seed in args.seeds}
        versus_layer2 = {
            str(seed): delta(pairs[str(seed)], reference_pair(d119, seed, False))
            for seed in args.seeds
        }
        versus_full = {
            str(seed): delta(pairs[str(seed)], reference_pair(d119, seed, True))
            for seed in args.seeds
        }
        audits = {
            str(seed): mechanism_audit(args.root, seed, method)
            for seed in args.seeds
        }
        entry = {
            "per_seed": pairs,
            "aggregate": aggregate_pairs(pairs),
            "versus_standard_layer2": {
                "per_seed": versus_layer2,
                "aggregate": aggregate_deltas(versus_layer2),
            },
            "versus_full_prba": {
                "per_seed": versus_full,
                "aggregate": aggregate_deltas(versus_full),
            },
            "audit": audits,
            "all_audits_pass": all(row["passes"] for row in audits.values()),
        }
        if method == NO_WILSON_PRBA:
            parent_pairs = {
                str(seed): pair(args.root, seed, NO_WILSON_LAYER2)
                for seed in args.seeds
            }
            versus_parent = {
                str(seed): delta(pairs[str(seed)], parent_pairs[str(seed)])
                for seed in args.seeds
            }
            entry["versus_no_wilson_layer2"] = {
                "per_seed": versus_parent,
                "aggregate": aggregate_deltas(versus_parent),
            }
        per_method[method] = entry

    report = {
        "stage": "D136 random-arbitration and no-Wilson design controls",
        "status": "completed_descriptive_three_seed_control",
        "seeds": args.seeds,
        "methods": list(METHODS),
        "reference": str(args.d119_summary),
        "per_method": per_method,
        "all_audits_pass": all(
            row["all_audits_pass"] for row in per_method.values()
        ),
        "interpretation_boundary": (
            "Three paired seeds support design interpretation, not universal "
            "optimality or large-sample significance claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
