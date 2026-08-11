"""Analyze the frozen Tiny ImageNet PRBA cross-dataset completion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import (
    deployment_stream_metrics,
)
from analyze_d15_strong_baseline_taskification import pair


HARD = "split_tinyimagenet"
BLURRY = "equal_exposure_blurry_tinyimagenet"
LAYER2 = "persistent_srrd_selective_swap_1"
PRBA = "persistent_srrd_prequential_arbitration_1"
OBC = "persistent_srrd_obc_1"
CEACE = "causal_er_ace"
METHODS = (LAYER2, PRBA, OBC, CEACE)
DEPLOYMENT_HEAD_METHODS = (PRBA, OBC)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def runtime(root: Path, seed: int, benchmark: str, method: str) -> float:
    path = root / "runtime" / f"seed{seed}_{benchmark}_{method}.seconds"
    return float(path.read_text(encoding="utf-8").strip())


def load_summary(root: Path, seed: int, benchmark: str, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def absolute_row(root: Path, seed: int, method: str) -> dict:
    summaries = [
        load_summary(root, seed, benchmark, method)
        for benchmark in (HARD, BLURRY)
    ]
    if method in DEPLOYMENT_HEAD_METHODS:
        hard = deployment_stream_metrics(summaries[0])
        blurry = deployment_stream_metrics(summaries[1])
        gaps = {key: blurry[key] - hard[key] for key in hard}
        row = {
            "hard": hard,
            "blurry": blurry,
            "blurry_minus_hard": gaps,
            "pss": sum(abs(value) for value in gaps.values()),
            "mean_accuracy": mean(
                stream["final_validation_accuracy"]
                for stream in (hard, blurry)
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
    return {
        **row,
        "serial_wall_seconds": sum(
            runtime(root, seed, benchmark, method)
            for benchmark in (HARD, BLURRY)
        ),
        "reported_elapsed_seconds": sum(
            float(summary["run_metadata"]["elapsed_seconds"])
            for summary in summaries
        ),
        "peak_cuda_memory_bytes": max(
            int(summary["run_metadata"]["peak_cuda_memory_bytes"])
            for summary in summaries
        ),
    }


def delta(parent: dict, child: dict) -> dict:
    parent_mean_af = 0.5 * (
        float(parent["hard"]["final_validation_forgetting"])
        + float(parent["blurry"]["final_validation_forgetting"])
    )
    child_mean_af = 0.5 * (
        float(child["hard"]["final_validation_forgetting"])
        + float(child["blurry"]["final_validation_forgetting"])
    )
    return {
        "mean_accuracy_delta": float(child["mean_accuracy"]) - float(parent["mean_accuracy"]),
        "mean_average_forgetting_improvement": parent_mean_af - child_mean_af,
        "worst_stream_average_forgetting_improvement": (
            float(parent["worst_forgetting"]) - float(child["worst_forgetting"])
        ),
        "pss_relative_change": (
            (float(child["pss"]) - float(parent["pss"])) / float(parent["pss"])
            if float(parent["pss"]) != 0.0
            else 0.0
        ),
        "runtime_relative_change": (
            (float(child["serial_wall_seconds"]) - float(parent["serial_wall_seconds"]))
            / float(parent["serial_wall_seconds"])
        ),
        "stream_delta": {
            stream: {
                "accuracy_delta": (
                    float(child[stream]["final_validation_accuracy"])
                    - float(parent[stream]["final_validation_accuracy"])
                ),
                "forgetting_delta": (
                    float(child[stream]["final_validation_forgetting"])
                    - float(parent[stream]["final_validation_forgetting"])
                ),
            }
            for stream in ("hard", "blurry")
        },
    }


def mechanism(root: Path, seed: int, method: str) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load_summary(root, seed, benchmark, LAYER2)
        child = load_summary(root, seed, benchmark, method)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        child_hash = child["strategy_audit"]["memory_trace_determinism"]
        calibration = child["strategy_audit"]["replay_feature_dual_head_calibration"]
        checks = {
            "parent_training_model_exact": parent_hash["final_model_hash"] == child_hash["final_model_hash"],
            "parent_memory_exact": parent_hash["final_memory_hash"] == child_hash["final_memory_hash"],
            "parent_replay_indices_exact": parent_hash["replay_index_hash"] == child_hash["replay_index_hash"],
            "deployment_head_active": calibration["deployment_uses_calibration_head"],
            "no_nonfinite_calibration_update": calibration["nonfinite_skips"] == 0,
        }
        if method == PRBA:
            arbitration = child["strategy_audit"]["risk_budgeted_head_arbitration"]
            checks.update(
                {
                    "prequential_order": arbitration["prequential_test_then_train"],
                    "no_same_update_reuse": not arbitration["post_update_replay_reuse_leakage"],
                    "no_extra_replay_or_backbone": (
                        arbitration["additional_replay_draws"] == 0
                        and arbitration["additional_backbone_forwards"] == 0
                    ),
                }
            )
        elif method == OBC:
            obc = child["strategy_audit"]["online_bias_correction"]
            checks.update(
                {
                    "canonical_second_draw_present": obc["second_memory_draws"] > 0,
                    "canonical_extra_backbone_forward_present": (
                        obc["additional_backbone_forwards"] == obc["second_memory_draws"]
                    ),
                }
            )
        streams[stream] = {"checks": checks, "passes": all(checks.values())}
    return {
        "streams": streams,
        "passes": all(row["passes"] for row in streams.values()),
    }


def aggregate_deltas(rows: list[dict]) -> dict:
    keys = (
        "mean_accuracy_delta",
        "mean_average_forgetting_improvement",
        "worst_stream_average_forgetting_improvement",
        "pss_relative_change",
        "runtime_relative_change",
    )
    aggregate = {
        key: summarize([float(row[key]) for row in rows]) for key in keys
    }
    aggregate["stream_delta"] = {
        stream: {
            metric: summarize(
                [float(row["stream_delta"][stream][metric]) for row in rows]
            )
            for metric in ("accuracy_delta", "forgetting_delta")
        }
        for stream in ("hard", "blurry")
    }
    return aggregate


def catastrophic_reasons(row: dict) -> list[str]:
    reasons = []
    if float(row["mean_accuracy_delta"]) < -0.03:
        reasons.append("mean_accuracy_below_minus_3pp")
    for stream in ("hard", "blurry"):
        if float(row["stream_delta"][stream]["accuracy_delta"]) < -0.03:
            reasons.append(f"{stream}_accuracy_below_minus_3pp")
        checks = row["mechanism"]["streams"][stream]["checks"]
        if not checks["no_nonfinite_calibration_update"]:
            reasons.append(f"{stream}_nonfinite_calibration_update")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    per_method: dict[str, object] = {}
    absolute: dict[str, dict[str, dict]] = {}
    for method in METHODS:
        rows = {str(seed): absolute_row(args.root, seed, method) for seed in args.seeds}
        absolute[method] = rows
        per_method[method] = {
            "per_seed": rows,
            "aggregate": {
                key: summarize([float(row[key]) for row in rows.values()])
                for key in ("pss", "mean_accuracy", "worst_forgetting")
            },
        }

    comparisons = {}
    for name, parent_method, child_method in (
        ("prba_vs_layer2", LAYER2, PRBA),
        ("obc_vs_layer2", LAYER2, OBC),
        ("prba_vs_obc", OBC, PRBA),
        ("prba_vs_ceace", CEACE, PRBA),
    ):
        rows = []
        per_seed = {}
        for seed in args.seeds:
            row = delta(absolute[parent_method][str(seed)], absolute[child_method][str(seed)])
            if child_method in (PRBA, OBC) and parent_method == LAYER2:
                row["mechanism"] = mechanism(args.root, seed, child_method)
            if child_method == PRBA and parent_method == LAYER2:
                row["catastrophic_reasons"] = catastrophic_reasons(row)
            per_seed[str(seed)] = row
            rows.append(row)
        comparisons[name] = {
            "parent": parent_method,
            "child": child_method,
            "per_seed": per_seed,
            "aggregate": aggregate_deltas(rows),
        }

    prba = comparisons["prba_vs_layer2"]
    prba_rows = list(prba["per_seed"].values())
    aggregate = prba["aggregate"]
    catastrophic_seed_count = sum(
        bool(row["catastrophic_reasons"]) for row in prba_rows
    )
    checks = {
        "mean_af_improves_at_least_1pp": aggregate["mean_average_forgetting_improvement"]["mean"] >= 0.01,
        "worst_af_improves_at_least_0_5pp": aggregate["worst_stream_average_forgetting_improvement"]["mean"] >= 0.005,
        "mean_af_improves_in_at_least_4_of_5_seeds": sum(
            float(row["mean_average_forgetting_improvement"]) > 0.0 for row in prba_rows
        ) >= 4,
        "mean_accuracy_within_minus_1pp": aggregate["mean_accuracy_delta"]["mean"] >= -0.01,
        "hard_accuracy_within_minus_1_5pp": (
            aggregate["stream_delta"]["hard"]["accuracy_delta"]["mean"]
            >= -0.015
        ),
        "blurry_accuracy_within_minus_1_5pp": (
            aggregate["stream_delta"]["blurry"]["accuracy_delta"]["mean"]
            >= -0.015
        ),
        "mean_pss_worsening_at_most_20pct": aggregate["pss_relative_change"]["mean"] <= 0.20,
        "no_catastrophic_seed": catastrophic_seed_count == 0,
        "all_mechanism_audits_pass": all(
            bool(row["mechanism"]["passes"]) for row in prba_rows
        ),
    }

    report = {
        "stage": "D123 Tiny ImageNet PRBA cross-dataset completion",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "methods": list(METHODS),
        "execution": {
            "server": "server2",
            "single_gpu": True,
            "gpu_index": 1,
            "single_worker": True,
            "serial": True,
        },
        "per_method": per_method,
        "comparisons": comparisons,
        "gate": {
            "catastrophic_seed_count": catastrophic_seed_count,
            "checks": checks,
            "passes": all(checks.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
