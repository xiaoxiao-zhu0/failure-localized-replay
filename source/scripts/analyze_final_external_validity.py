"""Analyze the final Layer-2/Full external-validity confirmation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import deployment_stream_metrics
from analyze_d15_strong_baseline_taskification import pair


LAYER2 = "persistent_srrd_selective_swap_1"
FULL = "persistent_srrd_prequential_arbitration_1"


def summarize(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def load_summary(root: Path, benchmark: str, seed: int, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def runtime(root: Path, benchmark: str, seed: int, method: str) -> float | None:
    path = root / "runtime" / f"seed{seed}_{benchmark}_{method}.seconds"
    return float(path.read_text(encoding="utf-8").strip()) if path.exists() else None


def absolute_row(root: Path, hard: str, blurry: str, seed: int, method: str) -> dict:
    hard_summary = load_summary(root, hard, seed, method)
    blurry_summary = load_summary(root, blurry, seed, method)
    if method == FULL:
        hard_metrics = deployment_stream_metrics(hard_summary)
        blurry_metrics = deployment_stream_metrics(blurry_summary)
        gaps = {
            key: blurry_metrics[key] - hard_metrics[key] for key in hard_metrics
        }
        row = {
            "hard": hard_metrics,
            "blurry": blurry_metrics,
            "pss": sum(abs(value) for value in gaps.values()),
            "mean_accuracy": mean(
                stream["final_validation_accuracy"]
                for stream in (hard_metrics, blurry_metrics)
            ),
            "worst_forgetting": max(
                hard_metrics["final_validation_forgetting"],
                blurry_metrics["final_validation_forgetting"],
            ),
            "evaluation_source": "deployment_head_per_class_audit_history",
        }
    else:
        row = pair(root, hard, blurry, seed, method)
        row["evaluation_source"] = "training_head_classwise_audit_history"
    wall = [runtime(root, benchmark, seed, method) for benchmark in (hard, blurry)]
    row["shared_gpu_wall_seconds"] = (
        sum(value for value in wall if value is not None)
        if all(value is not None for value in wall)
        else None
    )
    return row


def delta(parent: dict, child: dict) -> dict:
    parent_af = mean(
        parent[stream]["final_validation_forgetting"] for stream in ("hard", "blurry")
    )
    child_af = mean(
        child[stream]["final_validation_forgetting"] for stream in ("hard", "blurry")
    )
    return {
        "accuracy_delta": child["mean_accuracy"] - parent["mean_accuracy"],
        "mean_forgetting_improvement": parent_af - child_af,
        "worst_forgetting_improvement": (
            parent["worst_forgetting"] - child["worst_forgetting"]
        ),
        "pss_relative_change": (
            (child["pss"] - parent["pss"]) / parent["pss"]
            if parent["pss"] != 0.0
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


def mechanism_and_protocol(
    root: Path, hard: str, blurry: str, seed: int, expected_family: str
) -> dict:
    streams = {}
    for stream, benchmark in (("hard", hard), ("blurry", blurry)):
        parent = load_summary(root, benchmark, seed, LAYER2)
        child = load_summary(root, benchmark, seed, FULL)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        child_hash = child["strategy_audit"]["memory_trace_determinism"]
        memory = child["strategy_audit"]["semantic_representative_memory"]
        arbitration = child["strategy_audit"]["risk_budgeted_head_arbitration"]
        checks = {
            "dataset_family_matches": memory["dataset_family"] == expected_family,
            "normalization_is_recorded": (
                len(memory["input_normalization_mean"]) == 3
                and len(memory["input_normalization_std"]) == 3
            ),
            "parent_training_model_exact": (
                parent_hash["final_model_hash"] == child_hash["final_model_hash"]
            ),
            "parent_memory_exact": (
                parent_hash["final_memory_hash"] == child_hash["final_memory_hash"]
            ),
            "parent_replay_indices_exact": (
                parent_hash["replay_index_hash"] == child_hash["replay_index_hash"]
            ),
            "prequential_order": arbitration["prequential_test_then_train"],
            "no_same_update_reuse": not arbitration["post_update_replay_reuse_leakage"],
            "no_extra_replay_or_backbone": (
                arbitration["additional_replay_draws"] == 0
                and arbitration["additional_backbone_forwards"] == 0
            ),
        }
        streams[stream] = {"checks": checks, "passes": all(checks.values())}
    return {"streams": streams, "passes": all(row["passes"] for row in streams.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hard", required=True)
    parser.add_argument("--blurry", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-family", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    absolute = {LAYER2: {}, FULL: {}}
    comparisons = {}
    for seed in args.seeds:
        for method in (LAYER2, FULL):
            absolute[method][str(seed)] = absolute_row(
                args.root, args.hard, args.blurry, seed, method
            )
        row = delta(absolute[LAYER2][str(seed)], absolute[FULL][str(seed)])
        row["audit"] = mechanism_and_protocol(
            args.root, args.hard, args.blurry, seed, args.expected_family
        )
        comparisons[str(seed)] = row

    metric_names = (
        "accuracy_delta",
        "mean_forgetting_improvement",
        "worst_forgetting_improvement",
        "pss_relative_change",
    )
    report = {
        "stage": args.stage,
        "status": "completed_frozen_external_validity_evidence",
        "benchmarks": {"hard": args.hard, "blurry": args.blurry},
        "seeds": args.seeds,
        "methods": [LAYER2, FULL],
        "absolute": absolute,
        "full_vs_layer2": {
            "per_seed": comparisons,
            "aggregate": {
                key: summarize([row[key] for row in comparisons.values()])
                for key in metric_names
            },
            "pareto_seed_count": sum(
                row["accuracy_delta"] > 0.0
                and row["mean_forgetting_improvement"] > 0.0
                for row in comparisons.values()
            ),
            "all_protocol_audits_pass": all(
                row["audit"]["passes"] for row in comparisons.values()
            ),
        },
        "execution_note": (
            "Scientific metrics are valid for paired stability assessment; "
            "wall-clock timing is non-authoritative when workers share one GPU."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
