"""Analyze the CIFAR-100 same-seed PRBA/OBC completion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d111_dual_head_calibration_development import deployment_stream_metrics
from analyze_d15_strong_baseline_taskification import pair


HARD = "split_cifar100"
BLURRY = "equal_exposure_blurry_cifar100"
CEACE = "causal_er_ace"
LAYER2 = "persistent_srrd_selective_swap_1"
PRBA = "persistent_srrd_prequential_arbitration_1"
OBC = "persistent_srrd_obc_1"
DERPP = "derpp"
MIR = "mir"


def load_summary(root: Path, seed: int, benchmark: str, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def deployment_pair(root: Path, seed: int, method: str) -> dict:
    hard = deployment_stream_metrics(load_summary(root, seed, HARD, method))
    blurry = deployment_stream_metrics(load_summary(root, seed, BLURRY, method))
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


def mechanism(parent_root: Path, root: Path, seed: int, method: str) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        parent = load_summary(root, seed, benchmark, LAYER2)
        child = load_summary(root, seed, benchmark, method)
        parent_hash = parent["strategy_audit"]["memory_trace_determinism"]
        child_hash = child["strategy_audit"]["memory_trace_determinism"]
        calibration = child["strategy_audit"]["replay_feature_dual_head_calibration"]
        checks = {
            "reused_parent_training_model_exact": (
                parent_hash["final_model_hash"] == child_hash["final_model_hash"]
            ),
            "reused_parent_memory_exact": (
                parent_hash["final_memory_hash"] == child_hash["final_memory_hash"]
            ),
            "reused_parent_replay_indices_exact": (
                parent_hash["replay_index_hash"] == child_hash["replay_index_hash"]
            ),
            "deployment_head_active": calibration["deployment_uses_calibration_head"],
            "no_nonfinite_calibration_update": calibration["nonfinite_skips"] == 0,
        }
        if method == PRBA:
            arbitration = child["strategy_audit"]["risk_budgeted_head_arbitration"]
            checks.update(
                {
                    "prequential_test_then_train": arbitration[
                        "prequential_test_then_train"
                    ],
                    "no_same_update_reuse": not arbitration[
                        "post_update_replay_reuse_leakage"
                    ],
                    "no_extra_replay_draw": arbitration["additional_replay_draws"] == 0,
                    "no_extra_backbone_forward": (
                        arbitration["additional_backbone_forwards"] == 0
                    ),
                }
            )
        else:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    absolute = {
        LAYER2: {
            str(seed): pair(args.root, HARD, BLURRY, seed, LAYER2)
            for seed in args.seeds
        },
        CEACE: {
            str(seed): pair(args.parent_root, HARD, BLURRY, seed, CEACE)
            for seed in args.seeds
        },
        DERPP: {
            str(seed): pair(args.external_root, HARD, BLURRY, seed, DERPP)
            for seed in args.seeds
        },
        MIR: {
            str(seed): pair(args.external_root, HARD, BLURRY, seed, MIR)
            for seed in args.seeds
        },
        PRBA: {
            str(seed): deployment_pair(args.root, seed, PRBA) for seed in args.seeds
        },
        OBC: {
            str(seed): deployment_pair(args.root, seed, OBC) for seed in args.seeds
        },
    }

    comparisons = {}
    for name, parent_method, child_method in (
        ("prba_vs_layer2", LAYER2, PRBA),
        ("obc_vs_layer2", LAYER2, OBC),
        ("prba_vs_obc", OBC, PRBA),
        ("prba_vs_ceace", CEACE, PRBA),
        ("prba_vs_derpp", DERPP, PRBA),
        ("prba_vs_mir", MIR, PRBA),
    ):
        rows = {}
        for seed in args.seeds:
            row = delta(
                absolute[parent_method][str(seed)],
                absolute[child_method][str(seed)],
            )
            if parent_method == LAYER2 and child_method in (PRBA, OBC):
                row["mechanism"] = mechanism(
                    args.root, args.root, seed, child_method
                )
            rows[str(seed)] = row
        comparisons[name] = {
            "parent": parent_method,
            "child": child_method,
            "per_seed": rows,
            "aggregate": aggregate(rows),
            "positive_mean_af_seed_count": sum(
                row["mean_average_forgetting_improvement"] > 0.0
                for row in rows.values()
            ),
        }

    report = {
        "stage": "D124 CIFAR-100 same-seed PRBA/OBC completion",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "execution": {
            "server": "server2",
            "single_gpu": True,
            "gpu_index": 1,
            "seed_workers": 2,
            "runtime_authoritative": False,
        },
        "reuse": {
            "current_layer2_parent_root": str(args.root),
            "ceace_baseline_root": str(args.parent_root),
            "external_root": str(args.external_root),
            "d105_layer2_reuse_rejected_missing_hash_audit": True,
            "derpp_and_mir_have_no_full_metrics": True,
        },
        "absolute": absolute,
        "comparisons": comparisons,
        "mechanism_and_parent_reuse_pass": all(
            row["mechanism"]["passes"]
            for name in ("prba_vs_layer2", "obc_vs_layer2")
            for row in comparisons[name]["per_seed"].values()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
