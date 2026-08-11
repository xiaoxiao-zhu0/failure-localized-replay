"""Analyze PRBA/OBC transfer across CIFAR-100 memory budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_d124_cifar100_prba_obc_completion import (
    BLURRY,
    HARD,
    LAYER2,
    OBC,
    PRBA,
    aggregate,
    delta,
    deployment_pair,
    load_summary,
)
from analyze_d15_strong_baseline_taskification import pair


COMMON_METADATA = (
    "benchmark",
    "model",
    "seed",
    "n_experiences",
    "train_epochs",
    "train_mb_size",
    "eval_mb_size",
    "mem_size",
    "lr",
    "momentum",
    "validation_fraction",
    "historical_reference",
    "deterministic",
)


def audit(root: Path, c_root: Path, memory: int, seed: int) -> dict:
    streams = {}
    for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
        c = load_summary(c_root, seed, benchmark, LAYER2)
        prba = load_summary(root / f"memory_{memory}", seed, benchmark, PRBA)
        obc = load_summary(root / f"memory_{memory}", seed, benchmark, OBC)
        prba_hash = prba["strategy_audit"]["memory_trace_determinism"]
        obc_hash = obc["strategy_audit"]["memory_trace_determinism"]
        prba_cal = prba["strategy_audit"]["replay_feature_dual_head_calibration"]
        obc_cal = obc["strategy_audit"]["replay_feature_dual_head_calibration"]
        arbitration = prba["strategy_audit"]["risk_budgeted_head_arbitration"]
        obc_audit = obc["strategy_audit"]["online_bias_correction"]
        checks = {
            "c_prba_metadata_exact": all(
                c["run_metadata"].get(key) == prba["run_metadata"].get(key)
                for key in COMMON_METADATA
            ),
            "c_obc_metadata_exact": all(
                c["run_metadata"].get(key) == obc["run_metadata"].get(key)
                for key in COMMON_METADATA
            ),
            "prba_obc_training_model_exact": prba_hash["final_model_hash"]
            == obc_hash["final_model_hash"],
            "prba_obc_memory_exact": prba_hash["final_memory_hash"]
            == obc_hash["final_memory_hash"],
            "prba_obc_replay_indices_exact": prba_hash["replay_index_hash"]
            == obc_hash["replay_index_hash"],
            "prba_deployment_active": prba_cal["deployment_uses_calibration_head"],
            "obc_deployment_active": obc_cal["deployment_uses_calibration_head"],
            "no_nonfinite_updates": prba_cal["nonfinite_skips"] == 0
            and obc_cal["nonfinite_skips"] == 0,
            "prba_prequential": arbitration["prequential_test_then_train"],
            "prba_no_same_update_reuse": not arbitration[
                "post_update_replay_reuse_leakage"
            ],
            "prba_no_extra_replay_or_backbone": arbitration[
                "additional_replay_draws"
            ]
            == 0
            and arbitration["additional_backbone_forwards"] == 0,
            "obc_canonical_extra_path": obc_audit["second_memory_draws"] > 0
            and obc_audit["additional_backbone_forwards"]
            == obc_audit["second_memory_draws"],
        }
        streams[stream] = {"checks": checks, "passes": all(checks.values())}
    return {
        "streams": streams,
        "passes": all(row["passes"] for row in streams.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mem50-c-root", type=Path, required=True)
    parser.add_argument("--mem200-c-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_memories = {}
    all_audits = []
    for memory, c_root in ((50, args.mem50_c_root), (200, args.mem200_c_root)):
        run_root = args.root / f"memory_{memory}"
        absolute = {
            LAYER2: {
                str(seed): pair(c_root, HARD, BLURRY, seed, LAYER2)
                for seed in args.seeds
            },
            PRBA: {
                str(seed): deployment_pair(run_root, seed, PRBA)
                for seed in args.seeds
            },
            OBC: {
                str(seed): deployment_pair(run_root, seed, OBC)
                for seed in args.seeds
            },
        }
        comparisons = {}
        for name, parent, child in (
            ("prba_vs_layer2", LAYER2, PRBA),
            ("obc_vs_layer2", LAYER2, OBC),
            ("prba_vs_obc", OBC, PRBA),
        ):
            rows = {
                str(seed): delta(
                    absolute[parent][str(seed)], absolute[child][str(seed)]
                )
                for seed in args.seeds
            }
            comparisons[name] = {
                "parent": parent,
                "child": child,
                "per_seed": rows,
                "aggregate": aggregate(rows),
                "positive_mean_af_seed_count": sum(
                    row["mean_average_forgetting_improvement"] > 0.0
                    for row in rows.values()
                ),
            }
        audits = {
            str(seed): audit(args.root, c_root, memory, seed)
            for seed in args.seeds
        }
        all_audits.extend(audits.values())
        report_memories[str(memory)] = {
            "c_reuse_root": str(c_root),
            "absolute": absolute,
            "comparisons": comparisons,
            "audits": audits,
            "all_audits_pass": all(row["passes"] for row in audits.values()),
        }

    report = {
        "stage": "D126 CIFAR-100 Layer-3 memory transfer",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "memories": [50, 200],
        "execution": {
            "server": "server2",
            "single_gpu": True,
            "gpu_index": 1,
            "workers": 2,
            "runtime_authoritative": False,
        },
        "per_memory": report_memories,
        "all_config_mechanism_and_cross_child_hash_audits_pass": all(
            row["passes"] for row in all_audits
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

