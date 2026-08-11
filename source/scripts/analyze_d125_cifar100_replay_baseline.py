"""Assemble the CIFAR-100 memory-100 external baseline matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d124_cifar100_prba_obc_completion import (
    BLURRY,
    CEACE,
    DERPP,
    HARD,
    LAYER2,
    MIR,
    OBC,
    PRBA,
    aggregate,
    delta,
    deployment_pair,
)
from analyze_d15_strong_baseline_taskification import pair


REPLAY = "replay"


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--d124-root", type=Path, required=True)
    parser.add_argument("--d105-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    absolute = {
        REPLAY: {str(seed): pair(args.root, HARD, BLURRY, seed, REPLAY) for seed in args.seeds},
        LAYER2: {str(seed): pair(args.d124_root, HARD, BLURRY, seed, LAYER2) for seed in args.seeds},
        PRBA: {str(seed): deployment_pair(args.d124_root, seed, PRBA) for seed in args.seeds},
        OBC: {str(seed): deployment_pair(args.d124_root, seed, OBC) for seed in args.seeds},
        CEACE: {str(seed): pair(args.d105_root, HARD, BLURRY, seed, CEACE) for seed in args.seeds},
        DERPP: {str(seed): pair(args.external_root, HARD, BLURRY, seed, DERPP) for seed in args.seeds},
        MIR: {str(seed): pair(args.external_root, HARD, BLURRY, seed, MIR) for seed in args.seeds},
    }

    comparisons = {}
    for child in (CEACE, LAYER2, OBC, PRBA, DERPP, MIR):
        rows = {
            str(seed): delta(absolute[REPLAY][str(seed)], absolute[child][str(seed)])
            for seed in args.seeds
        }
        comparisons[f"{child}_vs_replay"] = {
            "parent": REPLAY,
            "child": child,
            "per_seed": rows,
            "aggregate": aggregate(rows),
            "positive_mean_af_seed_count": sum(
                row["mean_average_forgetting_improvement"] > 0.0
                for row in rows.values()
            ),
        }

    per_method = {
        method: {
            metric: summarize([float(row[metric]) for row in rows.values()])
            for metric in ("mean_accuracy", "worst_forgetting", "pss")
        }
        for method, rows in absolute.items()
    }
    report = {
        "stage": "D125 CIFAR-100 memory-100 external baseline matrix",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "benchmarks": {"hard": HARD, "blurry": BLURRY},
        "methods": list(absolute),
        "execution": {
            "server": "server2",
            "single_gpu": True,
            "gpu_index": 1,
            "seed_workers": 2,
            "runtime_authoritative": False,
        },
        "per_method": per_method,
        "absolute": absolute,
        "comparisons": comparisons,
        "metric_boundary": {
            "derpp_and_mir_reuse_has_no_full_metrics": True,
            "accuracy_forgetting_pss_are_available_for_all_methods": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

