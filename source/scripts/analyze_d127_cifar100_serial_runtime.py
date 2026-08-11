"""Analyze dedicated serial runtime evidence for C, OBC, and PRBA."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from analyze_d124_cifar100_prba_obc_completion import (
    BLURRY,
    HARD,
    LAYER2,
    OBC,
    PRBA,
    load_summary,
)


METHODS = (LAYER2, OBC, PRBA)


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
    }


def runtime(root: Path, seed: int, benchmark: str, method: str) -> float:
    return float(
        (root / "runtime" / f"seed{seed}_{benchmark}_{method}.seconds")
        .read_text()
        .strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.root / "runtime" / "execution_manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    gpu_contaminated = [
        row for row in manifest if int(row["gpu1_busy_before"]) != 0
    ]
    failed = [row for row in manifest if row["status"] != "completed"]

    per_method = {}
    for method in METHODS:
        paired_seconds = []
        reported_seconds = []
        peak_memory = []
        for seed in args.seeds:
            paired_seconds.append(
                sum(
                    runtime(args.root, seed, benchmark, method)
                    for benchmark in (HARD, BLURRY)
                )
            )
            summaries = [
                load_summary(args.root, seed, benchmark, method)
                for benchmark in (HARD, BLURRY)
            ]
            reported_seconds.append(
                sum(float(row["run_metadata"]["elapsed_seconds"]) for row in summaries)
            )
            peak_memory.append(
                max(
                    int(row["run_metadata"]["peak_cuda_memory_bytes"])
                    for row in summaries
                )
            )
        per_method[method] = {
            "paired_stream_wall_seconds": summarize(paired_seconds),
            "reported_elapsed_seconds": summarize(reported_seconds),
            "peak_cuda_memory_bytes": summarize(
                [float(value) for value in peak_memory]
            ),
            "per_seed_paired_stream_wall_seconds": dict(
                zip(map(str, args.seeds), paired_seconds)
            ),
        }

    comparisons = {}
    for method in (OBC, PRBA):
        changes = []
        for seed in args.seeds:
            parent = per_method[LAYER2]["per_seed_paired_stream_wall_seconds"][
                str(seed)
            ]
            child = per_method[method]["per_seed_paired_stream_wall_seconds"][
                str(seed)
            ]
            changes.append(child / parent - 1.0)
        comparisons[f"{method}_vs_layer2"] = {
            "runtime_relative_change": summarize(changes)
        }

    report = {
        "stage": "D127 dedicated serial runtime/profile evidence",
        "seeds": args.seeds,
        "execution": {
            "server": "server2",
            "single_gpu": True,
            "gpu_index": 1,
            "single_worker": True,
            "serial": True,
            "latin_method_order": True,
            "full_metrics_disabled": True,
            "memory_trace_signature_disabled": True,
        },
        "per_method": per_method,
        "comparisons": comparisons,
        "audit": {
            "manifest_rows": len(manifest),
            "failed_rows": len(failed),
            "gpu1_contaminated_rows": len(gpu_contaminated),
            "runtime_authoritative": not failed
            and not gpu_contaminated
            and len(manifest) == 18,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

