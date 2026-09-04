#!/usr/bin/env python3
"""Aggregate the CORe50 hard A/B/C/D confirmation chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


METHODS = {
    "A": "causal_er_ace",
    "B": "semantic_proto_hybrid_75_25",
    "C": "persistent_srrd_selective_swap_1",
    "D": "persistent_srrd_prequential_arbitration_1",
}


def metric(metrics: dict, prefix: str) -> float:
    for key, value in metrics.items():
        if key.startswith(prefix):
            return float(value)
    raise KeyError(prefix)


def optional(mapping: dict, key: str):
    value = mapping.get(key)
    return value if isinstance(value, (int, float)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for label, method in METHODS.items():
            path = args.root / "core50" / f"seed_{seed}" / method / "summary.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["last_metrics"]
            metadata = payload.get("run_metadata", {})
            audit = payload.get("strategy_audit", {})
            replay = audit.get("persistent_srrd_replay", {})
            calibration = audit.get("replay_feature_dual_head_calibration", {})
            arbitration = audit.get("risk_budgeted_head_arbitration", {})
            row = {
                "seed": seed,
                "label": label,
                "method": method,
                "summary": str(path),
                "accuracy": metric(metrics, "Top1_Acc_Stream/eval_phase/test_stream"),
                "stream_forgetting": float(metrics["StreamForgetting/eval_phase/test_stream"]),
                "stream_bwt": float(metrics["StreamBWT/eval_phase/test_stream"]),
                "elapsed_seconds": optional(metadata, "elapsed_seconds"),
                "peak_cuda_memory_bytes": optional(metadata, "peak_cuda_memory_bytes"),
                "swap_count": optional(replay, "swap_count"),
                "replay_calls": optional(calibration, "replay_calls"),
                "calibration_updates": optional(calibration, "calibration_updates"),
                "deployment_alpha": optional(arbitration, "deployment_alpha"),
                "additional_training_head_forwards": optional(
                    calibration, "additional_training_calibration_head_forwards"),
                "additional_evaluation_head_forwards": optional(
                    calibration, "additional_evaluation_calibration_head_forwards"),
                "pss": None,
            }
            row["finite"] = all(math.isfinite(float(row[k])) for k in (
                "accuracy", "stream_forgetting", "stream_bwt"))
            rows.append(row)

    by_key = {(row["seed"], row["label"]): row for row in rows}
    comparisons = []
    for seed in args.seeds:
        for child, parent in (("B", "A"), ("C", "B"), ("D", "C"), ("D", "A")):
            left = by_key[(seed, child)]
            right = by_key[(seed, parent)]
            comparisons.append({
                "seed": seed,
                "child": child,
                "parent": parent,
                "accuracy_delta_pp": 100.0 * (left["accuracy"] - right["accuracy"]),
                "stream_forgetting_delta_pp": 100.0 * (
                    left["stream_forgetting"] - right["stream_forgetting"]),
                "stream_bwt_delta_pp": 100.0 * (left["stream_bwt"] - right["stream_bwt"]),
            })

    output = {
        "protocol": {
            "benchmark": "core50",
            "stream": "hard",
            "seeds": args.seeds,
            "methods": METHODS,
            "lr": 0.01,
            "epochs": 5,
            "gpu": 1,
            "worker_policy": "serial",
        },
        "rows": rows,
        "comparisons": comparisons,
        "all_finite": all(row["finite"] for row in rows),
        "pss_note": "PSS requires paired hard/blurry streams and is intentionally not computed in this hard-only confirmation.",
    }
    args.root.joinpath("analysis").mkdir(parents=True, exist_ok=True)
    args.root.joinpath("analysis/d135_abcd_3seed_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")

    lines = [
        "# D135 CORe50 hard A/B/C/D 3-seed confirmation",
        "",
        "All metrics finite: " + str(output["all_finite"]),
        "",
        "| Seed | A/B/C/D | Accuracy (%) | Stream forgetting | Stream BWT | Time (s) | Peak CUDA (GiB) |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        for label in METHODS:
            row = by_key[(seed, label)]
            peak_gib = (
                f"{row['peak_cuda_memory_bytes'] / (1024**3):.3f}"
                if row["peak_cuda_memory_bytes"] is not None else "NA"
            )
            lines.append(
                f"| {seed} | {label} | {100*row['accuracy']:.3f} | "
                f"{row['stream_forgetting']:.6f} | {row['stream_bwt']:.6f} | "
                f"{row['elapsed_seconds'] if row['elapsed_seconds'] is not None else 'NA'} | "
                f"{peak_gib} |"
            )
    lines += ["", "| Seed | Contrast | Acc delta (pp) | Forgetting delta (pp) | BWT delta (pp) |",
              "|---:|:---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(
            f"| {row['seed']} | {row['child']}-{row['parent']} | "
            f"{row['accuracy_delta_pp']:+.3f} | {row['stream_forgetting_delta_pp']:+.3f} | "
            f"{row['stream_bwt_delta_pp']:+.3f} |"
        )
    args.root.joinpath("analysis/d135_abcd_3seed_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
