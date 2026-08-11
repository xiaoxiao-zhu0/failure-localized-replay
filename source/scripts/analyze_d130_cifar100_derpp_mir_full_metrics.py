"""Analyze fresh full-metrics DER++/MIR CIFAR-100 baselines."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze_d128_retention_mechanism_offline import summarize


HARD = "split_cifar100"
BLURRY = "equal_exposure_blurry_cifar100"
METHODS = ("derpp", "mir")


def load(root: Path, benchmark: str, seed: int, method: str) -> dict:
    path = root / benchmark / f"seed_{seed}" / method / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory(summary: dict) -> dict[str, float]:
    stages = summary["experience_results"]
    units = []
    for experience in range(len(stages)):
        suffix = f"/Exp{experience:03d}"
        values = []
        for stage in stages[experience:]:
            matches = [
                value for key, value in stage["eval"].items()
                if key.startswith("Top1_Acc_Exp/eval_phase/test_stream/")
                and key.endswith(suffix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one accuracy metric for experience {experience}, "
                    f"found {len(matches)}"
                )
            values.append(float(matches[0]))
        acquisition = values[0]
        peak = max(values)
        final = values[-1]
        units.append({
            "acquisition": acquisition,
            "peak": peak,
            "final": final,
            "forgetting": peak - final,
            "bwt": final - acquisition,
        })
    return {
        key: statistics.mean(row[key] for row in units)
        for key in ("acquisition", "peak", "final", "forgetting", "bwt")
    }


def full_metrics_checks(summary: dict) -> dict[str, bool]:
    last = summary["last_metrics"]
    return {
        "ten_experience_results": len(summary["experience_results"]) == 10,
        "experience_forgetting_present": any(
            key.startswith("ExperienceForgetting/") for key in last
        ),
        "experience_bwt_present": any(key.startswith("ExperienceBWT/") for key in last),
        "run_metadata_present": bool(summary.get("run_metadata")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods = {}
    all_checks = []
    for method in METHODS:
        rows = []
        for seed in args.seeds:
            streams = {}
            checks = {}
            for stream, benchmark in (("hard", HARD), ("blurry", BLURRY)):
                summary = load(args.root, benchmark, seed, method)
                streams[stream] = trajectory(summary)
                checks[stream] = full_metrics_checks(summary)
                all_checks.extend(checks[stream].values())
            rows.append({
                "seed": seed,
                "streams": streams,
                "checks": checks,
                "mean": {
                    key: statistics.mean(streams[stream][key] for stream in streams)
                    for key in ("acquisition", "peak", "final", "forgetting", "bwt")
                },
                "pss": sum(
                    abs(streams["blurry"][key] - streams["hard"][key])
                    for key in ("final", "forgetting")
                ),
            })
        methods[method] = {
            "per_seed": rows,
            "aggregate": {
                key: summarize(row["mean"][key] for row in rows)
                for key in ("acquisition", "peak", "final", "forgetting", "bwt")
            },
            "pss": summarize(row["pss"] for row in rows),
        }

    report = {
        "stage": "D130 CIFAR-100 DER++/MIR full-metrics completion",
        "status": "completed_not_used_for_hyperparameter_tuning",
        "seeds": args.seeds,
        "methods": methods,
        "full_metrics_gate_passes": all(all_checks),
        "execution_boundary": {
            "server": "teacher-server2",
            "physical_gpu": 1,
            "workers": 2,
            "runtime_authoritative": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
