"""Exact small-sample paired statistics and interaction figure for D133."""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
from pathlib import Path
from statistics import mean, stdev


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def exact_bootstrap_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    bootstrap_means = [
        mean(values[index] for index in sample)
        for sample in itertools.product(range(n), repeat=n)
    ]
    return quantile(bootstrap_means, 0.025), quantile(bootstrap_means, 0.975)


def exact_sign_flip_p(values: list[float], expected_sign: int) -> tuple[float, float]:
    observed = mean(values)
    observed_abs = abs(observed)
    favorable = expected_sign * observed
    permuted = [
        mean(sign * value for sign, value in zip(signs, values))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    epsilon = 1e-15
    two_sided = sum(abs(value) >= observed_abs - epsilon for value in permuted) / len(
        permuted
    )
    directional = sum(
        expected_sign * value >= favorable - epsilon for value in permuted
    ) / len(permuted)
    return two_sided, directional


def summarize(values: list[float], expected_sign: int = 1) -> dict[str, float | int]:
    low, high = exact_bootstrap_ci(values)
    two_sided, directional = exact_sign_flip_p(values, expected_sign)
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": sample_std,
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "exact_sign_flip_p_two_sided": two_sided,
        "exact_sign_flip_p_directional": directional,
        "cohen_dz": mean(values) / sample_std if sample_std > 0.0 else 0.0,
        "positive_count": sum(value > 0.0 for value in values),
        "negative_count": sum(value < 0.0 for value in values),
        "values": values,
    }


def cell_summary(rows: list[dict], key: str) -> dict[str, float | int]:
    values = [float(row[key]) for row in rows]
    low, high = exact_bootstrap_ci(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "sample_std": stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_95_ci_low": low,
        "bootstrap_95_ci_high": high,
        "values": values,
    }


def svg_figure(report: dict, output: Path) -> None:
    cells = report["cell_statistics"]
    effects = report["paired_effects"]
    width, height = 1280, 600
    left = {"x": 82, "y": 70, "w": 610, "h": 430}
    right = {"x": 760, "y": 70, "w": 450, "h": 430}
    colors = {
        "base": "#4C78A8",
        "layer2_only": "#F58518",
        "layer3_only": "#54A24B",
        "full_three_layer": "#B279A2",
        "accuracy": "#4C78A8",
        "retention": "#E45756",
    }
    labels = {
        "base": "CE-ACE",
        "layer2_only": "Layer 2 only",
        "layer3_only": "Layer 3 only",
        "full_three_layer": "Full three-layer",
    }

    all_accuracy = []
    all_forgetting = []
    for row in cells.values():
        all_accuracy.extend(
            [row["accuracy"]["bootstrap_95_ci_low"], row["accuracy"]["bootstrap_95_ci_high"]]
        )
        all_forgetting.extend(
            [row["forgetting"]["bootstrap_95_ci_low"], row["forgetting"]["bootstrap_95_ci_high"]]
        )
    x_min = min(all_accuracy) * 100.0 - 0.6
    x_max = max(all_accuracy) * 100.0 + 0.6
    y_min = min(all_forgetting) * 100.0 - 0.8
    y_max = max(all_forgetting) * 100.0 + 0.8

    def sx(value: float) -> float:
        return left["x"] + (value * 100.0 - x_min) / (x_max - x_min) * left["w"]

    def sy(value: float) -> float:
        return left["y"] + left["h"] - (value * 100.0 - y_min) / (y_max - y_min) * left["h"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222} .title{font-size:20px;font-weight:600}.panel{font-size:16px;font-weight:600}.axis{font-size:13px}.label{font-size:13px;font-weight:600}.value{font-size:12px}.grid{stroke:#d9d9d9;stroke-width:1}.frame{fill:none;stroke:#777;stroke-width:1.2}.zero{stroke:#555;stroke-width:1.2}</style>',
        '<text x="640" y="32" text-anchor="middle" class="title">Layer-2 × PRBA interaction on CIFAR-100 (5 paired seeds)</text>',
        f'<text x="{left["x"]}" y="56" class="panel">A. Four-cell accuracy-retention working points</text>',
        f'<rect x="{left["x"]}" y="{left["y"]}" width="{left["w"]}" height="{left["h"]}" class="frame"/>',
    ]
    for index in range(6):
        x_value = x_min + index * (x_max - x_min) / 5.0
        x = left["x"] + index * left["w"] / 5.0
        svg.append(f'<line x1="{x:.1f}" y1="{left["y"]}" x2="{x:.1f}" y2="{left["y"] + left["h"]}" class="grid"/>')
        svg.append(f'<text x="{x:.1f}" y="{left["y"] + left["h"] + 23}" text-anchor="middle" class="axis">{x_value:.1f}</text>')
    for index in range(6):
        y_value = y_min + index * (y_max - y_min) / 5.0
        y = left["y"] + left["h"] - index * left["h"] / 5.0
        svg.append(f'<line x1="{left["x"]}" y1="{y:.1f}" x2="{left["x"] + left["w"]}" y2="{y:.1f}" class="grid"/>')
        svg.append(f'<text x="{left["x"] - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{y_value:.1f}</text>')
    svg.extend(
        [
            f'<text x="{left["x"] + left["w"] / 2}" y="{left["y"] + left["h"] + 52}" text-anchor="middle" class="axis">Mean final accuracy (%) — higher is better</text>',
            f'<text transform="translate(24 {left["y"] + left["h"] / 2}) rotate(-90)" text-anchor="middle" class="axis">Mean forgetting (%) — lower is better</text>',
        ]
    )
    offsets = {
        "base": (10, -12),
        "layer2_only": (-122, -12),
        "layer3_only": (10, 20),
        "full_three_layer": (10, 20),
    }
    for name, row in cells.items():
        accuracy = row["accuracy"]
        forgetting = row["forgetting"]
        x, y = sx(accuracy["mean"]), sy(forgetting["mean"])
        x_low, x_high = sx(accuracy["bootstrap_95_ci_low"]), sx(accuracy["bootstrap_95_ci_high"])
        y_low, y_high = sy(forgetting["bootstrap_95_ci_high"]), sy(forgetting["bootstrap_95_ci_low"])
        color = colors[name]
        svg.extend(
            [
                f'<line x1="{x_low:.1f}" y1="{y:.1f}" x2="{x_high:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>',
                f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="#222" stroke-width="1"/>',
                f'<text x="{x + offsets[name][0]:.1f}" y="{y + offsets[name][1]:.1f}" class="label">{html.escape(labels[name])}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{right["x"]}" y="56" class="panel">B. Layer-3 effect with and without Layer 2</text>',
            f'<rect x="{right["x"]}" y="{right["y"]}" width="{right["w"]}" height="{right["h"]}" class="frame"/>',
        ]
    )
    series = [
        (
            "Without Layer 2",
            effects["layer3_accuracy_without_layer2"],
            effects["layer3_retention_without_layer2"],
        ),
        (
            "With Layer 2",
            effects["layer3_accuracy_with_layer2"],
            effects["layer3_retention_with_layer2"],
        ),
    ]
    effect_values = []
    for _, accuracy, retention in series:
        effect_values.extend(
            [
                accuracy["bootstrap_95_ci_low"] * 100.0,
                accuracy["bootstrap_95_ci_high"] * 100.0,
                retention["bootstrap_95_ci_low"] * 100.0,
                retention["bootstrap_95_ci_high"] * 100.0,
            ]
        )
    effect_min = min(effect_values) - 0.8
    effect_max = max(effect_values) + 0.8

    def ey(value: float) -> float:
        return right["y"] + right["h"] - (value * 100.0 - effect_min) / (effect_max - effect_min) * right["h"]

    zero_y = ey(0.0)
    svg.append(f'<line x1="{right["x"]}" y1="{zero_y:.1f}" x2="{right["x"] + right["w"]}" y2="{zero_y:.1f}" class="zero"/>')
    for index in range(6):
        value = effect_min + index * (effect_max - effect_min) / 5.0
        y = right["y"] + right["h"] - index * right["h"] / 5.0
        svg.append(f'<text x="{right["x"] - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.1f}</text>')
    centers = [right["x"] + right["w"] * 0.28, right["x"] + right["w"] * 0.72]
    bar_width = 62
    for center, (context, accuracy, retention) in zip(centers, series):
        for offset, metric, color in (
            (-36, accuracy, colors["accuracy"]),
            (36, retention, colors["retention"]),
        ):
            value = metric["mean"]
            top = min(ey(value), zero_y)
            height_bar = abs(zero_y - ey(value))
            x = center + offset - bar_width / 2
            svg.append(f'<rect x="{x:.1f}" y="{top:.1f}" width="{bar_width}" height="{height_bar:.1f}" fill="{color}" opacity="0.88"/>')
            ci_low = ey(metric["bootstrap_95_ci_high"])
            ci_high = ey(metric["bootstrap_95_ci_low"])
            svg.extend(
                [
                    f'<line x1="{center + offset:.1f}" y1="{ci_low:.1f}" x2="{center + offset:.1f}" y2="{ci_high:.1f}" stroke="#222" stroke-width="1.5"/>',
                    f'<line x1="{center + offset - 7:.1f}" y1="{ci_low:.1f}" x2="{center + offset + 7:.1f}" y2="{ci_low:.1f}" stroke="#222" stroke-width="1.5"/>',
                    f'<line x1="{center + offset - 7:.1f}" y1="{ci_high:.1f}" x2="{center + offset + 7:.1f}" y2="{ci_high:.1f}" stroke="#222" stroke-width="1.5"/>',
                    f'<text x="{center + offset:.1f}" y="{ey(value) - 9 if value >= 0 else ey(value) + 18:.1f}" text-anchor="middle" class="value">{value * 100.0:+.2f}</text>',
                ]
            )
        svg.append(f'<text x="{center:.1f}" y="{right["y"] + right["h"] + 28}" text-anchor="middle" class="axis">{html.escape(context)}</text>')
    svg.extend(
        [
            f'<text transform="translate({right["x"] - 55} {right["y"] + right["h"] / 2}) rotate(-90)" text-anchor="middle" class="axis">Effect (percentage points)</text>',
            f'<rect x="{right["x"] + 55}" y="{height - 50}" width="14" height="14" fill="{colors["accuracy"]}"/><text x="{right["x"] + 76}" y="{height - 38}" class="axis">Accuracy delta</text>',
            f'<rect x="{right["x"] + 235}" y="{height - 50}" width="14" height="14" fill="{colors["retention"]}"/><text x="{right["x"] + 256}" y="{height - 38}" class="axis">Forgetting improvement</text>',
            f'<text x="{right["x"] + right["w"] / 2}" y="{height - 10}" text-anchor="middle" class="axis">Penalty mitigation: +{effects["accuracy_penalty_mitigation"]["mean"] * 100.0:.2f} pp; retention preserved: {effects["retention_preservation_fraction"]["mean"] * 100.0:.1f}%</text>',
            '</svg>',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    per_seed = source["per_seed"]
    cell_names = ("base", "layer2_only", "layer3_only", "full_three_layer")
    cell_statistics = {}
    for cell_name in cell_names:
        rows = [row["cells"][cell_name] for row in per_seed.values()]
        cell_statistics[cell_name] = {
            "accuracy": cell_summary(rows, "mean_accuracy"),
            "forgetting": cell_summary(rows, "mean_forgetting"),
            "pss": cell_summary(rows, "pss"),
        }

    def values(effect_name: str, metric: str) -> list[float]:
        return [
            float(row["effects"][effect_name][metric])
            for row in per_seed.values()
        ]

    paired_effects = {
        "full_vs_base_accuracy": summarize(values("full_vs_base", "accuracy_delta")),
        "full_vs_base_forgetting": summarize(
            values("full_vs_base", "forgetting_improvement")
        ),
        "full_vs_base_pss": summarize(
            values("full_vs_base", "pss_relative_change"), expected_sign=-1
        ),
        "layer3_accuracy_without_layer2": summarize(
            values("layer3_without_layer2", "accuracy_delta"), expected_sign=-1
        ),
        "layer3_accuracy_with_layer2": summarize(
            values("layer3_with_layer2", "accuracy_delta"), expected_sign=-1
        ),
        "layer3_retention_without_layer2": summarize(
            values("layer3_without_layer2", "forgetting_improvement")
        ),
        "layer3_retention_with_layer2": summarize(
            values("layer3_with_layer2", "forgetting_improvement")
        ),
        "layer2_accuracy_with_layer3": summarize(
            values("layer2_with_layer3", "accuracy_delta")
        ),
        "accuracy_penalty_mitigation": summarize(
            [
                float(row["interaction"]["layer3_accuracy_penalty_mitigation"])
                for row in per_seed.values()
            ]
        ),
        "retention_preservation_fraction": summarize(
            [
                float(
                    row["interaction"][
                        "layer3_retention_gain_preservation_fraction"
                    ]
                )
                for row in per_seed.values()
            ]
        ),
    }
    report = {
        "stage": "D133 exact paired statistics",
        "source": str(args.input),
        "small_sample_warning": (
            "With five pairs, the minimum attainable two-sided exact sign-flip "
            "p-value for a fully consistent nonzero direction is 0.0625. "
            "Directional p-values are reported only for pre-specified internal signs."
        ),
        "cell_statistics": cell_statistics,
        "paired_effects": paired_effects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    svg_figure(report, args.figure)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
