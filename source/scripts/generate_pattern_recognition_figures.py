"""Generate the Pattern Recognition figures from frozen experiment summaries.

The script reads only frozen analyzer outputs. It writes editable PDF/SVG
figures, high-resolution PNG previews, and compact CSV files containing the
values used in each figure.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ARTIFACT_ROOT / "figures"
DATA_DIR = ARTIFACT_ROOT / "figure_data"

D133_PATH = ARTIFACT_ROOT / "results" / "analysis" / "d133_layer2_prba_interaction.json"
D126_PATH = ARTIFACT_ROOT / "results" / "analysis" / "d126_memory_summary.json"

COLORS = {
    "base": "#4D4D4D",
    "layer2": "#0072B2",
    "layer3": "#D55E00",
    "full": "#009E73",
    "c": "#4D4D4D",
    "prba": "#0072B2",
    "obc": "#D55E00",
}

FACTORIAL_STYLES = {
    "CE-ACE": {"color": COLORS["base"], "marker": "o"},
    "Layer 2 only": {"color": COLORS["layer2"], "marker": "s"},
    "Layer 3 only": {"color": COLORS["layer3"], "marker": "^"},
    "Full chain": {"color": COLORS["full"], "marker": "D"},
}

METHOD_STYLES = {
    "C": {"color": COLORS["c"], "marker": "o", "linestyle": "-"},
    "PRBA": {"color": COLORS["prba"], "marker": "s", "linestyle": "--"},
    "OBC": {"color": COLORS["obc"], "marker": "^", "linestyle": ":"},
}

FINAL_WIDTH_IN = 4.22

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_sd(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    """Exhaustive percentile bootstrap used for the five paired effects."""

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = [float(np.mean(arr[list(indices)])) for indices in itertools.product(range(n), repeat=n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.svg", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def d133_cells() -> dict[str, list[dict[str, float]]]:
    data = load_json(D133_PATH)
    mapping = {
        "CE-ACE": "base",
        "Layer 2 only": "layer2_only",
        "Layer 3 only": "layer3_only",
        "Full chain": "full_three_layer",
    }
    cells: dict[str, list[dict[str, float]]] = {label: [] for label in mapping}
    for seed, block in data["per_seed"].items():
        for label, key in mapping.items():
            cell = block["cells"][key]
            cells[label].append(
                {
                    "seed": int(seed),
                    "accuracy": float(cell["mean_accuracy"]) * 100.0,
                    "mean_af": float(cell["mean_forgetting"]) * 100.0,
                    "worst_af": float(cell["worst_forgetting"]) * 100.0,
                    "pss": float(cell["pss"]),
                }
            )
    for values in cells.values():
        values.sort(key=lambda row: int(row["seed"]))
    return cells


def plot_figure_2() -> None:
    cells = d133_cells()
    order = ["CE-ACE", "Layer 2 only", "Layer 3 only", "Full chain"]
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(FINAL_WIDTH_IN, 3.2))
    for label in order:
        values = cells[label]
        x = np.asarray([row["mean_af"] for row in values], dtype=float)
        y = np.asarray([row["accuracy"] for row in values], dtype=float)
        style = FACTORIAL_STYLES[label]
        color = style["color"]
        marker = style["marker"]
        ax.scatter(
            x,
            y,
            s=27,
            alpha=0.48,
            color=color,
            marker=marker,
            edgecolors="none",
            zorder=2,
        )
        mean_x = float(np.mean(x))
        mean_y = float(np.mean(y))
        sd_x = sample_sd(x.tolist())
        sd_y = sample_sd(y.tolist())
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=sd_x,
            yerr=sd_y,
            fmt=marker,
            markersize=6.0,
            color=color,
            ecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            capsize=2.5,
            elinewidth=0.9,
            zorder=4,
            label=label,
        )
        rows.extend(
            {
                "figure": "Figure 2",
                "structure": label,
                "seed": int(row["seed"]),
                "accuracy_percent": row["accuracy"],
                "mean_af_percent": row["mean_af"],
                "pss_fraction": row["pss"],
                "pss_pp": row["pss"] * 100.0,
            }
            for row in values
        )
    ax.set_xlabel("Mean average forgetting (%)")
    ax.set_ylabel("Final accuracy (%)")
    ax.set_xlim(28.5, 40.5)
    ax.set_ylim(14.5, 21.5)
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [0, 2, 1, 3]
    ax.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="lower right",
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.16, top=0.98)
    save_figure(fig, "Figure_2_pareto")
    write_csv(DATA_DIR / "figure2_d133_operating_points.csv", rows)


def plot_figure_3() -> None:
    cells = d133_cells()
    base = cells["CE-ACE"]
    comparisons = [
        ("Layer 2 only", cells["Layer 2 only"]),
        ("Layer 3 only", cells["Layer 3 only"]),
        ("Full chain", cells["Full chain"]),
    ]
    metrics = [
        ("Accuracy delta (pp)", "accuracy"),
        ("Forgetting improvement (pp)", "forgetting"),
        ("PSS improvement (pp)", "pss"),
    ]
    effect_rows: list[dict[str, object]] = []
    effect_map: dict[str, dict[str, list[float]]] = {}
    for label, child in comparisons:
        if [row["seed"] for row in base] != [row["seed"] for row in child]:
            raise ValueError(f"Seed mismatch in D133 comparison: {label}")
        accuracy = [c["accuracy"] - b["accuracy"] for b, c in zip(base, child)]
        forgetting = [b["mean_af"] - c["mean_af"] for b, c in zip(base, child)]
        pss = [(b["pss"] - c["pss"]) * 100.0 for b, c in zip(base, child)]
        effect_map[label] = {
            "Accuracy delta (pp)": accuracy,
            "Forgetting improvement (pp)": forgetting,
            "PSS improvement (pp)": pss,
        }
        for metric, values in effect_map[label].items():
            low, high = bootstrap_ci(values)
            for row, value in zip(child, values):
                effect_rows.append(
                    {
                        "figure": "Figure 3",
                        "row_type": "seed",
                        "comparison": f"{label} vs CE-ACE",
                        "metric": metric,
                        "seed": int(row["seed"]),
                        "value": value,
                    }
                )
            effect_rows.append(
                {
                    "figure": "Figure 3",
                    "row_type": "summary",
                    "comparison": f"{label} vs CE-ACE",
                    "metric": metric,
                    "seed": "",
                    "value": "",
                    "mean": float(np.mean(values)),
                    "sd": sample_sd(values),
                    "ci_low": low,
                    "ci_high": high,
                    "n": len(values),
                }
            )

    fig, axes = plt.subplots(3, 1, figsize=(FINAL_WIDTH_IN, 5.4), sharey=True)
    y = np.arange(len(comparisons))
    panel_labels = ("(a)", "(b)", "(c)")
    for panel_label, axis, (metric, _) in zip(panel_labels, axes, metrics):
        for idx, (label, _) in enumerate(comparisons):
            values = effect_map[label][metric]
            mean = float(np.mean(values))
            low, high = bootstrap_ci(values)
            style = FACTORIAL_STYLES[label]
            color = style["color"]
            marker = style["marker"]
            seed_offsets = np.linspace(-0.10, 0.10, len(values))
            axis.scatter(
                values,
                idx + seed_offsets,
                s=19,
                marker=marker,
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.85,
                zorder=2,
            )
            axis.errorbar(
                mean,
                idx,
                xerr=[[mean - low], [high - mean]],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=5.5,
                capsize=2.5,
                elinewidth=1.1,
                zorder=4,
            )
        axis.axvline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        axis.set_xlabel(metric)
        axis.set_title(panel_label, loc="left", fontweight="bold", fontsize=9, pad=2)
        axis.text(
            1.0,
            1.02,
            "positive is better",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            color="#555555",
        )
        axis.grid(True, axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_yticks(y)
        axis.set_yticklabels([label for label, _ in comparisons])
        axis.invert_yaxis()
    fig.subplots_adjust(left=0.27, right=0.98, top=0.98, bottom=0.08, hspace=0.50)
    save_figure(fig, "Figure_3_paired_effects")
    write_csv(DATA_DIR / "figure3_d133_paired_effects.csv", effect_rows)


def plot_figure_4() -> None:
    data = load_json(D126_PATH)
    method_map = {
        "persistent_srrd_selective_swap_1": "C",
        "persistent_srrd_prequential_arbitration_1": "PRBA",
        "persistent_srrd_obc_1": "OBC",
    }
    memories = [50, 200]
    metric_specs = [
        ("Mean accuracy (%)", "mean_accuracy", True),
        ("Mean AF (%)", "mean_forgetting", False),
        ("PSS (pp)", "pss", False),
    ]
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(3, 1, figsize=(FINAL_WIDTH_IN, 5.4), sharex=True)

    def metric_value(item: dict, metric: str) -> float:
        if metric == "mean_forgetting":
            hard = float(item["hard"]["final_validation_forgetting"])
            blurry = float(item["blurry"]["final_validation_forgetting"])
            return (hard + blurry) / 2.0
        return float(item[metric])

    panel_labels = ("(a)", "(b)", "(c)")
    memory_positions = np.arange(len(memories), dtype=float)
    method_offsets = {"C": -0.10, "PRBA": 0.0, "OBC": 0.10}
    for panel_label, axis, (ylabel, metric, higher_is_better) in zip(
        panel_labels, axes, metric_specs
    ):
        for method_key, label in method_map.items():
            style = METHOD_STYLES[label]
            means = []
            sds = []
            values_by_memory = []
            for memory in memories:
                absolute = data["per_memory"][str(memory)]["absolute"][method_key]
                ordered = sorted(absolute.items(), key=lambda item: int(item[0]))
                values = [metric_value(item, metric) * 100.0 for _, item in ordered]
                values_by_memory.append(values)
                means.append(float(np.mean(values)))
                sds.append(sample_sd(values))
                for (seed, _), value in zip(ordered, values):
                    rows.append(
                        {
                            "figure": "Figure 4",
                            "memory": memory,
                            "method": label,
                            "seed": int(seed),
                            "metric": metric,
                            "value": value,
                        }
                    )
            x_positions = memory_positions + method_offsets[label]
            for x_position, values in zip(x_positions, values_by_memory):
                seed_jitter = np.linspace(-0.022, 0.022, len(values))
                axis.scatter(
                    x_position + seed_jitter,
                    values,
                    s=16,
                    marker=style["marker"],
                    facecolors="white",
                    edgecolors=style["color"],
                    linewidths=0.75,
                    alpha=0.85,
                    zorder=2,
                )
            axis.errorbar(
                x_positions,
                means,
                yerr=sds,
                marker=style["marker"],
                markersize=5.2,
                linewidth=1.25,
                capsize=2.5,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                markeredgecolor="black",
                markeredgewidth=0.45,
                zorder=4,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(memory_positions, [str(memory) for memory in memories])
        axis.set_xlim(-0.28, 1.28)
        axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        direction = "higher is better" if higher_is_better else "lower is better"
        axis.set_title(f"{panel_label} {direction}", loc="left", fontsize=8.5, pad=2)
    axes[-1].set_xlabel("Memory budget")
    axes[0].legend(
        loc="best",
        frameon=False,
        ncol=3,
        fontsize=7.5,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.98, bottom=0.08, hspace=0.46)
    save_figure(fig, "Figure_4_memory_budget")
    write_csv(DATA_DIR / "figure4_d126_memory_sensitivity.csv", rows)


def main() -> None:
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote figure data to {DATA_DIR}")


if __name__ == "__main__":
    main()
