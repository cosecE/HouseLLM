"""
Plots for HouseLLM constrained-decoding results.

Generates 6 visualizations from results_full.json, mapped to the rubric:

  Performance + Comparative analysis ........ 1, 2
  Statistical significance .................. 3
  Ablation studies .......................... 2 (each adjacent pair = one ablation)
  Error analysis ............................ 4, 5
  Limitations / tradeoffs ................... 6

Each figure is saved as a separate 300-DPI PNG so it can be dropped
straight into a slide.

Usage (in Colab cell):
  !python make_plots.py --results /content/drive/MyDrive/HouseLLM/results_full.json --out /content/drive/MyDrive/HouseLLM/plots/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import stats

# ---- Style ---------------------------------------------------------------

CONDITIONS = ["baseline", "soft", "medium", "hard"]

# Match the slide deck palette
COLORS = {
    "baseline": "#7C3AED",  # violet
    "soft":     "#06B6D4",  # light cyan
    "medium":   "#0891B2",  # cyan
    "hard":     "#C026D3",  # magenta
}
TEXT_COLOR  = "#1E293B"
MUTED_COLOR = "#64748B"
GRID_COLOR  = "#E2E8F0"
BG_COLOR    = "#F8F9FA"

mpl.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    MUTED_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.8,
    "font.family":       "DejaVu Sans",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ---- Helpers -------------------------------------------------------------

def load_results(path):
    with open(path) as f:
        return json.load(f)


def per_record_field(results, condition, field):
    """Pull a list of per-record values for one field+condition."""
    return [r[field] for r in results[condition]["per_record_deterministic"]]


def annotate_bars(ax, bars, fmt="{:.2f}", offset=0.01):
    """Add value labels on top of bars."""
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + offset, fmt.format(h),
                ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)


def style_axes(ax, ylabel=None, ylim=None, ygrid=True):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    if ygrid:
        ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)


# ---- Plot 1: Headline 4-panel overview -----------------------------------

def plot_headline_overview(results, out_path):
    """2x2 panel: validity, mean F1 across fields, hallucination, latency."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Headline metrics across constraint conditions",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=0.995)

    colors = [COLORS[c] for c in CONDITIONS]
    x = np.arange(len(CONDITIONS))

    # Top-left: validity rate
    validities = [results[c]["validity_rate"] for c in CONDITIONS]
    ax = axes[0, 0]
    bars = ax.bar(x, validities, color=colors, edgecolor="white", linewidth=1.5)
    annotate_bars(ax, bars, fmt="{:.0%}", offset=0.015)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
    ax.set_title("Schema validity rate")
    style_axes(ax, ylabel="fraction of valid JSON", ylim=(0, 1.10))

    # Top-right: mean F1 across the four list-typed fields
    f1_fields = ["symptoms_f1", "negated_symptoms_f1", "history_f1",
                 "diagnosis_f1", "treatment_f1"]
    mean_f1s = [np.mean([results[c]["deterministic"][f] for f in f1_fields])
                for c in CONDITIONS]
    ax = axes[0, 1]
    bars = ax.bar(x, mean_f1s, color=colors, edgecolor="white", linewidth=1.5)
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.005)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
    ax.set_title("Mean F1 across list fields")
    style_axes(ax, ylabel="F1 (averaged over 5 fields)", ylim=(0, 0.50))

    # Bottom-left: hallucination rate
    halls = [results[c]["deterministic"]["hallucination_rate"] for c in CONDITIONS]
    ax = axes[1, 0]
    bars = ax.bar(x, halls, color=colors, edgecolor="white", linewidth=1.5)
    annotate_bars(ax, bars, fmt="{:.3f}", offset=0.002)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
    ax.set_title("Hallucination rate (lower = better)")
    style_axes(ax, ylabel="fraction of ungrounded symptoms", ylim=(0, 0.13))

    # Bottom-right: avg latency
    lats = [results[c]["avg_latency_sec"] for c in CONDITIONS]
    ax = axes[1, 1]
    bars = ax.bar(x, lats, color=colors, edgecolor="white", linewidth=1.5)
    annotate_bars(ax, bars, fmt="{:.1f}s", offset=0.3)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
    ax.set_title("Mean latency per record")
    style_axes(ax, ylabel="seconds", ylim=(0, max(lats) * 1.18))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Plot 2: Per-field F1 grouped bar chart ------------------------------

def plot_per_field_f1(results, out_path):
    """Headline plot: F1 for each field, grouped by condition.

    This is the visualization that shows the 'errors shift' story most
    directly — diagnosis F1 craters at hard while validity stays at 100%.
    """
    fields = [
        ("symptoms_f1",          "symptoms"),
        ("negated_symptoms_f1",  "negated_symptoms"),
        ("history_f1",           "history*"),
        ("diagnosis_f1",         "diagnosis*"),
        ("treatment_f1",         "treatment"),
    ]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    n_fields = len(fields)
    n_cond   = len(CONDITIONS)
    bar_w    = 0.18
    x        = np.arange(n_fields)

    # Compute std errors from per-record data for error bars
    for i, cond in enumerate(CONDITIONS):
        means = []
        sems  = []
        for fkey, _ in fields:
            vals = per_record_field(results, cond, fkey)
            means.append(np.mean(vals))
            sems.append(stats.sem(vals))
        offset = (i - (n_cond - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w,
                      yerr=sems, capsize=3,
                      label=cond, color=COLORS[cond],
                      edgecolor="white", linewidth=1,
                      error_kw={"elinewidth": 1, "ecolor": MUTED_COLOR})

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in fields])
    ax.set_ylabel("F1 score")
    ax.set_title("Per-field F1 by condition  //  * = ICD-10 vocabulary applied at hard")
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper right", frameon=False, ncol=4)
    ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)

    # Highlight the diagnosis cratering
    ax.annotate("vocab constraint\ncraters F1",
                xy=(3 + (3 - 1.5) * bar_w, 0.02),
                xytext=(3.2, 0.30),
                fontsize=10, color=COLORS["hard"],
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["hard"], lw=1.2))

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Plot 3: Per-record F1 boxplots (statistical significance) -----------

def plot_diagnosis_distribution(results, out_path):
    """Per-record F1 distributions for the diagnosis field, with paired
    Wilcoxon significance tests between adjacent conditions."""
    field = "diagnosis_f1"
    data = {c: per_record_field(results, c, field) for c in CONDITIONS}

    fig, ax = plt.subplots(figsize=(10, 5.5))

    positions = np.arange(len(CONDITIONS))
    bp = ax.boxplot(
        [data[c] for c in CONDITIONS],
        positions=positions, widths=0.55,
        patch_artist=True, showmeans=True,
        medianprops=dict(color="white", linewidth=2),
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor=TEXT_COLOR, markersize=6),
        flierprops=dict(marker="o", markersize=4, markeredgecolor=MUTED_COLOR,
                        markerfacecolor="none", linewidth=0.8),
        whiskerprops=dict(color=MUTED_COLOR),
        capprops=dict(color=MUTED_COLOR),
    )
    for patch, c in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(COLORS[c])
        patch.set_alpha(0.85)
        patch.set_edgecolor(COLORS[c])

    # Overlay individual record points (jittered)
    rng = np.random.default_rng(42)
    for i, c in enumerate(CONDITIONS):
        jitter = rng.normal(0, 0.04, len(data[c]))
        ax.scatter(np.full(len(data[c]), i) + jitter, data[c],
                   color=TEXT_COLOR, alpha=0.18, s=10, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(CONDITIONS)
    ax.set_ylabel("diagnosis F1 (per record)")
    ax.set_title("Per-record diagnosis F1 distribution  //  paired Wilcoxon tests")
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)

    # Annotate paired Wilcoxon between adjacent conditions
    pairs = [(0, 1), (1, 2), (2, 3)]
    bracket_y = 1.02
    for x1, x2 in pairs:
        c1, c2 = CONDITIONS[x1], CONDITIONS[x2]
        try:
            stat, p = stats.wilcoxon(data[c1], data[c2])
        except ValueError:
            p = 1.0
        sig = ("***" if p < 0.001 else "**" if p < 0.01
               else "*" if p < 0.05 else "ns")
        ax.plot([x1, x1, x2, x2],
                [bracket_y, bracket_y + 0.02, bracket_y + 0.02, bracket_y],
                color=TEXT_COLOR, linewidth=0.9)
        ax.text((x1 + x2) / 2, bracket_y + 0.025,
                f"p={p:.3f} {sig}",
                ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
        bracket_y += 0.10

    ax.set_ylim(-0.05, bracket_y + 0.05)

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Plot 4: TP/FP/FN error decomposition for diagnosis ------------------

def plot_error_decomposition(results, out_path):
    """Stacked horizontal bars: TP, FP, FN counts (mean per record) for
    diagnosis. Shows where errors come from at each constraint level.

    Hard is expected to have very high FP — these are vocabulary-valid
    but semantically wrong predictions ('semantic substitution')."""
    field = "diagnosis"
    parts = ["tp", "fp", "fn"]
    part_colors = {
        "tp": "#10B981",  # emerald, "correct"
        "fp": COLORS["hard"],       # magenta, "wrong but predicted"
        "fn": "#F59E0B",  # amber, "missed"
    }
    part_labels = {
        "tp": "true positive (correct)",
        "fp": "false positive (wrong prediction)",
        "fn": "false negative (missed)",
    }

    counts = {c: {p: results[c]["deterministic"][f"{field}_{p}"]
                  for p in parts}
              for c in CONDITIONS}

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(CONDITIONS))
    left = np.zeros(len(CONDITIONS))
    for p in parts:
        vals = np.array([counts[c][p] for c in CONDITIONS])
        bars = ax.barh(y, vals, left=left, color=part_colors[p],
                       label=part_labels[p], edgecolor="white", linewidth=1)
        # Annotate values inside bars
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.15:  # only annotate visible chunks
                ax.text(l + v / 2, i, f"{v:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(CONDITIONS)
    ax.invert_yaxis()
    ax.set_xlabel("mean count per record")
    ax.set_title("Diagnosis error decomposition  //  TP+FP+FN per record")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              frameon=False, fontsize=9, ncol=3)
    ax.xaxis.grid(True, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)

    # Annotate the FP spike at hard
    hard_idx = CONDITIONS.index("hard")
    ax.annotate("substitution:\nvocab-valid but wrong",
                xy=(counts["hard"]["tp"] + counts["hard"]["fp"] / 2, hard_idx),
                xytext=(7.5, hard_idx - 1.5),
                fontsize=9, color=COLORS["hard"],
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["hard"], lw=1.2))

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Plot 5: LLM judge equivalence heatmap -------------------------------

def plot_llm_judge_heatmap(results, out_path):
    """Heatmap of LLM-judge semantic-equivalence rates (field × condition)."""
    fields = [
        ("name_equivalent",             "name"),
        ("symptoms_equivalent",         "symptoms"),
        ("negated_symptoms_equivalent", "negated_symptoms"),
        ("diagnosis_equivalent",        "diagnosis"),
        ("treatment_equivalent",        "treatment"),
    ]

    matrix = np.array([
        [results[c]["llm_judge"][fkey] for c in CONDITIONS]
        for fkey, _ in fields
    ])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    # Cell labels
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v < 0.5 else TEXT_COLOR
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([label for _, label in fields])
    ax.set_title("LLM-judge semantic equivalence  //  field × condition")
    ax.set_xlabel("condition")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("equivalence rate (0–1)", fontsize=10)

    # Note about baseline judge n (placed below the figure)
    fig.text(0.05, 0.02,
             "n_judge:  baseline=2   soft=52   medium=64   hard=65   "
             "(judge runs only on schema-valid records)",
             fontsize=8, color=MUTED_COLOR, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Plot 6: Latency vs F1 tradeoff --------------------------------------

def plot_latency_vs_f1(results, out_path):
    """Scatter: x = latency, y = mean F1. Each point = one condition.
    Shows the cost of constraint."""
    f1_fields = ["symptoms_f1", "negated_symptoms_f1", "history_f1",
                 "diagnosis_f1", "treatment_f1"]

    xs, ys, labels, colors_list = [], [], [], []
    for c in CONDITIONS:
        xs.append(results[c]["avg_latency_sec"])
        ys.append(np.mean([results[c]["deterministic"][f] for f in f1_fields]))
        labels.append(c)
        colors_list.append(COLORS[c])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for x, y, l, color in zip(xs, ys, labels, colors_list):
        ax.scatter(x, y, s=300, color=color, edgecolor="white",
                   linewidth=2.5, zorder=3, alpha=0.9)
        ax.annotate(l, (x, y), xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=11, color=TEXT_COLOR, fontweight="bold")

    # Connect with dashed line in condition order
    ax.plot(xs, ys, color=MUTED_COLOR, linestyle="--", linewidth=1,
            alpha=0.5, zorder=1)

    ax.set_xlabel("mean latency per record (seconds)")
    ax.set_ylabel("mean F1 across list fields")
    ax.set_title("Cost of constraint  //  latency vs accuracy tradeoff")
    ax.yaxis.grid(True, linestyle="-", alpha=0.6)
    ax.xaxis.grid(True, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(min(xs) - 3, max(xs) + 3)
    ax.set_ylim(0, max(ys) * 1.3)

    plt.tight_layout()
    plt.savefig(out_path, facecolor="white")
    plt.close()
    print(f"  ✓ {out_path.name}")


# ---- Statistical summary -------------------------------------------------

def print_significance_summary(results):
    """Prints paired Wilcoxon p-values for every adjacent-condition pair
    on every F1 metric. Useful for the report and for choosing what to
    annotate on plots."""
    print("\n=== Paired Wilcoxon tests (adjacent conditions) ===")
    print("comparing per-record values across the 65 records\n")
    metrics = ["symptoms_f1", "negated_symptoms_f1", "history_f1",
               "diagnosis_f1", "treatment_f1", "hallucination_rate"]
    pairs = [("baseline", "soft"), ("soft", "medium"), ("medium", "hard")]

    print(f"{'metric':<22}" + "".join(f"{a}→{b:<10}" for a, b in pairs))
    for m in metrics:
        line = f"{m:<22}"
        for c1, c2 in pairs:
            v1 = per_record_field(results, c1, m)
            v2 = per_record_field(results, c2, m)
            try:
                _, p = stats.wilcoxon(v1, v2)
                sig = ("***" if p < 0.001 else "**" if p < 0.01
                       else "*" if p < 0.05 else "ns")
                line += f"p={p:.3f} {sig:<5}".ljust(13)
            except ValueError:
                line += f"{'tied':<13}"
        print(line)


# ---- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results_full.json")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)

    print(f"Generating plots in {out_dir}/ ...\n")
    plot_headline_overview(results,        out_dir / "01_headline_overview.png")
    plot_per_field_f1(results,             out_dir / "02_per_field_f1.png")
    plot_diagnosis_distribution(results,   out_dir / "03_diagnosis_distribution.png")
    plot_error_decomposition(results,      out_dir / "04_error_decomposition.png")
    plot_llm_judge_heatmap(results,        out_dir / "05_llm_judge_heatmap.png")
    plot_latency_vs_f1(results,            out_dir / "06_latency_vs_f1.png")

    print_significance_summary(results)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots in {out_dir}/")


if __name__ == "__main__":
    main()
