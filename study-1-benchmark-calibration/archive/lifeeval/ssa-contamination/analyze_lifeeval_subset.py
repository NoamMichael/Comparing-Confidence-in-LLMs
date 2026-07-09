"""
Analyze the LifeEval no_evidence subset.

Loads judge verdicts from LLM_SSA_Analysis/sonnet_judge_results.csv,
joins with LifeEval data from Combined Results/ssa_reference_examples.csv,
filters to verdict == "no_evidence", and produces calibration plots
matching the existing analysis style.

Outputs:
  Plots/LifeEval_Subset/Calibration Plots/cal_plot_life_eval_subset_<model>.png
  Plots/LifeEval_Subset/aggregate_cal_plot_subset.png
  Plots/LifeEval_Subset/summary_bars_lifeeval_subset.png
  Plots/LifeEval_Subset/oc_by_radii_subset.png
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent
CLEAN_DATA = REPO / "Combined Results" / "combined_clean.csv"
JUDGE_DATA = REPO / "LLM_SSA_Analysis" / "sonnet_judge_results.csv"
OUT_DIR    = REPO / "Plots" / "LifeEval_Subset"
CAL_DIR    = OUT_DIR / "Calibration Plots"

# ── Constants ────────────────────────────────────────────────────────────────
RADIUS_LIST = [1, 5, 10, 20]
BIN_EDGES = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.000001]
BIN_CENTERS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
TICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MODEL_ORDER = [
    "Claude Haiku 3", "Claude-Sonnet-3.7", "Claude-Sonnet-4",
    "DeepSeek-R1", "DeepSeek-V3",
    "GPT-4o", "GPT-o3",
    "Gemini-2.5-Flash", "Gemini-2.5-Pro",
    "Llama-3.1-70B", "Llama-3.1-8B",
]

family_palettes = {
    "GPT":      sns.color_palette("Greens", 6),
    "Claude":   sns.color_palette("Blues", 6),
    "Gemini":   sns.color_palette("Purples", 6),
    "DeepSeek": sns.color_palette("Oranges", 6),
    "Llama":    sns.color_palette("RdPu", 6),
}


# ── Helper functions ─────────────────────────────────────────────────────────

def model_family(name: str) -> str:
    s = name.lower()
    if "gpt" in s or "o3" in s:   return "GPT"
    if "claude" in s:             return "Claude"
    if "gemini" in s:             return "Gemini"
    if "deepseek" in s:           return "DeepSeek"
    if "llama" in s:              return "Llama"
    return "Other"


def pick_color(name: str) -> tuple:
    fam = model_family(name)
    pal = family_palettes[fam]
    s = name.lower()
    if fam == "GPT":
        if "gpt-4o" in s: return pal[-2]
        if "o3" in s:     return pal[1]
    if fam == "Claude":
        if "sonnet" in s: return pal[-2]
        if "haiku"  in s: return pal[2]
    if fam == "Gemini":
        if "pro"   in s:  return pal[-2]
        if "flash" in s:  return pal[2]
    if fam == "DeepSeek":
        if "r1"   in s:  return pal[-2]
        if "v3" in s:    return pal[2]
    return pal[3]


def get_ece(score: pd.Series, confidence: pd.Series, n_bins: int = 10) -> float:
    score = score.astype(float).values
    confidence = confidence.astype(float).values
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges, right=True)
    ece = 0.0
    total = len(score)
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        n = mask.sum()
        if n > 0:
            ece += (n / total) * abs(score[mask].mean() - confidence[mask].mean())
    return ece


def qid_to_radius(qid: int) -> int:
    return RADIUS_LIST[qid % 4]


# ── Per-model calibration plot (matches existing style) ─────────────────────

def calibration_plot_individual(df, model_name, save_path):
    """Per-model calibration plot with histogram + scatter, matching existing style."""
    conf = df["Stated Confidence Answer"].astype(float).values
    score = df["Score"].astype(float).values
    n = len(df)
    ece = get_ece(pd.Series(score), pd.Series(conf))
    accuracy = score.mean()

    # Bin data
    bin_ids = np.digitize(conf, BIN_EDGES[:-1], right=False)  # 1-indexed
    bin_ids = np.clip(bin_ids, 1, len(BIN_EDGES) - 1)

    bin_means_score = []
    bin_means_conf = []
    bin_counts = []
    bin_se = []

    for i in range(1, len(BIN_EDGES)):
        mask = bin_ids == i
        cnt = mask.sum()
        bin_counts.append(cnt)
        if cnt > 0:
            bin_means_score.append(score[mask].mean())
            bin_means_conf.append(conf[mask].mean())
            se = score[mask].std(ddof=0) / np.sqrt(cnt) if cnt > 1 else 0
            bin_se.append(1.96 * se)
        else:
            bin_means_score.append(np.nan)
            bin_means_conf.append(np.nan)
            bin_se.append(0)

    bin_means_score = np.array(bin_means_score)
    bin_means_conf = np.array(bin_means_conf)
    bin_counts = np.array(bin_counts)
    bin_se = np.array(bin_se)

    # Normalize histogram
    bin_prop = bin_counts / bin_counts.sum() if bin_counts.sum() > 0 else bin_counts

    fig, ax1 = plt.subplots(figsize=(6, 6), dpi=300)

    # Histogram bars (orange) on secondary y-axis
    ax2 = ax1.twinx()
    bar_width = 0.08
    ax2.bar(BIN_CENTERS, bin_prop, width=bar_width, color="orange",
            alpha=0.8, zorder=1)
    ax2.set_ylabel("Proportion of Stated Confidence", fontsize=10)
    ax2.set_ylim(0, 1.0)

    # Perfect calibration diagonal
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.7, zorder=2)

    # Error bars
    valid = ~np.isnan(bin_means_conf)
    ax1.errorbar(
        bin_means_conf[valid], bin_means_score[valid],
        yerr=bin_se[valid],
        fmt="none", color="grey", ecolor="grey", capsize=3, linewidth=1.5, zorder=3
    )

    # Scatter
    ax1.scatter(bin_means_conf[valid], bin_means_score[valid],
                color="steelblue", s=50, zorder=4, edgecolors="white", linewidths=0.5)

    ax1.set_xlabel("Stated Confidence", fontsize=10)
    ax1.set_ylabel("Average Accuracy", fontsize=10)
    ax1.set_xlim(-0.02, 1.08)
    ax1.set_ylim(-0.02, 1.08)
    ax1.set_xticks(TICKS)
    ax1.set_yticks(TICKS)
    ax1.set_title(
        f"Calibration Plot for {model_name} on LifeEval\n"
        f"ECE: {ece:.3f} | Accuracy: {accuracy:.3f} | n = {n}",
        fontsize=11
    )
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return ece, accuracy


# ── Aggregate calibration plot ───────────────────────────────────────────────

def calibration_plot_aggregate(df, save_path, title=None):
    """Aggregate calibration plot with scatter + CI error bars."""
    conf_col = "Stated Confidence Answer"
    score_col = "Score"
    sc_df = df[[conf_col, score_col]].copy()
    sc_df[conf_col] = pd.to_numeric(sc_df[conf_col], errors="coerce")
    sc_df[score_col] = pd.to_numeric(sc_df[score_col], errors="coerce")
    sc_df = sc_df.dropna()

    sc_df["bin"] = pd.cut(sc_df[conf_col], bins=BIN_EDGES, right=False)

    grouped = sc_df.groupby("bin", observed=False)[[conf_col, score_col]].agg(["mean", "count", "std"])
    sc_out = grouped.loc[:, [
        (score_col, "mean"), (conf_col, "mean"),
        (score_col, "count"), (score_col, "std")
    ]]
    sc_out.columns = ["mean_score", "mean_confidence", "count", "std"]
    se = sc_out["std"] / np.sqrt(sc_out["count"])
    sc_out["ci"] = 1.96 * se

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.set_xticks(TICKS)
    ax.set_yticks(TICKS)
    ax.axline([0, 0], [1, 1], color="grey", linestyle="--",
              label="Line of Perfect Calibration", alpha=0.5)
    ax.errorbar(
        sc_out["mean_confidence"], sc_out["mean_score"],
        yerr=sc_out["ci"], fmt="none", linewidth=2, capsize=0,
        color="lightgrey", ecolor="lightgrey", barsabove=False
    )
    sns.scatterplot(sc_out, y="mean_score", x="mean_confidence",
                    label="Stated Confidence", color="black", zorder=3, ax=ax)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    n = len(df)
    if title is None:
        title = f"Aggregate Calibration — LifeEval (no_evidence subset, n={n})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Summary bars (2x2) ──────────────────────────────────────────────────────

def make_summary_bars(stats_df, save_path, label="no_evidence"):
    """4-panel summary: ECE, Overconfidence, Accuracy, n per model."""
    stats = stats_df.sort_values("Model")
    models = stats["Model"].values
    colors = [pick_color(m) for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
    fig.suptitle(f"Summary Statistics for LifeEval ({label} subset)", fontsize=14)

    panels = [
        ("ECE", "ECE", axes[0, 0]),
        ("Overconfidence", "Over Confidence", axes[0, 1]),
        ("Accuracy", "Accuracy (%)", axes[1, 0]),
        ("n", "Count (n)", axes[1, 1]),
    ]

    for col, ylabel, ax in panels:
        vals = stats[col].values
        if col == "Accuracy":
            vals = vals * 100
        bars = ax.bar(range(len(models)), vals, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"LifeEval — {col} by Model")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        if col == "Overconfidence":
            ax.axhline(0, color="black", linewidth=0.8)
        if col == "n":
            # Add 751 reference line for full LifeEval
            ax.axhline(751, color="grey", linestyle="--", alpha=0.5)
            ax.annotate("751", xy=(len(models) - 0.5, 751), fontsize=8, color="grey")

    # Family legend
    families = list(dict.fromkeys(model_family(m) for m in models))
    patches = [Patch(color=family_palettes[f][3], label=f) for f in families]
    axes[0, 0].legend(handles=patches, title="Model Families", loc="upper right", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ── Overconfidence by radii ──────────────────────────────────────────────────

def plot_oc_by_radii(df, save_path):
    """Grouped bar chart: overconfidence per model per radius (matches analysis.ipynb style)."""
    data = df.copy()
    data["Score"] = data["Score"].astype(float)
    data["Stated Confidence Answer"] = data["Stated Confidence Answer"].astype(float)
    data["Radius"] = data["Question ID"].astype(int).apply(qid_to_radius)
    data["Stated Overconfidence"] = data["Stated Confidence Answer"] - data["Score"]

    grouped = data.groupby(["Model", "Radius"]).agg(
        Stated_OC=("Stated Overconfidence", "mean"),
        n=("Stated Overconfidence", "count")
    ).reset_index()
    grouped["Model Family"] = grouped["Model"].apply(model_family)

    family_colors = {
        "GPT": "#2ca02c", "Claude": "#1f77b4", "Gemini": "#9467bd",
        "DeepSeek": "#ff7f0e", "Llama": "#e377c2", "Other": "#7f7f7f"
    }
    radius_alphas = {1: 1.0, 5: 0.75, 10: 0.5, 20: 0.3}

    models = sorted(grouped["Model"].unique())
    radii = sorted(grouped["Radius"].unique())
    n_models = len(models)
    n_radii = len(radii)

    fig, ax = plt.subplots(figsize=(14, 7))

    group_width = 0.8
    bar_width = group_width / n_radii
    x_positions = np.arange(n_models)
    legend_handles = {}

    for i, model in enumerate(models):
        model_data = grouped[grouped["Model"] == model].sort_values("Radius")
        for j, radius in enumerate(radii):
            bar_data = model_data[model_data["Radius"] == radius]
            if bar_data.empty:
                continue
            value = bar_data["Stated_OC"].values[0]
            family = bar_data["Model Family"].values[0]
            color = family_colors.get(family, "grey")
            alpha = radius_alphas.get(radius, 0.5)
            x_pos = x_positions[i] - (group_width / 2) + (j * bar_width) + (bar_width / 2)
            bar = ax.bar(x_pos, value, width=bar_width, color=color, alpha=alpha,
                         label=family)
            if family not in legend_handles:
                legend_handles[family] = bar

    # No title (matches original)
    ax.set_ylabel("Average Overconfidence", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")

    # Radius sub-labels
    radius_labels = ["1", "5", "10", "20"]
    for i in x_positions:
        shift = -bar_width * 2 + 0.5 * bar_width
        for r in radius_labels:
            xpos = i + shift
            ax.text(xpos, 0.05, r, ha="center", va="top",
                    transform=ax.get_xaxis_transform(), fontsize=9, color="black")
            shift += bar_width

    ax.axhline(0, color="black", linewidth=1.0)
    ax.yaxis.grid(True, linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)

    # Legend outside the plot (matches original)
    ax.legend(
        legend_handles.values(), legend_handles.keys(),
        title="Model Families",
        bbox_to_anchor=(1.02, 1), loc="upper left"
    )

    plt.subplots_adjust(bottom=0.25, right=0.85)
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(verdicts=None):
    if verdicts is None:
        verdicts = ["no_evidence"]

    subset_label = "+".join(verdicts)
    out_dir = REPO / "Plots" / f"LifeEval_Subset_{subset_label}"
    cal_dir = out_dir / "Calibration Plots"

    # Create output directories
    os.makedirs(cal_dir, exist_ok=True)

    # ── Load & join data ─────────────────────────────────────────────────
    print(f"Loading data... (verdicts: {verdicts})")
    clean_df = pd.read_csv(CLEAN_DATA)
    judge_df = pd.read_csv(JUDGE_DATA, usecols=["Model", "Question ID", "verdict"])

    # Filter to LifeEval
    life_eval = clean_df[clean_df["Question Set"] == "LifeEval"].copy()
    print(f"  LifeEval rows in combined_clean: {len(life_eval)}")

    # Harmonize join keys
    life_eval["Question ID"] = life_eval["Question ID"].astype(str)
    judge_df["Question ID"] = judge_df["Question ID"].astype(str)

    # Merge
    merged = life_eval.merge(judge_df, on=["Model", "Question ID"], how="inner")
    print(f"  Merged rows: {len(merged)}")

    # Filter to selected verdicts
    subset = merged[merged["verdict"].isin(verdicts)].copy()
    print(f"  {subset_label} rows: {len(subset)}")

    # Derive radius
    subset["Question ID"] = subset["Question ID"].astype(int)
    subset["Score"] = subset["Score"].astype(float)
    subset["Stated Confidence Answer"] = subset["Stated Confidence Answer"].astype(float)
    subset["Radius"] = subset["Question ID"].apply(qid_to_radius)
    subset["Overconfidence"] = subset["Stated Confidence Answer"] - subset["Score"]

    # ── Verdict distribution (for appendix context) ──────────────────────
    print("\n=== Verdict Distribution ===")
    verdict_counts = merged["verdict"].value_counts()
    for v, c in verdict_counts.items():
        print(f"  {v}: {c} ({100*c/len(merged):.1f}%)")

    print("\n=== Per-Model no_evidence Counts ===")
    model_counts = subset.groupby("Model").size().sort_values(ascending=False)
    for m, c in model_counts.items():
        print(f"  {m:25s} {c:5d}")

    # ── Per-model calibration plots + stats ──────────────────────────────
    print("\n=== Per-Model Summary ===")
    print(f"{'Model':25s} {'n':>5s} {'ECE':>7s} {'Acc':>7s} {'OC':>7s} {'MeanConf':>9s}")
    print("-" * 65)

    stats_rows = []
    models_in_data = sorted(subset["Model"].unique())

    for model in models_in_data:
        mdf = subset[subset["Model"] == model]
        n = len(mdf)
        ece_val = get_ece(mdf["Score"], mdf["Stated Confidence Answer"])
        acc = mdf["Score"].mean()
        mean_conf = mdf["Stated Confidence Answer"].mean()
        oc = mean_conf - acc

        save_name = f"cal_plot_life_eval_subset_{model}.png"
        save_path = cal_dir / save_name
        calibration_plot_individual(mdf, model, save_path)

        stats_rows.append({
            "Model": model, "n": n, "ECE": ece_val,
            "Accuracy": acc, "Overconfidence": oc, "MeanConf": mean_conf
        })
        flag = " *" if n < 100 else ""
        print(f"  {model:25s} {n:5d} {ece_val:7.3f} {acc:7.3f} {oc:+7.3f} {mean_conf:9.3f}{flag}")

    stats_df = pd.DataFrame(stats_rows)
    print(f"\n  * = fewer than 100 rows (noisy estimates)")

    # ── Aggregate stats ──────────────────────────────────────────────────
    agg_ece = get_ece(subset["Score"], subset["Stated Confidence Answer"])
    agg_acc = subset["Score"].mean()
    agg_conf = subset["Stated Confidence Answer"].mean()
    agg_oc = agg_conf - agg_acc

    print(f"\n=== Aggregate Statistics (n={len(subset)}) ===")
    print(f"  ECE:              {agg_ece:.4f}")
    print(f"  Accuracy:         {agg_acc:.4f}")
    print(f"  Mean Confidence:  {agg_conf:.4f}")
    print(f"  Overconfidence:   {agg_oc:+.4f}")

    # ── Per-radius stats (hard-easy effect) ──────────────────────────────
    print(f"\n=== Overconfidence by Radius (Hard-Easy Effect) ===")
    print(f"{'Radius':>8s} {'n':>6s} {'MeanOC':>8s} {'MeanConf':>10s} {'MeanAcc':>9s}")
    print("-" * 50)
    for r in RADIUS_LIST:
        rdf = subset[subset["Radius"] == r]
        n_r = len(rdf)
        oc_r = rdf["Overconfidence"].mean()
        conf_r = rdf["Stated Confidence Answer"].mean()
        acc_r = rdf["Score"].mean()
        print(f"  {r:>6d} {n_r:>6d} {oc_r:>+8.4f} {conf_r:>10.4f} {acc_r:>9.4f}")

    # ── Per-model per-radius matrix ──────────────────────────────────────
    print(f"\n=== Per-Model Per-Radius Overconfidence ===")
    header = f"{'Model':25s}" + "".join(f"{'R='+str(r):>10s}" for r in RADIUS_LIST)
    print(header)
    print("-" * (25 + 10 * len(RADIUS_LIST)))
    for model in models_in_data:
        mdf = subset[subset["Model"] == model]
        vals = []
        for r in RADIUS_LIST:
            rdf = mdf[mdf["Radius"] == r]
            if len(rdf) > 0:
                vals.append(f"{rdf['Overconfidence'].mean():>+10.4f}")
            else:
                vals.append(f"{'N/A':>10s}")
        print(f"  {model:25s}" + "".join(vals))

    # ── Generate plots ───────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Aggregate calibration
    agg_title = f"Aggregate Calibration — LifeEval ({subset_label} subset, n={len(subset)})"
    calibration_plot_aggregate(subset, out_dir / "aggregate_cal_plot_subset.png",
                               title=agg_title)
    print(f"  Saved: aggregate_cal_plot_subset.png")

    # Summary bars
    make_summary_bars(stats_df, out_dir / "summary_bars_lifeeval_subset.png",
                      label=subset_label)
    print(f"  Saved: summary_bars_lifeeval_subset.png")

    # Overconfidence by radii
    plot_oc_by_radii(subset, out_dir / "oc_by_radii_subset.png")
    print(f"  Saved: oc_by_radii_subset.png")

    print(f"\nAll plots saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LifeEval subset calibration analysis")
    parser.add_argument(
        "--verdicts", nargs="+", default=["no_evidence"],
        choices=["no_evidence", "weak_evidence", "strong_evidence"],
        help="Verdict(s) to include (default: no_evidence)"
    )
    args = parser.parse_args()
    main(verdicts=args.verdicts)
