"""
Shared Study-2 (BayesEval) figure and table builders.

Factored out of analysis/analysis.ipynb so the human-replication pipeline
(human_replication.py) and the notebook can single-source the calibration plots,
overconfidence figures, ECE metrics, and the RQ1/RQ3 summary table.

Everything is parameterized by a *series* list (LLMs plus an optional "Humans"
series), so adding humans is a config change, not new plotting code. Humans are
the point of comparison, drawn in a distinct style (black, diamond markers)
against the four Okabe-Ito model colors. A panel silently omits any series that
has no rows in that domain (e.g. humans in MedEval).

The universal outcome column is `true_probability` (scoring.py): a 0/1 hit for
WGD, the SSA window probability for LifeEval, the differential lookup for
MedEval.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from evaluate_diff import add_difficulty
from human_io import HUMAN_MODEL, MODEL_SLUGS, difficulty_column

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

MODEL_LABELS = {
    "anthropic_claude-haiku-4.5": "Claude Haiku 4.5",
    "google_gemini-2.5-flash": "Gemini 2.5 Flash",
    "meta-llama_llama-4-maverick": "Llama 4 Maverick",
    "openai_gpt-5.4-mini": "GPT-5.4 Mini",
}

# Okabe-Ito colorblind-safe palette (one per model).
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MODEL_COLORS = dict(zip(MODEL_SLUGS, CB_PALETTE))

HUMAN_LABEL = "Humans"
HUMAN_COLOR = "#000000"
HUMAN_MARKER = "D"
MODEL_MARKER = "o"

# 11 bins: [0.0,0.1), [0.1,0.2), ..., [0.9,1.0), [1.0]
BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
BIN_LABELS = ["[0.0,0.1)", "[0.1,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)",
              "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0)", "[1.0]"]
BIN_MIDPOINTS = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])

SIZE_MIN, SIZE_MAX = 10, 50


@dataclass(frozen=True)
class Series:
    """One line/bar group on a plot: a model or the human replication."""
    key: str      # matches the `model` column
    label: str
    color: str
    marker: str


def build_series(include_human: bool = True) -> list[Series]:
    """The four models, then Humans (if requested), in plot order."""
    series = [Series(s, MODEL_LABELS[s], MODEL_COLORS[s], MODEL_MARKER) for s in MODEL_SLUGS]
    if include_human:
        series.append(Series(HUMAN_MODEL, HUMAN_LABEL, HUMAN_COLOR, HUMAN_MARKER))
    return series


def apply_style() -> None:
    """Match the notebook's Matplotlib rcParams (call once before plotting)."""
    mpl.rcParams.update({
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":             8,
        "axes.labelsize":        7,
        "axes.titlesize":        7,
        "xtick.labelsize":       6,
        "ytick.labelsize":       6,
        "legend.fontsize":       6,
        "legend.title_fontsize": 8,
        "lines.linewidth":       1.2,
        "lines.markersize":      4,
        "patch.linewidth":       0.5,
        "axes.linewidth":        0.6,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.prop_cycle":       plt.cycler(color=CB_PALETTE),
        "xtick.major.width":     0.6,
        "ytick.major.width":     0.6,
        "xtick.direction":       "in",
        "ytick.direction":       "in",
        "text.color":            "#333333",
        "axes.edgecolor":        "#333333",
        "axes.labelcolor":       "#333333",
        "xtick.color":           "#333333",
        "ytick.color":           "#333333",
        "axes.grid":             False,
        "legend.frameon":        False,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
        "figure.dpi":            150,
    })


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_ece(conf: pd.Series, outcome: pd.Series, bin_edges=BIN_EDGES) -> float:
    """Expected Calibration Error over fixed confidence bins."""
    bins = pd.cut(conf, bins=bin_edges, right=False, include_lowest=True)
    grouped = pd.DataFrame(
        {"conf": conf.values, "outcome": outcome.values, "bin": bins.values}
    ).groupby("bin", observed=False)
    n = grouped.size()
    total = n.sum()
    if total == 0:
        return np.nan
    acc_per_bin = grouped["outcome"].mean()
    conf_per_bin = grouped["conf"].mean()
    return (n / total * (acc_per_bin - conf_per_bin).abs()).sum()


def bootstrap_ece(conf: pd.Series, outcome: pd.Series, n_boot=2000, rng=None) -> np.ndarray:
    """Bootstrap distribution of ECE.

    Uses the identity ECE = (1/N) * Σ_bin |Σ_{i in bin}(outcome_i - conf_i)| with
    the fixed BIN_EDGES, so each resample is a single np.bincount instead of a
    pandas groupby (orders of magnitude faster over thousands of iterations).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    conf_arr = np.asarray(conf, dtype=float)
    out_arr = np.asarray(outcome, dtype=float)
    n = len(conf_arr)
    n_bins = len(BIN_EDGES) - 1
    # Left-closed bins matching pd.cut(bins=BIN_EDGES, right=False).
    bin_idx = np.clip(np.digitize(conf_arr, BIN_EDGES) - 1, 0, n_bins - 1)
    diff = out_arr - conf_arr
    eces = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sums = np.bincount(bin_idx[idx], weights=diff[idx], minlength=n_bins)
        eces[i] = np.abs(sums).sum() / n
    return eces


def two_sided_p(boot_delta: np.ndarray) -> float:
    p_lower = (boot_delta <= 0).mean()
    return 2 * min(p_lower, 1 - p_lower)


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# --------------------------------------------------------------------------
# Plot primitives
# --------------------------------------------------------------------------

def calibration_plot(ax, df: pd.DataFrame, title: str, series: list[Series],
                     outcome_col: str = "true_probability") -> None:
    """Reliability curve per series; marker size scales with bin count."""
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
    for s in series:
        model_df = df[df["model"] == s.key].copy()
        if model_df.empty:
            continue
        model_df["bin"] = pd.cut(model_df["Confidence"], bins=BIN_EDGES,
                                 right=False, labels=BIN_LABELS, include_lowest=True)
        bin_stats = model_df.groupby("bin", observed=False).agg(
            frac_correct=(outcome_col, "mean"),
            count=(outcome_col, "count"),
        ).reset_index()
        p = bin_stats["frac_correct"].values
        n = bin_stats["count"].values
        mask = n > 0
        if not mask.any():
            continue
        n_max = n[mask].max()
        sizes = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * n[mask] / n_max
        ax.plot(BIN_MIDPOINTS[mask], p[mask], color=s.color, linewidth=1.2, alpha=0.8)
        ax.scatter(BIN_MIDPOINTS[mask], p[mask], s=sizes, color=s.color, marker=s.marker,
                   label=s.label, edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Confidence")
    ax.set_title(title)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Figure builders — each returns the saved file path
# --------------------------------------------------------------------------

# Difficulty x-axis config per domain for the overconfidence-by-difficulty plot.
_DIFF_XCONFIG = {
    "WGD": ("Radius (lbs)", range(1, 21, 2)),
    "LifeEval": ("Radius (years)", range(1, 21, 2)),
    "MedEval": ("% Candidates Removed", [0, 10, 25, 50]),
}


def fig_calibration(dfs: dict[str, pd.DataFrame], series: list[Series],
                    out_dir: Path, spd: bool = False) -> Path:
    """3-panel reliability diagram (WGD, LifeEval, MedEval)."""
    suffix = " (SPD)" if spd else ""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), layout="constrained")
    for ax, domain in zip(axes, ["WGD", "LifeEval", "MedEval"]):
        calibration_plot(ax, dfs[domain], f"{domain}{suffix}", series)
        ax.set_aspect("auto")
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel("Proportion Correct")
    axes[0].legend(bbox_to_anchor=(0.02, 1), loc="upper left")
    name = "calibration_spd_combined.png" if spd else "calibration_combined.png"
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_overconfidence_by_difficulty(dfs: dict[str, pd.DataFrame], series: list[Series],
                                     out_dir: Path) -> Path:
    """Mean overconfidence vs. task difficulty, one panel per domain (DCE)."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), sharey=True, layout="constrained")
    for ax, domain in zip(axes, ["WGD", "LifeEval", "MedEval"]):
        ddf = dfs[domain]
        diff_col = difficulty_column(domain)
        xlabel, xticks = _DIFF_XCONFIG[domain]
        for s in series:
            sub = ddf[ddf["model"] == s.key]
            if sub.empty:
                continue
            grouped = sub.groupby(diff_col)["overconfidence"].mean()
            ax.plot(grouped.index, grouped.values, marker="o", markersize=3,
                    label=s.label, color=s.color, linewidth=1.2)
        ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(domain)
        ax.set_xticks(list(xticks))
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean Overconfidence")
    axes[0].legend(loc="upper left")
    path = out_dir / "overconfidence_by_difficulty.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_overconfidence_by_percentile(dfs: dict[str, pd.DataFrame], series: list[Series],
                                     out_dir: Path, spd: bool = False) -> Path:
    """OLS fit of overconfidence on difficulty percentile, one panel per domain."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.5), sharey=True, layout="constrained")
    for ax, domain in zip(axes, ["WGD", "LifeEval", "MedEval"]):
        ddf = add_difficulty(dfs[domain], domain)
        for s in series:
            sub = ddf[ddf["model"] == s.key].dropna(subset=["diff", "overconfidence"])
            if sub.empty:
                continue
            slope, intercept, _, _, _ = sp_stats.linregress(sub["diff"], sub["overconfidence"])
            x_line = np.array([0, 1])
            ax.plot(x_line, intercept + slope * x_line, color=s.color,
                    linewidth=1.2, label=s.label)
        ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_xlabel("Difficulty Percentile")
        ax.set_title(domain)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Overconfidence")
    axes[0].legend(loc="upper left")
    name = "rq3_overconfidence_by_percentile_spd.png" if spd else "rq2_overconfidence_by_percentile.png"
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _ece_sig_label(p: float) -> str:
    """Significance stars for the ECE brackets ('ns' when not significant)."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fig_ece_spd_improvement(dce: dict[str, pd.DataFrame], spd: dict[str, pd.DataFrame],
                            series: list[Series], out_dir: Path, seed: int = 42) -> Path:
    """Grouped bars: baseline ECE (solid) vs. SPD ECE (faded), per series per domain,
    with bootstrap significance brackets. A significant *worsening* under SPD is
    flagged with a red bracket and a dagger (matches the paper's rq3_ece_significance)."""
    domains = ["WGD", "LifeEval", "MedEval"]
    rng = np.random.default_rng(seed)

    # Precompute ECE + bootstrap significance per (domain, present series).
    stats: dict[str, list[dict]] = {}
    global_top = 0.0
    for domain in domains:
        base_df, spd_df = dce[domain], spd[domain]
        rows = []
        for s in series:
            b = base_df[base_df["model"] == s.key].dropna(subset=["Confidence", "true_probability"])
            p = spd_df[spd_df["model"] == s.key].dropna(subset=["Confidence", "true_probability"])
            if b.empty or p.empty:
                continue
            ece_base = compute_ece(b["Confidence"], b["true_probability"])
            ece_spd = compute_ece(p["Confidence"], p["true_probability"])
            boot_base = bootstrap_ece(b["Confidence"], b["true_probability"], rng=rng)
            boot_spd = bootstrap_ece(p["Confidence"], p["true_probability"], rng=rng)
            p_val = two_sided_p(boot_base - boot_spd)
            rows.append({
                "series": s, "ece_base": ece_base, "ece_spd": ece_spd,
                "p": p_val, "direction": "better" if ece_spd < ece_base else "worse",
            })
            global_top = max(global_top, ece_base, ece_spd)
        stats[domain] = rows

    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True, layout="constrained")
    bw = 0.38
    for ax, domain in zip(axes, domains):
        rows = stats[domain]
        x = np.arange(len(rows))
        colors = [r["series"].color for r in rows]
        ax.bar(x - bw / 2, [r["ece_base"] for r in rows], bw, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.bar(x + bw / 2, [r["ece_spd"] for r in rows], bw, color=colors,
               edgecolor="white", linewidth=0.5, alpha=0.4)

        for i, r in enumerate(rows):
            label = _ece_sig_label(r["p"])
            worse_sig = label != "ns" and r["direction"] == "worse"
            if worse_sig:
                label += "†"  # dagger
            y = max(r["ece_base"], r["ece_spd"]) + 0.008
            color = "#CC0000" if worse_sig else "#333333"
            ax.plot([i - bw / 2, i - bw / 2, i + bw / 2, i + bw / 2],
                    [y, y + 0.004, y + 0.004, y], color=color, linewidth=0.6)
            ax.text(i, y + 0.005, label, ha="center", va="bottom", fontsize=5.5, color=color)

        ax.set_title(domain)
        ax.set_xticks(x)
        ax.set_xticklabels([r["series"].label for r in rows], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylim(0, global_top + 0.06)  # headroom for the brackets/labels
    axes[0].set_ylabel("ECE")
    handles = [plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=1.0),
               plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.4)]
    axes[0].legend(handles, ["Baseline", "SPD"], loc="upper left")
    path = out_dir / "rq3_ece_spd_improvement.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_sex_bias_wgd(wgd_df: pd.DataFrame, labels_csv: Path, series: list[Series],
                     out_dir: Path) -> Path:
    """Directional weight-estimation error, Male minus Female, per series."""
    labels = pd.read_csv(labels_csv).dropna(subset=["sex"])
    sex_map = labels.set_index("Participant Number")["sex"]

    bias = wgd_df.copy()
    bias["Answer_num"] = pd.to_numeric(bias["Answer"], errors="coerce")
    bias["participant"] = bias["photo"].astype(str).str.replace(".jpg", "", regex=False)
    bias["sex"] = bias["participant"].map(sex_map)
    bias = bias.dropna(subset=["sex", "Answer_num", "true_weight"])
    bias["dir_error"] = bias["Answer_num"] - bias["true_weight"]
    # One error per (series, participant, sex); within_lbs doesn't move the point estimate.
    photo_error = bias.groupby(["model", "participant", "sex"])["dir_error"].mean().reset_index()

    present, deltas, cis, pvals = [], [], [], []
    for s in series:
        sub = photo_error[photo_error["model"] == s.key]
        err_m = sub.loc[sub["sex"] == "Male", "dir_error"]
        err_f = sub.loc[sub["sex"] == "Female", "dir_error"]
        if len(err_m) < 2 or len(err_f) < 2:
            continue
        delta = err_m.mean() - err_f.mean()
        n_m, n_f = len(err_m), len(err_f)
        se_diff = np.sqrt(err_m.var() / n_m + err_f.var() / n_f)
        df_welch = se_diff**4 / (
            (err_m.var() / n_m)**2 / (n_m - 1) + (err_f.var() / n_f)**2 / (n_f - 1)
        )
        _, p_val = sp_stats.ttest_ind(err_m, err_f, equal_var=False)
        present.append(s)
        deltas.append(delta)
        cis.append(sp_stats.t.ppf(0.975, df=df_welch) * se_diff)
        pvals.append(p_val)

    fig, ax = plt.subplots(figsize=(5.0, 3.0), layout="constrained")
    x = np.arange(len(present))
    deltas = np.array(deltas)
    cis = np.array(cis)
    ax.bar(x, deltas, 0.55, yerr=cis, capsize=4, color=[s.color for s in present],
           edgecolor="white", linewidth=0.5, error_kw={"linewidth": 0.8})
    ax.axhline(0, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    for i, p in enumerate(pvals):
        label = f"p={p:.3f}" if p >= 0.001 else f"p={p:.1e}"
        y = deltas[i] + cis[i] if deltas[i] >= 0 else deltas[i] - cis[i]
        va = "bottom" if deltas[i] >= 0 else "top"
        offset = 0.3 if deltas[i] >= 0 else -0.3
        ax.text(i, y + offset, label, ha="center", va=va, fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([s.label for s in present], rotation=20, ha="right")
    ax.set_ylabel("Δ Mean Error, M − F (lbs)")
    ax.set_title("Sex Bias in Weight Estimation (Δ = $\\bar{E}_M - \\bar{E}_F$)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    path = out_dir / "posthoc_sex_bias_wgd.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_wgd_demographics(labels_csv: Path, out_dir: Path) -> list[Path]:
    """WGD subject demographics (model-independent): weight/age violins + pies."""
    labels = pd.read_csv(labels_csv).dropna(subset=["Age", "Weight", "sex", "ethnicity"])

    # Weight/age distributions (matplotlib violinplot, no seaborn dependency).
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5), layout="constrained")
    for ax, col, unit, color in [
        (axes[0], "Weight", "lbs", "#4E79A7"),
        (axes[1], "Age", "years", "#CC79A7"),
    ]:
        vals = labels[col].to_numpy(dtype=float)
        parts = ax.violinplot(vals, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.7)
        parts["cmedians"].set_color("#333333")
        ax.set_ylabel(f"{col} ({unit})")
        ax.set_xticks([])
        ax.set_title(f"{col} Distribution\n$\\mu$ = {vals.mean():.1f}, $\\sigma$ = {vals.std(ddof=1):.1f}")
    path1 = out_dir / "wgd_weight_age.png"
    fig.savefig(path1)
    plt.close(fig)

    # Ethnicity + sex pies.
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.5), layout="constrained")
    eth_counts = labels["ethnicity"].value_counts()
    eth_pcts = eth_counts / eth_counts.sum()
    eth_colors = {
        "Asian": "#4E79A7", "White": "#E15759", "Indian": "#76B7B2",
        "Hispanic": "#F28E2B", "Black": "#59A14F", "Middle Eastern": "#EDC948",
        "Other": "#BAB0AC",
    }
    axes[0].pie(eth_counts, labels=[e if eth_pcts[e] >= 0.05 else "" for e in eth_counts.index],
                colors=[eth_colors.get(e, "#BAB0AC") for e in eth_counts.index], startangle=90,
                autopct=lambda p: f"{p:.0f}%" if p >= 9 else "")
    axes[0].set_title("Ethnicity")
    sex_counts = labels["sex"].value_counts()
    axes[1].pie(sex_counts, labels=sex_counts.index, autopct="%1.0f%%", startangle=90,
                colors=["#4E79A7", "#BB6BB6"])
    axes[1].set_title("Sex")
    path2 = out_dir / "wgd_demographics.png"
    fig.savefig(path2)
    plt.close(fig)
    return [path1, path2]


# --------------------------------------------------------------------------
# Summary table (RQ1 + RQ2 β₁ + RQ3 ECE_SPD/ΔECE) with humans as rows
# --------------------------------------------------------------------------

def build_summary(dce: dict[str, pd.DataFrame], spd: dict[str, pd.DataFrame],
                  series: list[Series], seed: int = 42) -> pd.DataFrame:
    """Tidy summary table, one row per (domain, series) present in the DCE data."""
    rng = np.random.default_rng(seed)
    rows = []
    for domain in ["WGD", "LifeEval", "MedEval"]:
        ddf = dce[domain]
        sdf = spd[domain]
        ddf_diff = add_difficulty(ddf, domain)
        for s in series:
            sub = ddf[ddf["model"] == s.key]
            if sub.empty:
                continue
            mean_conf = sub["Confidence"].mean()
            mean_acc = sub["true_probability"].mean()
            ece_base = compute_ece(sub["Confidence"], sub["true_probability"])

            # RQ2: OLS slope of overconfidence on difficulty percentile.
            subd = ddf_diff[ddf_diff["model"] == s.key].dropna(subset=["diff", "overconfidence"])
            slope, _, _, p_beta, _ = sp_stats.linregress(subd["diff"], subd["overconfidence"])

            # RQ3: SPD ECE and bootstrap significance of the change (if SPD data exists).
            sub_spd = sdf[sdf["model"] == s.key]
            if sub_spd.empty:
                ece_spd = np.nan
                pct_change = np.nan
                ece_sig = ""
            else:
                ece_spd = compute_ece(sub_spd["Confidence"], sub_spd["true_probability"])
                pct_change = (ece_spd - ece_base) / ece_base * 100 if ece_base else np.nan
                boot_base = bootstrap_ece(sub["Confidence"], sub["true_probability"], rng=rng)
                boot_spd = bootstrap_ece(sub_spd["Confidence"], sub_spd["true_probability"], rng=rng)
                ece_sig = sig_label(two_sided_p(boot_base - boot_spd))

            rows.append({
                "Domain": domain,
                "Model": s.label,
                "Accuracy": mean_acc,
                "Mean Confidence": mean_conf,
                "Overconfidence": mean_conf - mean_acc,
                "beta1": slope,
                "beta1_sig": sig_label(p_beta),
                "ECE": ece_base,
                "ECE_SPD": ece_spd,
                "Pct_Change": pct_change,
                "ece_sig": ece_sig,
            })
    return pd.DataFrame(rows)


def _fmt_signed(v: float, decimals: int = 3) -> str:
    if pd.isna(v):
        return "---"
    s = f"{abs(v):.{decimals}f}"
    return f"$-${s}" if v < 0 else f"$+${s}" if v > 0 else s


def summary_to_latex(summary: pd.DataFrame) -> str:
    """RQ1/RQ3 summary LaTeX table matching the notebook's rq1_summary_table.txt."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary of model calibration across domains. Accuracy denotes mean correctness (WGD) or mean true probability (LifeEval, MedEval). Overconfidence = Mean Confidence $-$ Accuracy. $\beta_1$ is the OLS slope of overconfidence on difficulty percentile. ECE\textsubscript{SPD} and $\Delta$ECE show the effect of subjective probability distribution prompting.}",
        r"\label{tab:rq1_summary}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Domain & Model & Accuracy & Mean Conf. & Overconf. & $\beta_1$ & ECE & ECE\textsubscript{SPD} & $\Delta$ECE (\%) \\",
        r"\midrule",
    ]
    domains = [d for d in ["WGD", "LifeEval", "MedEval"] if (summary["Domain"] == d).any()]
    for di, domain in enumerate(domains):
        sub = summary[summary["Domain"] == domain]
        lines.append(rf"\multirow{{{len(sub)}}}{{*}}{{{domain}}}")
        for _, r in sub.iterrows():
            oc = _fmt_signed(r["Overconfidence"])
            b1 = _fmt_signed(r["beta1"])
            pct = _fmt_signed(r["Pct_Change"], 1)
            ece_spd = "---" if pd.isna(r["ECE_SPD"]) else f"{r['ECE_SPD']:.3f}"
            lines.append(
                f" & {r['Model']:20s} & {r['Accuracy']:.3f} & {r['Mean Confidence']:.3f} "
                f"& {oc} & {b1}\\textsuperscript{{{r['beta1_sig']}}} "
                f"& {r['ECE']:.3f} & {ece_spd} & {pct}\\textsuperscript{{{r['ece_sig']}}} \\\\"
            )
        if di != len(domains) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\multicolumn{9}{l}{\footnotesize \textsuperscript{*}$p<0.05$, \textsuperscript{**}$p<0.01$, \textsuperscript{***}$p<0.001$} \\",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def summary_to_png(summary: pd.DataFrame, out_path: Path) -> Path:
    """Render the summary table as a PNG so a non-technical user can read it."""
    disp = summary.copy()
    disp["ΔECE (%)"] = disp["Pct_Change"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}")
    disp["ECE_SPD"] = disp["ECE_SPD"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    disp["β₁"] = disp.apply(lambda r: f"{r['beta1']:+.3f}{r['beta1_sig']}", axis=1)
    for col in ["Accuracy", "Mean Confidence", "Overconfidence", "ECE"]:
        disp[col] = disp[col].map(lambda v: f"{v:.3f}")
    disp = disp.rename(columns={"Mean Confidence": "Mean Conf.", "Overconfidence": "Overconf.",
                                "ECE_SPD": "ECE (SPD)"})
    cols = ["Domain", "Model", "Accuracy", "Mean Conf.", "Overconf.",
            "β₁", "ECE", "ECE (SPD)", "ΔECE (%)"]
    disp = disp[cols]

    fig, ax = plt.subplots(figsize=(12, 0.4 * len(disp) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    tbl.auto_set_column_width(range(len(cols)))
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#0072B2")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
