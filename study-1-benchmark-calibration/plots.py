"""Regenerate every figure under Plots/ from the combined study-1 results.

Usage (from anywhere, after `python combine.py && python clean.py`):

    python plots.py

Reads   Combined Results/combined_clean.csv  (LifeEval-free; see clean.py)
Writes  the full Plots/ tree:
    Plots/<Benchmark>/Calibration Plots/   per-model calibration plots
                                           (+ _tokens variants where token
                                           probabilities are available)
    Plots/<Benchmark>/summary_bars_*.png   2x2 ECE / overconfidence /
                                           accuracy / n panels
    Plots/Main Plots/                      aggregate calibration plots
    Plots/Summary Plots/                   cross-benchmark comparisons

The plotting logic is consolidated from analysis.ipynb and
Workflow/get_results_analysis.ipynb so the whole tree can be rebuilt in one
deterministic pass.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
CLEAN_CSV = ROOT / "Combined Results" / "combined_clean.csv"
PLOTS = ROOT / "Plots"

QSET_SLUGS = {
    "BoolQ": "boolq",
    "HaluEval": "halu_eval",
    "LSAT-AR": "lsat_ar",
    "SAT-EN": "sat_en",
    "SciQ": "sciq",
}
MCQ_QSETS = ["LSAT-AR", "SAT-EN", "SciQ"]
MCQ_LETTERS = {"LSAT-AR": ["A", "B", "C", "D", "E"], "SAT-EN": ["A", "B", "C", "D"], "SciQ": ["A", "B", "C", "D"]}

MODEL_ORDER = [
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "GPT-o3",
    "GPT-4o",
    "Gemini-2.5-Pro",
    "Gemini-2.5-Flash",
    "DeepSeek-V3",
    "DeepSeek-R1",
    "Claude-Sonnet-4",
    "Claude-Sonnet-3.7",
    "Claude Haiku 3",
]
REASONING_MODELS = ["Claude-Sonnet-3.7", "Claude-Sonnet-4", "DeepSeek-R1", "Gemini-2.5-Pro", "GPT-o3"]
TOKEN_MODELS = ["GPT-4o", "Llama-3.1-8B", "Llama-3.1-70B"]

FAMILY_PALETTES = {
    "GPT": sns.color_palette("Greens", 6),
    "Claude": sns.color_palette("Blues", 6),
    "Gemini": sns.color_palette("Purples", 6),
    "DeepSeek": sns.color_palette("Oranges", 6),
    "Llama": sns.color_palette("RdPu", 6),
}


def model_family(name: str) -> str:
    s = name.lower()
    if "gpt" in s or "o3" in s:
        return "GPT"
    if "claude" in s:
        return "Claude"
    if "gemini" in s:
        return "Gemini"
    if "deepseek" in s:
        return "DeepSeek"
    if "llama" in s:
        return "Llama"
    return "other"


def pick_color(name: str) -> tuple:
    fam = model_family(name)
    pal = FAMILY_PALETTES[fam]
    s = name.lower()
    if fam == "GPT":
        if "gpt-4o" in s:
            return pal[-2]
        if "o3" in s:
            return pal[1]
    if fam == "Claude":
        if "sonnet" in s:
            return pal[-2]
        if "haiku" in s:
            return pal[2]
    if fam == "Gemini":
        if "pro" in s:
            return pal[-2]
        if "flash" in s:
            return pal[2]
    if fam == "DeepSeek":
        if "r1" in s:
            return pal[-2]
        if "v3" in s:
            return pal[2]
    return pal[3]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def get_ece(score: pd.Series, confidence: pd.Series, n_bins: int = 10) -> float:
    score = pd.Series(score).astype(float).reset_index(drop=True)
    confidence = pd.Series(confidence).astype(float).reset_index(drop=True)
    keep = score.notna() & confidence.notna()
    score, confidence = score[keep], confidence[keep]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges, right=True)

    ece, total = 0.0, len(score)
    for i in range(1, n_bins + 1):
        mask = bin_ids == i
        if mask.sum() > 0:
            ece += (mask.sum() / total) * abs(score[mask].mean() - confidence[mask].mean())
    return ece


def melt_series(score, confidence) -> pd.DataFrame:
    """Bin (confidence, score) pairs into 0.1-wide bins for reliability plots."""
    df = pd.DataFrame({"confidence": np.asarray(confidence, dtype=float), "score": np.asarray(score, dtype=float)})
    df = df.dropna()
    df["confidence"] = df["confidence"].clip(0, 1)
    bins = np.append(np.linspace(0, 1, 11), 1.0000001)
    labels = [round(x, 1) for x in np.linspace(0.1, 1.0, 11)]
    df["bin"] = pd.cut(df["confidence"], bins=bins, labels=labels, ordered=False, right=False)
    return (
        df.groupby("bin", observed=False)
        .agg(mean_score=("score", "mean"), count=("score", "size"), mean_confidence=("confidence", "mean"))
        .reset_index()
    )


def conf_col(qset: str) -> str:
    return "Stated Confidence Answer (MCQ)" if qset in MCQ_QSETS else "Stated Confidence Answer"


def expand_mcq(df: pd.DataFrame, qset: str, tokens: bool = False):
    """Per-option (score, confidence) pairs: every option letter contributes the
    confidence assigned to it, scored 1 iff it is the correct answer."""
    letters = MCQ_LETTERS[qset]
    prefix = "Token Probability" if tokens else "Stated Confidence"
    scores, confs = [], []
    correct = df["Correct Answer"].astype(str).str.strip().str.upper()
    for letter in letters:
        confs.append(pd.to_numeric(df[f"{prefix} {letter}"], errors="coerce"))
        scores.append((correct == letter).astype(float))
    return pd.concat(scores).to_numpy(), pd.concat(confs).to_numpy()


def expand_bool(df: pd.DataFrame, tokens: bool = False):
    """Chosen answer plus its complement: (score, conf) and (1-score, 1-conf)."""
    if tokens:
        conf = np.where(
            df["Answer"].astype(str).str.strip() == "True",
            pd.to_numeric(df["Token Probability True"], errors="coerce"),
            pd.to_numeric(df["Token Probability False"], errors="coerce"),
        )
    else:
        conf = pd.to_numeric(df["Stated Confidence Answer"], errors="coerce").to_numpy()
    score = df["Score"].astype(float).to_numpy()
    return np.concatenate([score, 1.0 - score]), np.concatenate([conf, 1.0 - conf])


def extended_pairs(df: pd.DataFrame, qset: str, tokens: bool = False):
    if qset in MCQ_QSETS:
        return expand_mcq(df, qset, tokens=tokens)
    return expand_bool(df, tokens=tokens)


# --------------------------------------------------------------------------
# Plot primitives
# --------------------------------------------------------------------------

def save(fig, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def per_model_cal_plot(scores, confidence, title: str, out_path: Path):
    """Single-model reliability diagram: 45-degree line, confidence histogram,
    binned accuracy with binomial standard errors, twin proportion axis."""
    ece = get_ece(pd.Series(scores), pd.Series(confidence))
    acc = float(np.nanmean(np.asarray(scores, dtype=float)))
    n = int(np.isfinite(np.asarray(confidence, dtype=float)).sum())
    melted = melt_series(scores, confidence)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1.1], [0, 1.1], linestyle="--", color="grey")
    ax.bar(
        melted["mean_confidence"].fillna(0),
        melted["count"].fillna(0) / max(melted["count"].sum(), 1),
        width=0.025,
        align="center",
        edgecolor="white",
        color="orange",
    )
    stderr = np.sqrt((melted["mean_score"] * (1 - melted["mean_score"])) / melted["count"]).fillna(0)
    ax.errorbar(melted["mean_confidence"], melted["mean_score"], yerr=stderr, fmt="o", ecolor="grey", capsize=5)
    ax.scatter(melted["mean_confidence"], melted["mean_score"], color="blue")

    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Stated Confidence")
    ax.set_ylabel("Average Accuracy", labelpad=15)
    ax.set_title(f"{title}\nECE: {ece:.3f} | Accuracy: {acc:.3f} | n = {n}")
    ticks = [round(0.1 * i, 1) for i in range(11)]
    ax.set_xticks(ticks)

    ax2 = ax.twinx()
    ax2.set_ylabel("Proportion of Stated Confidence", rotation=-90, labelpad=15)
    ax2.set_ylim(0, 1.1)

    save(fig, out_path)
    return ece, acc


def aggregate_cal_axis(scores, confidence, ax, color="black", label="Stated Confidence", legend=True):
    """Aggregate reliability curve with 95% CIs (analysis.ipynb style)."""
    melted = melt_series(scores, confidence)
    se = melted["mean_score"].mul(0).add(  # std of binary outcome per bin
        np.sqrt((melted["mean_score"] * (1 - melted["mean_score"])).clip(lower=0) / melted["count"])
    )
    ci = 1.96 * se

    ticks = [round(0.1 * i, 1) for i in range(11)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.axline([0, 0], [1, 1], color="grey", linestyle="--", label="Line of Perfect Calibration", alpha=0.5)
    ax.errorbar(
        melted["mean_confidence"], melted["mean_score"], yerr=ci, fmt="none", linewidth=2, color="lightgrey", ecolor="lightgrey"
    )
    sns.scatterplot(x=melted["mean_confidence"], y=melted["mean_score"], label=label, color=color, zorder=3, ax=ax)
    ax.grid(True, linestyle="--", alpha=0.35)
    if legend:
        ax.legend(loc="lower right")
    else:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)


def summary_bars(qset: str, stats: pd.DataFrame, n_used: int, out_path: Path):
    """2x2 panel of ECE / overconfidence / accuracy / usable-n by model."""
    stats = stats.sort_values(["family", "model"]).reset_index(drop=True)
    panels = [
        ("ece", "ECE", f"{qset} — Expected Calibration Error by Model"),
        ("overconfidence", "Over Confidence", f"{qset} — Overconfidence by Model"),
        ("accuracy", "Accuracy (%)", f"{qset} — Accuracy by Model"),
        ("n", "Count (n)", f"{qset} — Usable Rows by Model"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(max(12, 0.55 * len(stats) + 8), 10))
    for ax, (col, ylabel, title) in zip(axes.flatten(), panels):
        ax.bar(stats["model"], stats[col], color=list(stats["color"]), linewidth=0, zorder=2)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if col == "ece":
            ax.set_ylim(0, 1)
        else:
            lo = min(0.0, float(stats[col].min()) * 1.1)
            ax.set_ylim(lo, max(1.0, float(stats[col].max()) * 1.1))
            if lo < 0:
                ax.axhline(0, color="black", linewidth=0.75)
        if col == "n":
            ax.axhline(n_used, color="grey", linestyle="--", linewidth=1.5)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats["model"], rotation=45, ha="right")
        ax.grid(zorder=0)

    patches = [Patch(facecolor=FAMILY_PALETTES[f][3], label=f) for f in stats["family"].unique() if f in FAMILY_PALETTES]
    axes.flatten()[0].legend(handles=patches, title="Model Families", frameon=True, loc="upper left")
    fig.suptitle(f"Summary Statistics for {qset}", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, out_path)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def per_benchmark_plots(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model calibration plots + per-benchmark summary bars.
    Returns the tidy per-(qset, model) summary-statistics table."""
    rows = []
    for qset, slug in QSET_SLUGS.items():
        print(f"{qset}:")
        qdf = df[df["Question Set"] == qset]
        n_used = qdf["Question ID"].nunique()
        cal_dir = PLOTS / qset / "Calibration Plots"

        for model in MODEL_ORDER:
            mdf = qdf[qdf["Model"] == model]
            if mdf.empty:
                continue
            conf = pd.to_numeric(mdf[conf_col(qset)], errors="coerce")
            score = mdf["Score"].astype(float)
            ece, acc = per_model_cal_plot(
                score, conf, f"Calibration Plot for {model} on {qset}", cal_dir / f"cal_plot_{slug}_{model}.png"
            )
            rows.append(
                dict(
                    qset=qset,
                    model=model,
                    family=model_family(model),
                    color=pick_color(model),
                    ece=ece,
                    accuracy=acc * 100,
                    overconfidence=float(conf.mean() - score.mean()),
                    n=len(mdf),
                )
            )

            # Token-probability variants (models that expose logprobs; HaluEval
            # has no per-option token probabilities)
            if model in TOKEN_MODELS and qset != "HaluEval":
                t_score, t_conf = extended_pairs(mdf, qset, tokens=True)
                per_model_cal_plot(
                    t_score,
                    t_conf,
                    f"Calibration Plot for {model}'s Tokens on {qset}",
                    cal_dir / f"cal_plot_{slug}_{model}_tokens.png",
                )

        stats = pd.DataFrame([r for r in rows if r["qset"] == qset])
        summary_bars(qset, stats, n_used, PLOTS / qset / f"summary_bars_{slug}.png")
    return pd.DataFrame(rows)


def main_plots(df: pd.DataFrame):
    print("Main Plots:")
    out = PLOTS / "Main Plots"

    # Aggregate over all models: extended MCQ options + 2AFC complements
    combos = {
        "aggregate_extended_all_qsets_cal_plot.png": (
            "Aggregate Calibration Plot for all Models & Question Sets\nExtended MCQ & 2AFC responses",
            list(QSET_SLUGS),
        ),
        "aggregate_extended_mcq_cal_plot.png": (
            "Aggregate Calibration Plot on Only MCQ Question Sets",
            MCQ_QSETS,
        ),
        "aggregate_extended_2afc_cal_plot.png": (
            "Aggregate Calibration Plot on Only 2AFC Question Sets",
            ["BoolQ", "HaluEval"],
        ),
    }
    for filename, (title, qsets) in combos.items():
        scores, confs = [], []
        for qset in qsets:
            s, c = extended_pairs(df[df["Question Set"] == qset], qset)
            scores.append(s)
            confs.append(c)
        fig, ax = plt.subplots(figsize=(7, 7))
        aggregate_cal_axis(np.concatenate(scores), np.concatenate(confs), ax)
        ax.set_title(title)
        save(fig, out / filename)

    # Reasoning vs non-reasoning overlay, one panel per benchmark
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    for ax, qset in zip(axes.flatten(), QSET_SLUGS):
        qdf = df[df["Question Set"] == qset].copy()
        conf = pd.to_numeric(qdf[conf_col(qset)], errors="coerce").fillna(0)
        for is_reasoning, color, label in ((False, "black", "Non-Reasoning"), (True, "red", "Reasoning")):
            mask = qdf["Model"].isin(REASONING_MODELS) == is_reasoning
            aggregate_cal_axis(qdf.loc[mask, "Score"].astype(float), conf[mask], ax, color=color, label=label, legend=False)
        ax.set_title(qset)
    axes.flatten()[-1].axis("off")
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="lower right", fontsize="large")
    fig.suptitle("Reasoning vs Non-Reasoning Calibration by Question Set", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, out / "reasoning_vs_chat_by_qset_cal_plot.png", dpi=150)

    # Full model x benchmark grid (stated confidence, token overlay in red)
    fig, axes = plt.subplots(len(MODEL_ORDER), len(QSET_SLUGS), figsize=(len(QSET_SLUGS) * 5, len(MODEL_ORDER) * 4), squeeze=False)
    for i, model in enumerate(MODEL_ORDER):
        for j, qset in enumerate(QSET_SLUGS):
            ax = axes[i, j]
            sub = df[(df["Model"] == model) & (df["Question Set"] == qset)]
            if sub.empty:
                ax.set_title(f"No data for {model} - {qset}")
                continue
            s, c = extended_pairs(sub, qset)
            aggregate_cal_axis(s, c, ax, legend=False)
            if model in TOKEN_MODELS and qset != "HaluEval":
                ts, tc = extended_pairs(sub, qset, tokens=True)
                aggregate_cal_axis(ts, tc, ax, color="red", label="Token Probability", legend=False)
            ax.set_title(f"Model: {model} | Set: {qset}")
    handles_labels = {}
    for ax in axes.flatten():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles_labels.setdefault(l, h)
    fig.suptitle("Calibration Plots for all Models on all Question Sets", fontsize=48, y=0.995)
    fig.legend(handles_labels.values(), handles_labels.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, prop={"size": 24})
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    save(fig, out / "calibration_grid_all_models_all_qsets.png", dpi=100)


def summary_plots(df: pd.DataFrame, stats: pd.DataFrame):
    print("Summary Plots:")
    out = PLOTS / "Summary Plots"
    out.mkdir(parents=True, exist_ok=True)

    # One solid hue per family; reasoning models keep it bold, chat models get
    # the same hue faded (patch alpha below), so lightness encodes model type
    chat_alpha = 0.35
    hatch_alpha = 0.6
    palette = {m: FAMILY_PALETTES[model_family(m)][4] for m in stats["model"].unique()}
    reasoning_handles = [
        Patch(facecolor=(0.3, 0.3, 0.3, 1.0), label="Reasoning"),
        Patch(facecolor=(0.3, 0.3, 0.3, chat_alpha), edgecolor=(0.3, 0.3, 0.3, hatch_alpha), linewidth=0, hatch="//", label="Chat"),
    ]
    for col, label, filename in (
        ("ece", "ECE", "ece_all.png"),
        ("accuracy", "Accuracy (%)", "acc_all.png"),
        ("overconfidence", "Overconfidence", "oc_all.png"),
    ):
        g = sns.catplot(
            data=stats,
            kind="bar",
            x=col,
            y="model",
            col="qset",
            col_wrap=3,
            order=MODEL_ORDER,
            hue="model",
            palette=palette,
            height=3.2,
            aspect=1.1,
        )
        g.set_titles(col_template="{col_name}")
        g.set(xlabel=label, ylabel="")
        for ax in g.axes.flat:
            ax.axvline(0, linestyle="--", linewidth=1, color="0.2")
            ax.grid(True, axis="x", linestyle=":", alpha=0.5)
            # seaborn drops alpha from palette colors, so fade chat bars here;
            # each bar's y-center is its model's index in MODEL_ORDER
            for patch in ax.patches:
                idx = int(round(patch.get_y() + patch.get_height() / 2))
                if 0 <= idx < len(MODEL_ORDER) and MODEL_ORDER[idx] not in REASONING_MODELS:
                    # fade the fill only; hatch draws in the full-strength
                    # family color (hatch follows edgecolor, not facecolor)
                    fc = patch.get_facecolor()
                    patch.set_facecolor((fc[0], fc[1], fc[2], chat_alpha))
                    patch.set_edgecolor((fc[0], fc[1], fc[2], hatch_alpha))
                    patch.set_linewidth(0)
                    patch.set_hatch("//")
        g.figure.suptitle(f"{label} Across Question Set by Model", y=1.02)
        # center the legend in the empty sixth facet slot (below LSAT-AR)
        slot_x = g.axes.flat[2].get_position()
        slot_y = g.axes.flat[3].get_position()
        g.figure.legend(
            handles=reasoning_handles,
            title="Model Type",
            loc="center",
            frameon=True,
            fontsize="large",
            title_fontsize="large",
            bbox_to_anchor=(slot_x.x0 + slot_x.width / 2, slot_y.y0 + slot_y.height / 2),
            bbox_transform=g.figure.transFigure,
        )
        g.figure.savefig(out / filename, dpi=300, bbox_inches="tight")
        plt.close(g.figure)
        print(f"  wrote {out / filename}")

    # Stated vs token ECE, extended scoring, token-capable models only
    marker_map = {"BoolQ": "o", "LSAT-AR": "D", "SAT-EN": "^", "SciQ": "v"}
    color_map = {"GPT-4o": "tab:blue", "Llama-3.1-70B": "tab:orange", "Llama-3.1-8B": "tab:green"}
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(True, linestyle="--", alpha=0.6)
    pts = []
    for model in TOKEN_MODELS:
        for qset in marker_map:
            sub = df[(df["Model"] == model) & (df["Question Set"] == qset)]
            if sub.empty:
                continue
            s, c = extended_pairs(sub, qset)
            ts, tc = extended_pairs(sub, qset, tokens=True)
            stated_ece, token_ece = get_ece(pd.Series(s), pd.Series(c)), get_ece(pd.Series(ts), pd.Series(tc))
            pts.append(max(stated_ece, token_ece))
            ax.scatter(stated_ece, token_ece, color=color_map[model], marker=marker_map[qset], s=100)
    lim = max(pts) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5)
    ax.set_xlabel("Stated ECE")
    ax.set_ylabel("Token ECE")
    ax.set_title("Stated vs Token ECE by Model")
    model_handles = [plt.Line2D([], [], color=c, marker="o", linestyle="", markersize=10, label=m) for m, c in color_map.items()]
    qset_handles = [plt.Line2D([], [], color="k", marker=mk, linestyle="", markersize=10, label=q) for q, mk in marker_map.items()]
    ax.legend(handles=model_handles + qset_handles, loc="lower right")
    save(fig, out / "stated_vs_token_ece.png")

    # Side-by-side strips of the token calibration plots per token model
    for model in TOKEN_MODELS:
        paths = [
            PLOTS / qset / "Calibration Plots" / f"cal_plot_{QSET_SLUGS[qset]}_{model}_tokens.png"
            for qset in ["LSAT-AR", "SAT-EN", "SciQ", "BoolQ"]
        ]
        paths = [p for p in paths if p.exists()]
        if not paths:
            continue
        fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for ax, p in zip(np.atleast_1d(axes), paths):
            ax.imshow(mpimg.imread(p))
            ax.axis("off")
        fig.suptitle(f"Token Calibration Plots for {model}", fontsize=14, y=0.85)
        fig.tight_layout()
        save(fig, out / f"combined_token_cal_plots_{model}.png", dpi=150)


def main():
    df = pd.read_csv(CLEAN_CSV)
    if (df["Question Set"] == "LifeEval").any():
        raise SystemExit("combined_clean.csv still contains LifeEval rows — re-run combine.py and clean.py first.")
    sns.set_theme(style="whitegrid")
    stats = per_benchmark_plots(df)
    main_plots(df)
    summary_plots(df, stats)
    stats.to_csv(PLOTS / "summary_stats.csv", index=False)
    print(f"Done. Summary statistics written to {PLOTS / 'summary_stats.csv'}")


if __name__ == "__main__":
    main()
