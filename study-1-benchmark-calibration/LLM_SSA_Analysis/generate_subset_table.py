"""
generate_subset_table.py
------------------------
Computes LifeEval metrics for the no_evidence subset (from judge results)
and writes a LaTeX table to subset_aggregate.tex.

Usage:
    python generate_subset_table.py
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

REPO = Path(__file__).resolve().parent.parent
COMBINED = REPO / "Combined Results" / "combined_clean.csv"
JUDGE = Path(__file__).resolve().parent / "sonnet_judge_results.csv"
OUTPUT_TEX = Path(__file__).resolve().parent / "subset_aggregate.tex"

MODEL_TYPE = {
    "Claude-Sonnet-3.7": "Reasoning",
    "Claude-Sonnet-4":   "Reasoning",
    "DeepSeek-R1":       "Reasoning",
    "Gemini-2.5-Pro":    "Reasoning",
    "GPT-o3":            "Reasoning",
    "Claude Haiku 3":    "Chat",
    "DeepSeek-V3":       "Chat",
    "Gemini-2.5-Flash":  "Chat",
    "GPT-4o":            "Chat",
    "Llama-3.1-70B":     "Chat",
    "Llama-3.1-8B":      "Chat",
}

REASONING_ORDER = [
    "Claude-Sonnet-3.7", "Claude-Sonnet-4", "DeepSeek-R1",
    "Gemini-2.5-Pro", "GPT-o3",
]
CHAT_ORDER = [
    "Claude Haiku 3", "DeepSeek-V3", "Gemini-2.5-Flash",
    "GPT-4o", "Llama-3.1-70B", "Llama-3.1-8B",
]


def compute_ece(score: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
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


def pct_rounded(confidence: np.ndarray) -> float:
    """Percentage of confidence values that are multiples of 0.05."""
    remainder = np.abs(np.round(confidence * 20) - confidence * 20)
    return 100.0 * (remainder < 1e-6).mean()


def regression_coef(overconfidence: np.ndarray, difficulty: np.ndarray) -> float:
    """OLS slope of overconfidence ~ difficulty."""
    valid = np.isfinite(overconfidence) & np.isfinite(difficulty)
    if valid.sum() < 2:
        return np.nan
    slope, _, _, _, _ = sp_stats.linregress(difficulty[valid], overconfidence[valid])
    return slope


def compute_model_metrics(df: pd.DataFrame, mas_map: dict) -> dict:
    score = df["Score"].astype(float).values
    conf = df["Stated Confidence Answer"].astype(float).values
    n = len(df)

    ece = compute_ece(score, conf)
    mean_score = score.mean()
    mean_conf = conf.mean()
    rnd = pct_rounded(conf)

    # Regression: overconfidence vs difficulty
    qids = df["Question ID"].astype(str).values
    difficulty = np.array([1.0 - mas_map.get(q, np.nan) for q in qids])
    overconf = conf - score
    reg = regression_coef(overconf, difficulty)

    return {
        "Score": mean_score * 100,
        "ECE": ece,
        "Conf": mean_conf * 100,
        "PctRnd": rnd,
        "RegCoef": reg,
        "N": n,
    }


def fmt(val, decimals=1) -> str:
    return f"{val:.{decimals}f}"


def build_tex_row(model: str, mtype: str, m: dict) -> str:
    return (
        f"    {model:<20s} & {mtype} "
        f"& {fmt(m['Score'])} & {fmt(m['ECE'], 3)} "
        f"& {fmt(m['Conf'])} & {fmt(m['PctRnd'])}  "
        f"& {fmt(m['RegCoef'], 3)} & {m['N']} \\\\"
    )


def build_agg_row(metrics_list: list, common_n: int) -> dict:
    """Aggregate = mean of per-model metrics, N = common set size."""
    keys = ["Score", "ECE", "Conf", "PctRnd", "RegCoef"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if np.isfinite(m[k])]
        agg[k] = np.mean(vals) if vals else np.nan
    agg["N"] = common_n
    return agg


def main(verdicts=None):
    if verdicts is None:
        verdicts = ["no_evidence"]

    subset_label = "+".join(verdicts)
    tex_label = ", ".join(r"\texttt{" + v.replace("_", r"\_") + "}" for v in verdicts)
    output_tex = Path(__file__).resolve().parent / f"subset_aggregate_{subset_label}.tex"

    print(f"Loading data... (verdicts: {verdicts})")
    combined = pd.read_csv(COMBINED, low_memory=False)
    life_eval = combined[combined["Question Set"] == "LifeEval"].copy()
    print(f"  LifeEval rows: {len(life_eval)}")

    judge = pd.read_csv(JUDGE, usecols=["Model", "Question ID", "verdict"])
    life_eval["Question ID"] = life_eval["Question ID"].astype(str)
    judge["Question ID"] = judge["Question ID"].astype(str)

    merged = life_eval.merge(judge, on=["Model", "Question ID"], how="inner")
    subset = merged[merged["verdict"].isin(verdicts)].copy()
    print(f"  {subset_label} rows: {len(subset)}")

    # Compute MAS per question from the FULL dataset (all models, all verdicts)
    life_eval["Score"] = life_eval["Score"].astype(float)
    mas = life_eval.groupby("Question ID")["Score"].max().to_dict()

    # Per-model metrics
    all_metrics = {}
    for model in REASONING_ORDER + CHAT_ORDER:
        mdf = subset[subset["Model"] == model]
        if len(mdf) == 0:
            print(f"  WARNING: {model} has 0 no_evidence rows, skipping")
            continue
        all_metrics[model] = compute_model_metrics(mdf, mas)
        print(f"  {model:25s}  n={all_metrics[model]['N']:>5d}  "
              f"Score={all_metrics[model]['Score']:.1f}%  "
              f"ECE={all_metrics[model]['ECE']:.3f}")

    # Common question sets per type group
    def common_n(model_list):
        sets = []
        for m in model_list:
            mdf = subset[subset["Model"] == m]
            sets.append(set(mdf["Question ID"].astype(str).values))
        if not sets:
            return 0
        return len(set.intersection(*sets))

    reasoning_common = common_n(REASONING_ORDER)
    chat_common = common_n(CHAT_ORDER)
    print(f"\n  Reasoning common no_evidence questions: {reasoning_common}")
    print(f"  Chat common no_evidence questions: {chat_common}")

    reasoning_metrics = [all_metrics[m] for m in REASONING_ORDER if m in all_metrics]
    chat_metrics = [all_metrics[m] for m in CHAT_ORDER if m in all_metrics]

    reasoning_agg = build_agg_row(reasoning_metrics, reasoning_common)
    chat_agg = build_agg_row(chat_metrics, chat_common)

    # Build tex
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \begin{tabular}{lccccccc}")
    lines.append(r"    \toprule")
    lines.append(
        r"    \textbf{Model} & \textbf{Type} & \textbf{ Score (\%)} & \textbf{ECE} "
        r"& \textbf{Conf. (\%)} & \textbf{\% Rnd} & \textbf{Regression Coef.} & \textbf{$N$} \\"
    )
    lines.append(r"    \midrule")

    # Reasoning models
    for m in REASONING_ORDER:
        if m in all_metrics:
            lines.append(build_tex_row(m, "Reasoning", all_metrics[m]))

    # Reasoning aggregate
    lines.append(r"    \midrule  %---------------------------------------------------------------------")
    a = reasoning_agg
    lines.append(
        f"    {{\\textbf{{Aggregate}}}}  &           "
        f"& {fmt(a['Score'])} & {fmt(a['ECE'], 3)} "
        f"& {fmt(a['Conf'])} & {fmt(a['PctRnd'])}  "
        f"& {fmt(a['RegCoef'], 3)} & {a['N']} \\\\"
    )
    lines.append(r"    \toprule    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Chat models
    for m in CHAT_ORDER:
        if m in all_metrics:
            lines.append(build_tex_row(m, "Chat", all_metrics[m]))

    # Chat aggregate
    lines.append(r"    \midrule  %---------------------------------------------------------------------")
    a = chat_agg
    lines.append(
        f"    {{\\textbf{{Aggregate}}}}  &           "
        f"& {fmt(a['Score'])} & {fmt(a['ECE'], 3)} "
        f"& {fmt(a['Conf'])} & {fmt(a['PctRnd'])}  "
        f"& {fmt(a['RegCoef'], 3)} & {a['N']} \\\\"
    )

    lines.append(r"")
    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(
        f"    \\caption{{Performance metrics on the LifeEval {tex_label} subset. "
        r"We report Mean Score, Expected Calibration Error (ECE), Mean Confidence, "
        r"Percentage of Rounded outputs, the Regression Coefficient between difficulty "
        r"and overconfidence, and number of completions ($N$). "
        r"Aggregate rows take mean of all columns except for $N$ which is the number of "
        r"questions where every model in the group received a matching verdict.}"
    )
    lines.append(r"    \label{}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines) + "\n"
    output_tex.write_text(tex, encoding="utf-8")
    print(f"\nWrote {output_tex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LifeEval subset LaTeX table")
    parser.add_argument(
        "--verdicts", nargs="+", default=["no_evidence"],
        choices=["no_evidence", "weak_evidence", "strong_evidence"],
        help="Verdict(s) to include (default: no_evidence)"
    )
    args = parser.parse_args()
    main(verdicts=args.verdicts)
