"""Sensitivity of LifeEval difficulty rankings to Y_max."""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, rankdata, linregress

from scoring import lifeeval_true_probability

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    "anthropic_claude-haiku-4.5",
    "google_gemini-2.5-flash",
    "meta-llama_llama-4-maverick",
    "openai_gpt-5.4-mini",
]
Y_MAX_REF = 120
Y_MAX_RANGE = range(101, 141)


def main():
    bench = pd.read_csv(ROOT / "domains" / "LifeEval" / "Data" / "benchmark.csv")

    # Load and score model results
    le_df = pd.DataFrame()
    for model in MODELS:
        df = pd.read_csv(ROOT / "results" / "LifeEval" / f"{model}.csv")
        rename = {}
        for col in ["min_age", "sex", "radius"]:
            if col not in df.columns and f"{col}_x" in df.columns:
                rename[f"{col}_x"] = col
        if rename:
            df = df.rename(columns=rename)
        df["model"] = model
        le_df = pd.concat([le_df, df], ignore_index=True)

    le_df["Answer_num"] = pd.to_numeric(le_df["Answer"], errors="coerce")
    le_df = le_df.dropna(subset=["Answer_num", "sex", "min_age", "radius"])
    le_df["true_probability"] = [
        lifeeval_true_probability(a, age, sex, r)
        for a, age, sex, r in zip(le_df["Answer_num"], le_df["min_age"], le_df["sex"], le_df["radius"])
    ]
    le_df["overconfidence"] = le_df["Confidence"] - le_df["true_probability"]

    # Precompute cumulative window-probability sums (Y_max-independent) on
    # benchmark questions: cumw[k] = sum of empirical window probabilities
    # for integer guesses y in [a, a+k].
    y_max_hi = max(Y_MAX_RANGE)
    min_ages_bench = bench["min_age"].to_numpy(dtype=int)
    cumw_bench = []
    for _, row in bench.iterrows():
        a, r = int(row["min_age"]), float(row["radius"])
        sex = row["sex"].lower()
        w = [lifeeval_true_probability(float(y), a, sex, r) for y in range(a, y_max_hi + 1)]
        cumw_bench.append(np.cumsum(w))

    def eu_at(y_max: int) -> np.ndarray:
        """EU = mean window probability over integer guesses in [a, y_max]."""
        return np.array([
            cw[y_max - a] / (y_max - a + 1) if y_max >= a else 0.0
            for cw, a in zip(cumw_bench, min_ages_bench)
        ])

    # Map result rows to benchmark rows by question_id
    idx_by_qid = {qid: i for i, qid in enumerate(bench["question_id"])}
    le_df["bench_idx"] = le_df["question_id"].map(idx_by_qid)
    le_df = le_df.dropna(subset=["bench_idx"])
    bench_idx = le_df["bench_idx"].to_numpy(dtype=int)

    # Reference difficulty ranks (Y_max=120)
    ref_ranks_bench = rankdata(-eu_at(Y_MAX_REF), method="average")

    rows = []
    for y_max in Y_MAX_RANGE:
        eu_bench = eu_at(y_max)

        # Spearman rho vs reference
        ranks_bench = rankdata(-eu_bench, method="average")
        rho, _ = spearmanr(ref_ranks_bench, ranks_bench)

        # Difficulty percentile on results for this Y_max
        eu = eu_bench[bench_idx]
        le_df["diff"] = rankdata(-eu, method="average") / len(eu)

        # Beta_1 per model
        betas = []
        for model in MODELS:
            sub = le_df[le_df["model"] == model].dropna(subset=["diff", "overconfidence"])
            slope, _, _, _, _ = linregress(sub["diff"], sub["overconfidence"])
            betas.append(slope)
        mean_beta = np.mean(betas)

        rows.append((y_max, rho, mean_beta))

    # Write LaTeX table
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of LifeEval difficulty metric to $Y_{\max}$. "
        r"$\rho$ is the Spearman rank correlation of difficulty percentiles against the reference ($Y_{\max}=120$). "
        r"Mean $\beta_1$ is the average OLS slope of overconfidence on difficulty percentile across four models.}",
        r"\label{tab:ymax_sensitivity}",
        r"\begin{tabular}{rcc}",
        r"\toprule",
        r"$Y_{\max}$ & $\rho$ & Mean $\beta_1$ \\",
        r"\midrule",
    ]
    for y_max, rho, mean_beta in rows:
        ref = r" $\leftarrow$" if y_max == Y_MAX_REF else ""
        lines.append(
            f"{y_max} & {rho:.4f} & {mean_beta:.4f}{ref} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = Path(__file__).resolve().parent / "sensitivity_ymax_table.txt"
    out.write_text("\n".join(lines))
    print(f"LaTeX table written to {out}")

    # --- Dual-axis plot ---
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

    y_maxes = [r[0] for r in rows]
    rhos = [r[1] for r in rows]
    betas = [r[2] for r in rows]

    color_rho = "#0072B2"
    color_beta = "#D55E00"

    fig, ax1 = plt.subplots(figsize=(5.0, 3.0), layout="constrained")

    ax1.plot(y_maxes, rhos, color=color_rho, linewidth=1.2, marker="o", markersize=3, label=r"Spearman $\rho$")
    ax1.set_xlabel(r"$Y_{\max}$")
    ax1.set_ylabel(r"Spearman $\rho$", color=color_rho)
    ax1.tick_params(axis="y", labelcolor=color_rho)
    ax1.axvline(Y_MAX_REF, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax1.grid(alpha=0.3)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(y_maxes, betas, color=color_beta, linewidth=1.2, marker="s", markersize=3, label=r"Mean $\beta_1$")
    ax2.set_ylabel(r"Mean $\beta_1$", color=color_beta)
    ax2.tick_params(axis="y", labelcolor=color_beta)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    fig_path = Path(__file__).resolve().parent / "figs" / "sensitivity_ymax.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path)
    print(f"Plot saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
