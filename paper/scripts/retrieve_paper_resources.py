#!/usr/bin/env python3
"""Collect all paper-ready resources into paper/.

Copies (never moves) the aggregate/summary figures and LaTeX tables from both
studies into a single folder for the paper:

    paper/tex/            LaTeX tables (.tex)
    paper/plots/study1/   study-1 headline figures
    paper/plots/study2/   study-2 headline figures (+ human_supplement/)

Three sources feed paper/tex/:
  1. study-2 tables already saved as LaTeX .txt files      -> copied, renamed .tex
  2. study-1 LaTeX summary tables printed inside
     analysis.ipynb (reasoning + chat models)              -> extracted from outputs
  3. study-1 Table 1 (R pipeline CSV, publication layout)  -> converted via pandas

Idempotent: re-running overwrites paper/ contents and touches nothing else.
Run from anywhere: python3 paper/scripts/retrieve_paper_resources.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"

S1 = Path("study-1-benchmark-calibration")
S2 = Path("study-2-bayeseval")

# ---------------------------------------------------------------------------
# 1. Straight copies: (source relative to repo root, destination relative to paper/)
# ---------------------------------------------------------------------------
MANIFEST: list[tuple[Path, Path]] = [
    # -- study 1: aggregate calibration curves ------------------------------
    (S1 / "Plots/Main Plots/aggregate_extended_all_qsets_cal_plot.png", Path("plots/study1/aggregate_extended_all_qsets_cal_plot.png")),
    (S1 / "Plots/Main Plots/aggregate_extended_mcq_cal_plot.png",       Path("plots/study1/aggregate_extended_mcq_cal_plot.png")),
    (S1 / "Plots/Main Plots/aggregate_extended_2afc_cal_plot.png",      Path("plots/study1/aggregate_extended_2afc_cal_plot.png")),
    (S1 / "Plots/Main Plots/reasoning_vs_chat_by_qset_cal_plot.png",    Path("plots/study1/reasoning_vs_chat_by_qset_cal_plot.png")),
    (S1 / "Plots/Main Plots/calibration_grid_all_models_all_qsets.png", Path("plots/study1/calibration_grid_all_models_all_qsets.png")),
    # -- study 1: cross-benchmark metric summaries --------------------------
    (S1 / "Plots/Summary Plots/ece_all.png",             Path("plots/study1/ece_all.png")),
    (S1 / "Plots/Summary Plots/oc_all.png",              Path("plots/study1/oc_all.png")),
    (S1 / "Plots/Summary Plots/acc_all.png",             Path("plots/study1/acc_all.png")),
    (S1 / "Plots/Summary Plots/stated_vs_token_ece.png", Path("plots/study1/stated_vs_token_ece.png")),
    # -- study 2: RQ1-RQ3 headline panels ------------------------------------
    (S2 / "analysis/figs/calibration_combined.png",               Path("plots/study2/calibration_combined.png")),
    (S2 / "analysis/figs/calibration_spd_combined.png",           Path("plots/study2/calibration_spd_combined.png")),
    (S2 / "analysis/figs/overconfidence_by_difficulty.png",       Path("plots/study2/overconfidence_by_difficulty.png")),
    (S2 / "analysis/figs/rq2_overconfidence_by_percentile.png",   Path("plots/study2/rq2_overconfidence_by_percentile.png")),
    (S2 / "analysis/figs/rq3_overconfidence_by_percentile_spd.png", Path("plots/study2/rq3_overconfidence_by_percentile_spd.png")),
    (S2 / "analysis/figs/rq3_ece_spd_improvement.png",            Path("plots/study2/rq3_ece_spd_improvement.png")),
    (S2 / "analysis/figs/rq3_ece_pct_change.png",                 Path("plots/study2/rq3_ece_pct_change.png")),
    # -- study 2: robustness + post-hoc --------------------------------------
    (S2 / "analysis/figs/sensitivity_ymax.png",    Path("plots/study2/sensitivity_ymax.png")),
    (S2 / "analysis/figs/posthoc_sex_bias_wgd.png", Path("plots/study2/posthoc_sex_bias_wgd.png")),
    # -- study 2: dataset descriptives ---------------------------------------
    (S2 / "analysis/figs/wgd_demographics.png",           Path("plots/study2/wgd_demographics.png")),
    (S2 / "analysis/figs/wgd_weight_age.png",             Path("plots/study2/wgd_weight_age.png")),
    (S2 / "analysis/figs/poster/medeval_pathologies.png", Path("plots/study2/medeval_pathologies.png")),
    # -- study 2: human supplement (MTurk LifeEval) ---------------------------
    (S2 / "analysis/figs/human_supplement/calibration_human_vs_llm.png", Path("plots/study2/human_supplement/calibration_human_vs_llm.png")),
    (S2 / "analysis/figs/human_supplement/hard_easy_effect.png",         Path("plots/study2/human_supplement/hard_easy_effect.png")),
    (S2 / "analysis/figs/human_supplement/overconfidence_by_radius.png", Path("plots/study2/human_supplement/overconfidence_by_radius.png")),
    (S2 / "analysis/figs/human_supplement/overconfidence_by_age.png",    Path("plots/study2/human_supplement/overconfidence_by_age.png")),
    # -- study 2: tables already saved as LaTeX ------------------------------
    (S2 / "analysis/rq1_summary_table.txt",      Path("tex/study2_rq1_summary.tex")),
    (S2 / "analysis/sensitivity_ymax_table.txt", Path("tex/study2_sensitivity_ymax.tex")),
]

# ---------------------------------------------------------------------------
# 2. Study-1 LaTeX summary tables embedded in analysis.ipynb cell outputs.
#
# The notebook prints several \begin{table*} blocks; the two we want are
# identified by a model name unique to each group. For the reasoning table the
# notebook holds two variants and the *captioned* one (cell "Format decimals
# and convert to LaTeX") has corrupted LSAT/SAT/SciQ rows, so when several
# candidates match a marker we keep the one with the most table rows.
# ---------------------------------------------------------------------------
S1_NOTEBOOK = S1 / "analysis.ipynb"
NOTEBOOK_TABLES = {
    "tex/study1_summary_reasoning.tex": {
        "marker": "GPT-o3",  # reasoning-model column header
        "caption": "Summary metrics for reasoning models (study 1).",
        "label": "tab:study1_summary_reasoning",
    },
    "tex/study1_summary_chat.tex": {
        "marker": "GPT-4o",  # chat-model column header
        "caption": "Summary metrics for chat (non-reasoning) models (study 1).",
        "label": "tab:study1_summary_chat",
    },
}

TABLE_RE = re.compile(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", re.DOTALL)

# LifeEval was dropped from study 1 (see docs/lifeeval-history.md) but the notebook
# outputs and the R Table 1 CSV predate the drop; remove its rows at retrieval time.
LIFEEVAL_BLOCK_RE = re.compile(
    r"\\multirow\[[^\]]*\]\{\d+\}\{\*\}\{LifeEval\}.*?(?=\\multirow|\\bottomrule)",
    re.DOTALL,
)


def drop_lifeeval_rows(block: str) -> str:
    cleaned, n = LIFEEVAL_BLOCK_RE.subn("", block)
    if n != 1:
        raise SystemExit(f"ERROR: expected 1 LifeEval row group in extracted table, found {n}")
    return cleaned


def notebook_table_blocks(nb_path: Path) -> list[str]:
    """All \\begin{table(*)}...\\end{table(*)} blocks printed by code cells."""
    nb = json.loads(nb_path.read_text())
    blocks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                text = "".join(out["text"])
            elif "data" in out and "text/plain" in out.get("data", {}):
                text = "".join(out["data"]["text/plain"])
            else:
                continue
            blocks.extend(TABLE_RE.findall(text))
    return blocks


def ensure_caption_and_label(block: str, caption: str, label: str) -> str:
    """Normalize the extracted table: our caption/label right after \\begin{table*}."""
    block = re.sub(r"^\\caption\{[^}]*\}\n?", "", block, flags=re.MULTILINE)
    block = re.sub(r"^\\label\{[^}]*\}\n?", "", block, flags=re.MULTILINE)
    return re.sub(
        r"(\\begin\{table\*?\})",
        rf"\1\n\\caption{{{caption}}}\n\\label{{{label}}}",
        block,
        count=1,
    )


def extract_notebook_tables() -> list[str]:
    written = []
    blocks = notebook_table_blocks(REPO / S1_NOTEBOOK)
    for dest_rel, spec in NOTEBOOK_TABLES.items():
        candidates = [b for b in blocks if spec["marker"] in b]
        if not candidates:
            raise SystemExit(
                f"ERROR: no table with marker {spec['marker']!r} found in {S1_NOTEBOOK}"
            )
        block = max(candidates, key=lambda b: b.count(r"\\"))  # most rows wins
        block = drop_lifeeval_rows(block)
        block = ensure_caption_and_label(block, spec["caption"], spec["label"])
        dest = PAPER / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(block + "\n")
        written.append(dest_rel)
    return written


# ---------------------------------------------------------------------------
# 3. Study-1 Table 1: R pipeline CSV (publication layout) -> LaTeX.
# ---------------------------------------------------------------------------
TABLE1_CSV = S1 / "R/analysis/table1-similar-layout.csv"
TABLE1_DEST = "tex/study1_table1.tex"
METRIC_NAMES = {
    "num_tot": "N",
    "ece": "ECE",
    "accuracy": "Accuracy",
    "overconfidence": "Overconfidence",
}


def convert_table1() -> str:
    df = pd.read_csv(REPO / TABLE1_CSV)
    df = df[df["qset"] != "LifeEval"]
    df["metric"] = df["metric"].map(lambda m: METRIC_NAMES.get(m, m))
    df = df.set_index(["qset", "metric"])
    df.index.names = ["Question Set", "Metric"]

    def fmt(row):
        if row.name[1] == "N":
            return row.map(lambda v: f"{int(v)}")
        return row.map(lambda v: f"{v:.3f}")

    df = df.apply(fmt, axis=1)
    latex = df.to_latex(
        multirow=True,
        caption=(
            "Study 1 calibration metrics by question set and model "
            "(from the preregistered R analysis pipeline)."
        ),
        label="tab:study1_table1",
        column_format="ll" + "r" * len(df.columns),
    )
    dest = PAPER / TABLE1_DEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(latex)
    return TABLE1_DEST


# ---------------------------------------------------------------------------


def main() -> int:
    missing = [str(src) for src, _ in MANIFEST if not (REPO / src).is_file()]
    for extra in (S1_NOTEBOOK, TABLE1_CSV):
        if not (REPO / extra).is_file():
            missing.append(str(extra))
    if missing:
        print("ERROR: missing source files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    for src, dest_rel in MANIFEST:
        dest = PAPER / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / src, dest)
        print(f"copied     {src}  ->  paper/{dest_rel}")

    for dest_rel in extract_notebook_tables():
        print(f"extracted  {S1_NOTEBOOK}  ->  paper/{dest_rel}")

    print(f"converted  {TABLE1_CSV}  ->  paper/{convert_table1()}")

    n_files = len(MANIFEST) + len(NOTEBOOK_TABLES) + 1
    print(f"\nDone: {n_files} files in {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
