"""
analyze_ssa_references.py
-------------------------
Analyzes LifeEval responses in combined_clean.csv to detect whether LLMs
reference SSA or actuarial knowledge (life tables, mortality tables, etc.).

Outputs (saved to "Combined Results/"):
  - life_eval_ssa_analysis.csv     : Full LifeEval table with indicator column
  - ssa_reference_report.txt       : Summary report
  - ssa_reference_examples.csv     : Rows where indicator is TRUE (for review)
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "Combined Results"
INPUT_FILE = DATA_DIR / "combined_clean.csv"

OUTPUT_FULL    = DATA_DIR / "life_eval_ssa_analysis.csv"
OUTPUT_REPORT  = DATA_DIR / "ssa_reference_report.txt"
OUTPUT_EXAMPLES = DATA_DIR / "ssa_reference_examples.csv"

# ---------------------------------------------------------------------------
# Keywords to detect (case-insensitive)
# ---------------------------------------------------------------------------
KEYWORDS = [
    r"SSA",
    r"Social Security",
    r"actuarial",
    r"life table",
    r"actuarial table",
    r"SSA table",
    r"mortality table",
]

# Combined pattern: match any keyword
PATTERN = "|".join(KEYWORDS)

# ---------------------------------------------------------------------------
# Auto-detect the response text column
# ---------------------------------------------------------------------------
CANDIDATE_COLUMNS = ["Reasoning", "Content", "Answer", "response", "completion", "text"]

def detect_response_column(df: pd.DataFrame) -> str:
    """Return the first candidate column that exists and has non-null strings."""
    for col in CANDIDATE_COLUMNS:
        if col in df.columns and df[col].dropna().astype(str).str.len().gt(0).any():
            return col
    raise ValueError(
        f"Could not find a response text column. Available columns: {list(df.columns)}"
    )

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    # 1. Load combined_clean
    print(f"Loading data from: {INPUT_FILE}")
    combined_clean = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"  Total rows in combined_clean: {len(combined_clean):,}")

    # 2. Filter for LifeEval
    life_eval_df = combined_clean[combined_clean["Question Set"] == "LifeEval"].copy()
    print(f"  LifeEval rows: {len(life_eval_df):,}")

    # 3. Auto-detect response column
    response_col = detect_response_column(life_eval_df)
    print(f"  Using response column: '{response_col}'")

    # 4. Keyword detection — boolean indicator per row
    life_eval_df["references_ssa_actuarial"] = (
        life_eval_df[response_col]
        .astype(str)
        .str.contains(PATTERN, case=False, na=False, regex=True)
    )

    # 5. Also flag which specific keyword(s) matched (for transparency)
    def matched_keywords(text: str) -> str:
        text = str(text)
        hits = [kw for kw in KEYWORDS if re.search(kw, text, flags=re.IGNORECASE)]
        return "; ".join(hits) if hits else ""

    life_eval_df["matched_keywords"] = life_eval_df[response_col].apply(matched_keywords)

    # 6. Create output dataframe (copy of life_eval with new columns)
    life_eval_ssa_analysis = life_eval_df.copy()

    # ---------------------------------------------------------------------------
    # 7. Build summary report
    # ---------------------------------------------------------------------------
    total = len(life_eval_ssa_analysis)
    n_ref = life_eval_ssa_analysis["references_ssa_actuarial"].sum()
    pct_ref = 100 * n_ref / total if total > 0 else 0.0

    # Per-model breakdown
    model_breakdown = (
        life_eval_ssa_analysis
        .groupby("Model")["references_ssa_actuarial"]
        .agg(total_responses="count", ssa_references="sum")
        .assign(pct=lambda x: 100 * x["ssa_references"] / x["total_responses"])
        .sort_values("pct", ascending=False)
    )

    # Per-keyword breakdown (count rows mentioning each keyword)
    keyword_counts = {}
    for kw in KEYWORDS:
        keyword_counts[kw] = (
            life_eval_ssa_analysis[response_col]
            .astype(str)
            .str.contains(kw, case=False, na=False, regex=True)
            .sum()
        )

    report_lines = [
        "=" * 60,
        "SSA / ACTUARIAL REFERENCE ANALYSIS — LifeEval",
        "=" * 60,
        "",
        f"Total LifeEval responses analyzed : {total:,}",
        f"Responses referencing SSA/actuarial: {n_ref:,}",
        f"Percentage                         : {pct_ref:.1f}%",
        "",
        "-" * 60,
        "PER-KEYWORD BREAKDOWN",
        "-" * 60,
    ]
    for kw, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {kw:<25} {count:>6,} rows")

    report_lines += [
        "",
        "-" * 60,
        "PER-MODEL BREAKDOWN",
        "-" * 60,
    ]
    for model, row in model_breakdown.iterrows():
        report_lines.append(
            f"  {model:<35} {int(row['ssa_references']):>5,} / {int(row['total_responses']):>5,}  ({row['pct']:.1f}%)"
        )

    report_lines += [
        "",
        "=" * 60,
        f"Response column used: '{response_col}'",
        f"Keywords searched   : {', '.join(KEYWORDS)}",
        "=" * 60,
    ]

    report_text = "\n".join(report_lines)

    # 8. Print report
    print("\n" + report_text + "\n")

    # 9. Save outputs
    life_eval_ssa_analysis.to_csv(OUTPUT_FULL, index=False)
    print(f"Saved full analysis table  -> {OUTPUT_FULL}")

    with open(OUTPUT_REPORT, "w") as f:
        f.write(report_text + "\n")
    print(f"Saved summary report       -> {OUTPUT_REPORT}")

    examples = life_eval_ssa_analysis[life_eval_ssa_analysis["references_ssa_actuarial"]]
    examples.to_csv(OUTPUT_EXAMPLES, index=False)
    print(f"Saved {len(examples):,} flagged examples -> {OUTPUT_EXAMPLES}")

    # Sanity check: original combined_clean must be unchanged
    assert len(combined_clean) == len(pd.read_csv(INPUT_FILE, low_memory=False)), \
        "ERROR: combined_clean.csv was modified — this should not happen!"
    print("\nSanity check passed: combined_clean.csv is unchanged.")


if __name__ == "__main__":
    main()
