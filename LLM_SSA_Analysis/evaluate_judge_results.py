"""
evaluate_judge_results.py
-------------------------
CLI script to categorize and aggregate judge results from any
judge_results CSV. No API calls — pure post-processing analysis.

Usage:
    python evaluate_judge_results.py demo_judge_results.csv
    python evaluate_judge_results.py judge_results.csv
    python evaluate_judge_results.py /path/to/any_judge_results.csv
"""

import sys
import json
import pandas as pd
from pathlib import Path
from collections import Counter


def load_results(csv_path: str) -> pd.DataFrame:
    """Load judge results CSV."""
    path = Path(csv_path)
    if not path.exists():
        # Try relative to script directory
        path = Path(__file__).resolve().parent / csv_path
    if not path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    return pd.read_csv(path, low_memory=False), path


def print_header(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_section(title: str):
    print(f"\n{'-'*65}")
    print(f"  {title}")
    print(f"{'-'*65}")


def analyze_overall_distribution(df: pd.DataFrame):
    """1. Overall verdict distribution."""
    print_section("OVERALL VERDICT DISTRIBUTION")
    total = len(df)
    verdict_counts = df["verdict"].value_counts()

    for verdict in ["no_evidence", "weak_evidence", "strong_evidence"]:
        count = verdict_counts.get(verdict, 0)
        pct = 100 * count / total if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {verdict:<20} {count:>6,} ({pct:5.1f}%)  {bar}")

    # Show any other verdicts (parse_error, api_error, etc.)
    other_verdicts = [v for v in verdict_counts.index
                      if v not in ("no_evidence", "weak_evidence", "strong_evidence")]
    if other_verdicts:
        print()
        for verdict in other_verdicts:
            count = verdict_counts[verdict]
            pct = 100 * count / total
            print(f"  {verdict:<20} {count:>6,} ({pct:5.1f}%)  [non-standard]")


def analyze_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """2. Per-model breakdown, sorted by strong_evidence rate."""
    print_section("PER-MODEL BREAKDOWN (sorted by strong_evidence rate)")

    models = df["Model"].unique()
    rows = []
    for model in models:
        model_df = df[df["Model"] == model]
        total = len(model_df)
        no_ev = (model_df["verdict"] == "no_evidence").sum()
        weak = (model_df["verdict"] == "weak_evidence").sum()
        strong = (model_df["verdict"] == "strong_evidence").sum()
        strong_pct = 100 * strong / total if total > 0 else 0
        rows.append({
            "Model": model,
            "Total": total,
            "no_evidence": no_ev,
            "weak_evidence": weak,
            "strong_evidence": strong,
            "strong_pct": strong_pct,
        })

    summary = pd.DataFrame(rows).sort_values("strong_pct", ascending=False)

    for _, row in summary.iterrows():
        print(f"  {row['Model']:<35} "
              f"none={int(row['no_evidence']):>4}  "
              f"weak={int(row['weak_evidence']):>4}  "
              f"strong={int(row['strong_evidence']):>4}  "
              f"({row['strong_pct']:5.1f}% strong)")

    return summary


def analyze_keyword_vs_judge(df: pd.DataFrame):
    """3. Compare keyword-flagged rows vs judge verdicts."""
    if "references_ssa_actuarial" not in df.columns:
        print_section("KEYWORD vs JUDGE COMPARISON")
        print("  Skipped: 'references_ssa_actuarial' column not found")
        return

    print_section("KEYWORD vs JUDGE COMPARISON")

    # Keyword-flagged rows
    kw_flagged = df[df["references_ssa_actuarial"] == True]
    kw_unflagged = df[df["references_ssa_actuarial"] == False]

    print(f"\n  Keyword-flagged rows: {len(kw_flagged):,}")
    print(f"  Keyword-unflagged rows: {len(kw_unflagged):,}")

    if len(kw_flagged) > 0:
        # How many keyword-flagged rows did the judge downgrade?
        downgraded = kw_flagged[kw_flagged["verdict"] == "no_evidence"]
        kept_weak = kw_flagged[kw_flagged["verdict"] == "weak_evidence"]
        kept_strong = kw_flagged[kw_flagged["verdict"] == "strong_evidence"]

        print(f"\n  Of {len(kw_flagged):,} keyword-flagged rows:")
        print(f"    Judge says no_evidence:     {len(downgraded):>6,} "
              f"({100*len(downgraded)/len(kw_flagged):5.1f}%)  <- false positives from keyword matching")
        print(f"    Judge says weak_evidence:   {len(kept_weak):>6,} "
              f"({100*len(kept_weak)/len(kw_flagged):5.1f}%)")
        print(f"    Judge says strong_evidence: {len(kept_strong):>6,} "
              f"({100*len(kept_strong)/len(kw_flagged):5.1f}%)")


def analyze_judge_only_flags(df: pd.DataFrame):
    """4. Rows the judge flagged that were NOT keyword-flagged."""
    if "references_ssa_actuarial" not in df.columns:
        return

    print_section("JUDGE-ONLY FLAGS (not caught by keyword matching)")

    kw_unflagged = df[df["references_ssa_actuarial"] == False]
    if len(kw_unflagged) == 0:
        print("  No unflagged rows to check")
        return

    judge_flagged_only = kw_unflagged[
        kw_unflagged["verdict"].isin(["weak_evidence", "strong_evidence"])
    ]

    print(f"  Unflagged rows where judge found evidence: {len(judge_flagged_only):,} / {len(kw_unflagged):,}")

    if len(judge_flagged_only) > 0:
        print(f"\n  Breakdown:")
        for verdict in ["weak_evidence", "strong_evidence"]:
            count = (judge_flagged_only["verdict"] == verdict).sum()
            if count > 0:
                print(f"    {verdict}: {count}")

        # Show a few examples
        print(f"\n  Sample judge-only flags (up to 5):")
        for _, row in judge_flagged_only.head(5).iterrows():
            print(f"    Model={row['Model']}, QID={row['Question ID']}, "
                  f"verdict={row['verdict']}")
            explanation = str(row.get('explanation', ''))[:120]
            print(f"      {explanation}")


def analyze_top_claims(df: pd.DataFrame):
    """5. Most common specific_claims across strong_evidence rows."""
    print_section("TOP SPECIFIC CLAIMS (from strong_evidence rows)")

    strong = df[df["verdict"] == "strong_evidence"]
    if len(strong) == 0:
        print("  No strong_evidence rows found")
        return

    all_claims = []
    for claims_str in strong["specific_claims"].dropna():
        try:
            claims = json.loads(claims_str)
            if isinstance(claims, list):
                all_claims.extend(claims)
        except (json.JSONDecodeError, TypeError):
            pass

    if not all_claims:
        print("  No parseable specific_claims found")
        return

    claim_counts = Counter(all_claims)
    print(f"  Total claims extracted: {len(all_claims):,}")
    print(f"  Unique claims: {len(claim_counts):,}")
    print(f"\n  Top 15 most common claims:")
    for claim, count in claim_counts.most_common(15):
        print(f"    [{count:>4}x]  {claim[:100]}")


def export_summary(df: pd.DataFrame, summary: pd.DataFrame, input_path: Path):
    """6. Export per-model summary CSV."""
    output_path = input_path.parent / f"{input_path.stem}_summary.csv"
    summary.to_csv(output_path, index=False)
    print_section("EXPORTED")
    print(f"  Summary CSV: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_judge_results.py <judge_results.csv>")
        print("\nExamples:")
        print("  python evaluate_judge_results.py demo_judge_results.csv")
        print("  python evaluate_judge_results.py judge_results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df, resolved_path = load_results(csv_path)

    print_header(f"JUDGE RESULTS ANALYSIS: {resolved_path.name}")
    print(f"  Rows: {len(df):,}")
    print(f"  Models: {df['Model'].nunique()}")
    print(f"  Columns: {list(df.columns)}")

    # Run all analyses
    analyze_overall_distribution(df)
    summary = analyze_per_model(df)
    analyze_keyword_vs_judge(df)
    analyze_judge_only_flags(df)
    analyze_top_claims(df)
    export_summary(df, summary, resolved_path)

    print(f"\n{'='*65}")
    print(f"  Analysis complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
