#!/usr/bin/env python3
"""Summarize the reasoning-capture pilot: output lengths and projected cost.

Usage (after `python eval.py --config config_pilot_reasoning.yaml`):

    python -m src.features.pilot_summary
    python -m src.features.pilot_summary --results results_pilot_reasoning
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# OpenRouter pricing per million tokens (input, output), checked 2026-07-10.
LIVE_PRICING: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash": (0.30, 2.50),
    "openai/gpt-5.4-mini": (0.75, 4.50),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "meta-llama/llama-4-maverick": (0.15, 0.60),
}

# Full re-run scope: all 6 benchmark sets (LifeEval/WGD DCE+SPD, MedEval
# combined DCE+SPD) per model, input includes the Reasoning-field instruction.
FULL_RUN_QUESTIONS = 11_932
FULL_RUN_INPUT_TOK = 2_763_100


def slugify(model: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path,
                    default=REPO_ROOT / "results_pilot_reasoning")
    args = ap.parse_args()

    frames = []
    for csv in sorted(args.results.glob("*/*.csv")):
        df = pd.read_csv(csv)
        if "tok_out" not in df.columns:
            print(f"skipping {csv}: no tok_out column")
            continue
        df["domain"] = csv.parent.name
        df["model_slug"] = csv.stem
        frames.append(df)
    if not frames:
        raise SystemExit(f"No result CSVs with token counts under {args.results}")
    allr = pd.concat(frames, ignore_index=True)

    ok = allr[allr["error"].isna()]
    stats = (
        ok.groupby(["domain", "model_slug"])["tok_out"]
        .agg(n="count", mean="mean", median="median",
             p90=lambda s: s.quantile(0.9), max="max")
        .round(0).astype(int).reset_index()
    )
    print("\nOutput tokens per response (successful calls only):\n")
    print(stats.to_string(index=False))

    n_err = int(allr["error"].notna().sum())
    if n_err:
        print(f"\n{n_err} errored calls excluded "
              f"({n_err / len(allr):.1%} of {len(allr)}).")

    print("\nProjected full re-run (all 6 benchmark sets, "
          f"{FULL_RUN_QUESTIONS:,} questions/model), live pricing 2026-07-10:\n")
    rows = []
    for model, (p_in, p_out) in LIVE_PRICING.items():
        mine = ok[ok["model_slug"] == slugify(model)]
        if mine.empty:
            continue
        mean_out = mine["tok_out"].mean()
        cost = (FULL_RUN_INPUT_TOK * p_in
                + FULL_RUN_QUESTIONS * mean_out * p_out) / 1e6
        rows.append({"model": model, "mean_tok_out": round(mean_out),
                     "projected_cost": f"${cost:,.2f}"})
    proj = pd.DataFrame(rows)
    print(proj.to_string(index=False))
    total = sum(float(r["projected_cost"].lstrip("$").replace(",", ""))
                for r in rows)
    print(f"\nProjected total: ${total:,.2f}")
    print("Caveat: pilot covers DCE only (LifeEval + MedEval); SPD responses "
          "add a JSON array (~100-200 tok) on top of any reasoning.")


if __name__ == "__main__":
    main()
