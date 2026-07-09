#!/usr/bin/env python3
"""
Build the LifeEval SPD benchmark CSV for RQ3.

Instead of point estimate + scalar confidence, the SPD prompt asks models to
report their top-N most likely age-at-death bins of a specified width.
Model-driven bin placement: the model chooses where to place bins (no fixed
grid). The only constraint is that each bin must be exactly `bin_width` years
wide. This avoids arbitrary boundary artifacts from grid alignment.

For each (sex in {male, female}, age in 0-100, bin_width in {2, 10, 20, 40}):
  - question_prompt: same as original LifeEval
  - confidence_prompt: specifies bin width, requests top-N JSON array
  - radius = bin_width / 2 (so existing score_lifeeval works on the modal bin)

808 questions total: 101 ages x 2 sexes x 4 bin widths.

Columns written:
    question_id, question_prompt, confidence_prompt, min_age, sex,
    radius, bin_width, top_n

Usage:
    python build_spd_benchmark.py
"""

import argparse
import csv
from pathlib import Path


BIN_WIDTHS = [2, 10, 20, 40]

TOP_N = {2: 10, 10: 10, 20: 5, 40: 3}

QUESTION_TEMPLATE_MALE = (
    "Given that an American male has lived at least {age} years, "
    "estimate how old he will be when he dies."
)

QUESTION_TEMPLATE_FEMALE = (
    "Given that an American female has lived at least {age} years "
    "estimate how old she will be when she dies."
)

CONFIDENCE_TEMPLATE = (
    "Using age ranges of exactly {bw} years (e.g. [70, {bw_example})), "
    "report your top {top_n} most likely ranges where this person will die. "
    "Each range must be exactly {bw} years wide. You choose where to place them.\n\n"
    "Respond with ONLY a JSON array in this exact format:\n"
    '[{{"min": <start_age>, "max": <end_age>, "confidence": <probability between 0 and 1>}}, ...]\n'
    "No other text."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark_spd.csv")
    args = ap.parse_args()

    qid = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_id", "question_prompt", "confidence_prompt",
            "min_age", "sex", "radius", "bin_width", "top_n",
        ])

        for sex in ["male", "female"]:
            template = QUESTION_TEMPLATE_MALE if sex == "male" else QUESTION_TEMPLATE_FEMALE
            for age in range(0, 101):
                q_prompt = template.format(age=age)
                for bw in BIN_WIDTHS:
                    top_n = TOP_N[bw]
                    c_prompt = CONFIDENCE_TEMPLATE.format(
                        bw=bw, bw_example=70 + bw, top_n=top_n,
                    )
                    w.writerow([
                        f"le_spd_{age}_{sex}_{bw}",
                        q_prompt, c_prompt,
                        age, sex, bw // 2, bw, top_n,
                    ])
                    qid += 1

    print(f"Wrote {qid} questions to {args.out}")


if __name__ == "__main__":
    main()
