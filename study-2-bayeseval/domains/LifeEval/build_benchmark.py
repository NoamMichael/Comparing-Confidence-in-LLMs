#!/usr/bin/env python3
"""
Build the LifeEval benchmark CSV from the 2022 US Period Life Table.

For each (sex in {male, female}, age in 0-100, radius in 1-20):
  - Computes best_answer (optimal integer point estimate) and MAS (maximum
    achievable score) directly from the life table's empirical death
    probabilities (study-1 rule; smallest age wins ties)
  - Generates question and confidence prompts

4040 questions total: 101 ages x 2 sexes x 20 radii.

Columns written (matches BayesEval convention):
    question_id, question_prompt, confidence_prompt, true_lifespan,
    min_age, sex, radius, best_answer, MAS, gold_response

true_probability is computed at scoring time as the empirical probability of
death in the integer-age window [floor(Answer - radius), ceil(Answer + radius))
conditional on survival to min_age, clamped to the table bounds (ages 0-118).

Usage:
    python build_benchmark.py
    python build_benchmark.py --life-table Data/PeriodLifeTable_2022_RawData.csv
"""

import argparse
import csv
import json
from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.scoring import lifeeval_best_answer_and_mas


QUESTION_TEMPLATE_MALE = (
    "Given that an American male has lived at least {age} years, "
    "estimate how old he will be when he dies."
)

QUESTION_TEMPLATE_FEMALE = (
    "Given that an American female has lived at least {age} years "
    "estimate how old she will be when she dies."
)

CONFIDENCE_INSTRUCTION = (
    "How certain are you that your answer is within "
    "{radius} {year_word} of the true value?"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--life-table", type=Path,
                    default=Path(__file__).parent / "Data" / "PeriodLifeTable_2022_RawData.csv")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "Data" / "benchmark.csv")
    args = ap.parse_args()

    lt = pd.read_csv(args.life_table)

    radii = list(range(1, 21))
    qid = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "question_prompt", "confidence_prompt", "true_lifespan",
            "question_id", "min_age", "sex", "radius",
            "best_answer", "MAS", "gold_response",
        ])

        for sex in ["male", "female"]:
            col_prefix = "MALE" if sex == "male" else "FEMALE"
            life_exp_col = f"Life expectancy ({col_prefix})"
            template = QUESTION_TEMPLATE_MALE if sex == "male" else QUESTION_TEMPLATE_FEMALE

            for _, lt_row in lt.iterrows():
                age = int(lt_row["Age"])
                if age > 100:
                    break
                life_exp = float(lt_row[life_exp_col])
                true_lifespan = round(age + life_exp, 2)
                q_prompt = template.format(age=age)

                for r in radii:
                    year_word = "year" if r == 1 else "years"
                    c_prompt = CONFIDENCE_INSTRUCTION.format(radius=r, year_word=year_word)
                    best_y, mas = lifeeval_best_answer_and_mas(age, sex, r)
                    gold_response = json.dumps({
                        "Answer": str(int(best_y)),
                        "Confidence": str(round(mas, 2)),
                    })
                    w.writerow([
                        q_prompt, c_prompt, true_lifespan,
                        qid, age, sex, r,
                        best_y, round(mas, 6), gold_response,
                    ])
                    qid += 1

    print(f"Wrote {qid} questions to {args.out}")


if __name__ == "__main__":
    main()
