#!/usr/bin/env python3
"""
Build the LifeEval benchmark CSV from the 2022 US Period Life Table.

For each (sex in {male, female}, age in 0-100, radius in {1, 5, 10, 20}):
  - Fits Gompertz hazard h(x) = b*exp(cx) via MLE (ages 5-94)
  - Computes best_answer (optimal point estimate) and MAS (maximum achievable
    score) using the closed-form conditional survival CDF
  - Generates question and confidence prompts

808 questions total: 101 ages x 2 sexes x 4 radii.

Columns written (matches BayesEval convention):
    question_id, question_prompt, confidence_prompt, true_lifespan,
    min_age, sex, radius, best_answer, MAS, gold_response

true_probability is computed at scoring time from the Gompertz conditional
survival CDF over [Answer - radius, Answer + radius].

Usage:
    python build_benchmark.py
    python build_benchmark.py --life-table Data/PeriodLifeTable_2022_RawData.csv
"""

import argparse
import csv
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from analysis.scoring import GompertzParams, fit_gompertz_to_life_table


def conditional_survival(x: float, a: float, params: GompertzParams) -> float:
    """S(x | X >= a) = exp(-(b/c)(exp(cx) - exp(ca)))"""
    b, c = params.b, params.c
    return float(np.exp(-(b / c) * (np.exp(c * x) - np.exp(c * a))))


def window_probability(y: float, a: float, r: float, params: GompertzParams) -> float:
    """P(death in [y-r, y+r] | survived to a)."""
    lo = max(y - r, a)
    hi = y + r
    if lo >= hi:
        return 0.0
    return conditional_survival(lo, a, params) - conditional_survival(hi, a, params)


def find_best_answer_and_mas(
    a: float, r: float, params: GompertzParams, max_age: float = 130.0
) -> tuple[float, float]:
    """Find point estimate y* maximizing window_probability; return (y*, MAS)."""
    result = minimize_scalar(
        lambda y: -window_probability(y, a, r, params),
        bounds=(a, max_age),
        method="bounded",
        options={"xatol": 1e-8},
    )
    return result.x, -result.fun


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
    params = {
        "male": fit_gompertz_to_life_table(str(args.life_table), "male"),
        "female": fit_gompertz_to_life_table(str(args.life_table), "female"),
    }

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
                    best_y, mas = find_best_answer_and_mas(age, r, params[sex])
                    gold_response = json.dumps({
                        "Answer": str(int(round(best_y))),
                        "Confidence": str(round(mas, 2)),
                    })
                    w.writerow([
                        q_prompt, c_prompt, true_lifespan,
                        qid, age, sex, r,
                        round(best_y, 2), round(mas, 6), gold_response,
                    ])
                    qid += 1

    print(f"Wrote {qid} questions to {args.out}")


if __name__ == "__main__":
    main()
