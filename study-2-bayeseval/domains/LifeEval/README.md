# LifeEval

Actuarial mortality calibration benchmark using the 2022 US Period Life Table and the Gompertz mortality model.

## Task

Each question presents a conditional survival scenario: given that a person of a specified sex has survived to age `a`, estimate how old they will be when they die and report a confidence that the estimate is within `r` years of the true value.

There is no single "true" age of death — mortality is a statistical distribution. The ground-truth probability is computed by integrating the Gompertz conditional PDF over the window around the model's guess.

## Scoring

Unlike domains with deterministic ground truth (e.g., WGD where a guess is simply right or wrong), LifeEval scores against a continuous probability distribution. The model's guess defines a window `[y - r, y + r)` where `y` is the predicted age-at-death and `r` is the question's radius. The true probability is:

```
true_probability = P(death in [y - r, y + r) | survived to age a)
```

This is computed in closed form via the Gompertz conditional survival CDF:

```
S(x | X >= a) = exp(-(b/c)(exp(cx) - exp(ca)))

true_probability = S(y - r | a) - S(y + r | a)
```

where `b` and `c` are Gompertz hazard parameters (`h(x) = b * exp(c * x)`) fit via MLE to the 2022 US Period Life Table (ages 5-94, separate male/female).

The Brier Score is then:

```
brier = (confidence - true_probability)^2
```

A well-calibrated model reporting 70% confidence should be picking windows where the Gompertz integral is approximately 0.70. The model minimizes expected Brier Score by reporting its true belief about the probability mass in the window.

## Benchmark

4040 questions: 101 ages (0-100) x 2 sexes x 20 radii (1-20 years).

### Build

```bash
python build_benchmark.py
```

Produces `Data/benchmark.csv` with columns:
`question_prompt, confidence_prompt, true_lifespan, question_id, min_age, sex, radius, best_answer, MAS, gold_response`.

- `best_answer` — the point estimate `y*` that maximizes `P(death in [y-r, y+r) | a)` (the mode of the window probability)
- `MAS` — Maximum Achievable Score: the best possible Brier Score for this question, achieved by a perfectly calibrated model guessing `y*`
- `true_lifespan` — life expectancy (`a + e_a`) from the life table, included for reference but **not used in scoring**

### Score

Given a results CSV with `question_id, Answer, Confidence`:

```python
from analysis.scoring import score_lifeeval, murphy_decomposition
scored = score_lifeeval(results_df, benchmark_df)
print(f"Brier Score: {scored['brier'].mean():.4f}")
print(murphy_decomposition(scored))
```

The Murphy decomposition breaks Brier Score into:
- **Reliability** — how close confidence is to true window probability (lower = better calibrated)
- **Resolution** — ability to discriminate between easy and hard windows (higher = better)
- **Uncertainty** — irreducible difficulty of the question set

## Expected model response format

```json
{"Answer": "78", "Confidence": "0.45"}
```
