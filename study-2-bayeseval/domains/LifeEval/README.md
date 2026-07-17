# LifeEval

Actuarial mortality calibration benchmark scored directly against the 2022 US Period Life Table (study 1's original empirical rule).

## Task

Each question presents a conditional survival scenario: given that a person of a specified sex has survived to age `a`, estimate how old they will be when they die and report a confidence that the estimate is within `r` years of the true value.

There is no single "true" age of death — mortality is a statistical distribution. The ground-truth probability is the life table's empirical probability mass falling in the window around the model's guess.

## Scoring

Unlike domains with deterministic ground truth (e.g., WGD where a guess is simply right or wrong), LifeEval scores against a probability distribution. The model's guess defines an integer-age window `[floor(y - r), ceil(y + r))` where `y` is the predicted age-at-death and `r` is the question's radius, clamped to `[a, 119)`. The true probability is:

```
true_probability = P(death in [floor(y - r), ceil(y + r)) | survived to age a)
```

This is read directly from the 2022 US Period Life Table's per-year death probabilities `q_x` (ages 0-118, separate male/female):

```
S_rel(x) = prod_{k=a}^{x-1} (1 - q_k)          # survival from a to x

true_probability = sum_{x in window} S_rel(x) * q_x
```

This is the same empirical rule study 1 preregistered — no parametric fit. (An earlier iteration of study 2 used a Gompertz hazard MLE-fitted to the same table; the two agree at r ≈ 0.99 but the project reverted to the empirical rule. The Gompertz implementation survives in git history.)

The Brier Score is then:

```
brier = (confidence - true_probability)^2
```

A well-calibrated model reporting 70% confidence should be picking windows holding approximately 0.70 of the empirical probability mass. The model minimizes expected Brier Score by reporting its true belief about the probability mass in the window.

## Benchmark

4040 questions: 101 ages (0-100) x 2 sexes x 20 radii (1-20 years).

### Build

```bash
python build_benchmark.py
```

Produces `Data/benchmark.csv` with columns:
`question_prompt, confidence_prompt, true_lifespan, question_id, min_age, sex, radius, best_answer, MAS, gold_response`.

- `best_answer` — the integer point estimate `y*` that maximizes the empirical window probability (smallest age wins ties, matching study 1's convention)
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
