# Difficulty Scoring in Study 2

How BayesEval assigns each question a difficulty score, in plain language. Spec and
derivations: `study-2-bayeseval/thoughts/measuring_difficulty.md` (note: `thoughts/`
is untracked); implementation: [`analysis/evaluate_diff.py`](../study-2-bayeseval/analysis/evaluate_diff.py).

## The idea: how well would a clueless guesser do?

A question is *easy* if even a guesser with no information scores well on it, and
*hard* if guessing is nearly worthless. Concretely, imagine answering by drawing
uniformly at random from the plausible answer range. The question's raw difficulty
signal is that guesser's **expected reward**:

```
E_U[R] = ∫ R(y) · U(y) dy
```

where `R(y)` is the probability that answer `y` counts as correct and `U` is the
uniform distribution over the answer support. High `E_U[R]` → easy; low → hard. This
is the Eckhardt–Lee difficulty function θ(x) specialized to a uniform answer
distribution — it measures the question, not any particular model.

## Per-domain closed forms

**WGD** — an answer within `r` lbs of the true weight scores 1. Sliding a width-2r
window across the weight support `[80, 350]`:

```
E_U[R] = 2r / 270
```

Difficulty depends only on the tolerance radius `r`. (Example: r = 10 → 20/270 ≈ 0.074.)

**MedEval** — one correct pathology among `n` remaining candidates:

```
E_U[R] = 1 / n
```

Difficulty depends only on how many candidates survive the removal step.

**LifeEval** — the reward for guessing age `y` is the Gompertz window probability
`P(death in [y−r, y+r) | survived to a)`. Integrating that over a uniform guess on
`[a, Y_max]` has the closed form

```
E_U[R] = Z(a, r) / (Y_max − a)
Z(a, r) = r + (e^A / β) · [E₁(A) − E₁(A · e^(βr))]
A       = (α / β) · e^(βa)
```

with `(α, β)` the sex-specific Gompertz parameters fitted to the SSA 2022 life table,
`E₁` the exponential integral (`scipy.special.exp1`), and `Y_max = 120` (the oldest
reachable grid age, 100 + 20). Difficulty rises for younger ages (wider answer range)
and narrower radii. `analysis/sensitivity_ymax.py` shows the resulting ranking is
insensitive to the `Y_max` choice (Spearman ρ > 0.99 across 118–130).

## From expected reward to the `diff` column

Raw `E_U[R]` values aren't comparable across domains (a 1/n is not a window
probability), so each domain is converted to a **within-domain rank percentile**:

```python
df["diff"] = rankdata(-raw, method="average") / len(raw)   # evaluate_diff.py
```

`diff` ∈ (0, 1], higher = harder, ties get midranks, and percentiles are pooled over
the whole domain (never stratified by radius or removal fraction). A LifeEval question
at `diff = 0.9` is in the hardest 10% of the 4,040-question LifeEval grid.

## How it's used

The analysis regresses per-question overconfidence (confidence − true probability) on
`diff` to test the **hard–easy effect**: overconfidence growing with difficulty. That
slope is positive for every model in study 2 — and for the human participants in the
[human supplement](../study-2-bayeseval/analysis/human_supplement.ipynb).
