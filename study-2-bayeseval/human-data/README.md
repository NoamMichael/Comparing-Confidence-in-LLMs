# LifeEval Human Study (LEH)

Human benchmark for the LifeEval life-expectancy task: a preregistered MTurk/Qualtrics
study asking people the same questions the LLMs answer in this study's LifeEval domain.

- **Pre-registration:** AsPredicted #267677 (2026-01-13) — `prereg/LEH-study-AsPredicted-267677.pdf`
- **Research question:** are people overconfident in their estimates of life expectancies?
- **Analysis notebook:** [`../analysis/human_supplement.ipynb`](../analysis/human_supplement.ipynb)

## Design

Each item: *"Given that an American {male|female} has lived at least {age} years, estimate
how old {he|she} will be (in years) when {he|she} dies."* followed by *"How certain are you
that your answer is within {radius} years of the true value?"* (0–100 slider).

- **88 item conditions** = 11 ages {1, 10, 20, …, 100} × 2 sexes × 4 radii {1, 5, 10, 20}
- Each participant answered **10 randomly assigned items**
- Two DVs per item: point estimate of age at death (`PGuess`) and confidence (`PConf`, 0–100)
- The 88 conditions are a subset of study 1's 808 LifeEval questions (ages 0–100); `LifeEvalQ`
  carries the study-1 question ID (sex = QID < 404, age = floor(QID/4), radius = QID mod 4 →
  {1,5,10,20}). All 88 cells also exist inside this study's 4,040-question DCE grid.

## Files

| Path | Contents |
|---|---|
| `raw/LEH_January+13,+2026_14.26.csv` | Raw Qualtrics numeric export (109 response rows; MTurk worker IDs were not exported, so the file is anonymous) |
| `raw/LEH_…___CODEBOOK.csv` | Qualtrics codebook: `N_Q195` = age guess, `N_Q196_1` = confidence for loop item N |
| `code/260113LEHcleaning.R` | Original cleaning script (Qualtrics wide → long) |
| `code/260114LEHanalysis.R` | Original analysis script (overconfidence t-test, RM-ANOVA) |
| `cleaned/260113LEHcleaned.csv` | **Canonical cleaned data**: 980 observations × 98 participants — `ResponseId, PGuess, PConf, MinAge, Radius, SexMale, LifeEvalQ` |
| `cleaned/fin_le_data.csv` | Cleaned data scored under study 1's empirical life-table rule (`Score`) with study-1 `Best Age`/`MAS` — kept for provenance |
| `prereg/LEH-study-AsPredicted-267677.pdf` | Pre-registration |

## Exclusions (per `code/260113LEHcleaning.R`)

`Finished == TRUE`, duration > 120 s, and the "plant" attention check → 109 raw rows → 98
participants → 980 item-level observations. All 88 conditions received responses.

## Provenance notes

- The cleaning script references an input `260113LifeEvalQuestions.csv` (the 88-row map from
  Qualtrics loop item → MinAge/Radius/SexMale). That file was not preserved, so the raw→cleaned
  step is not re-runnable as-is; `cleaned/260113LEHcleaned.csv` (from the original run) is the
  canonical dataset, and its item attributes verify exactly against the `LifeEvalQ` QID scheme.
- `fin_le_data.csv`'s `Score` uses study 1's empirical SSA life-table rule. The supplement
  notebook re-scores humans with this study's Gompertz rule (`analysis/scoring.py`) so humans
  and LLMs are compared under the same ground truth; the two rules agree closely (see the
  sensitivity section of the notebook).
