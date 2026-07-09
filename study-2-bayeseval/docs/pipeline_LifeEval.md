# LifeEval — Pipeline Diagram Specification

Generate a pipeline/flowchart diagram for the LifeEval domain of the BayesEval benchmark. This is an actuarial mortality calibration benchmark using the Gompertz survival model. The diagram should clearly show two parallel tracks: **DCE (Direct Confidence Elicitation)** and **SPD (Sampled Predictive Distribution)**, and highlight the Gompertz model fitting step that feeds into both benchmark construction and scoring.

## Data Source

**PeriodLifeTable_2022_RawData.csv** — 2022 US Period Life Table containing 119 rows (ages 0–118) with columns:
- `Age`
- `Death probability (MALE)` / `Death probability (FEMALE)` — 1-year mortality probability q_x
- `Life expectancy (MALE)` / `Life expectancy (FEMALE)` — remaining life expectancy e_x

## Gompertz Model Fitting (`scoring.py:fit_gompertz_to_life_table`)

This step is shared by both benchmark construction and scoring. It produces parameters used throughout the pipeline.

- **Hazard model:** `h(x) = b * exp(c * x)` (Gompertz law of mortality)
- **Fit range:** Ages 5–94 only (excludes infant mortality bulge at 0–4 and sparse tail at 95+)
- **Method:** Maximum Likelihood Estimation via `scipy.optimize.minimize` (Nelder-Mead)
- **Parameterization:** `log(b)` is optimized to enforce positivity; initial guess `log(b) = log(1e-5)`, `c = 0.085`
- **Output:** `GompertzParams(b, c)` fitted separately for male and female
- **Caching:** Parameters are computed once per process via a global cache (`_GOMPERTZ_PARAMS`), populated lazily

## Benchmark Construction (two parallel tracks)

### DCE Track: `build_benchmark.py`

- **Input:** Life table + fitted Gompertz parameters (male and female)
- **Loop:** For each sex ∈ {male, female}, for each age ∈ 0–100, for each radius ∈ 1–20
- **Per-row computation:**
  - `true_lifespan` = age + life_expectancy (from life table, stored as metadata)
  - `best_answer` (y*): found via `scipy.optimize.minimize_scalar` over [age, 130] — the age-at-death guess that maximizes the window probability
  - `MAS` (Maximum Achievable Score): `window_probability(best_y, age, radius, params)` — the theoretical ceiling probability for this question
  - `gold_response`: `{"Answer": round(best_y), "Confidence": round(MAS, 2)}` — the ideal perfectly-calibrated response
  - `window_probability(y, a, r, params)` = P(death in [y-r, y+r] | survived to age a) = S(max(y-r, a) | a) - S(y+r | a)
  - where `S(x | a) = exp(-(b/c)(exp(cx) - exp(ca)))` is the conditional survival function
- **Output:** `benchmark.csv` with columns:
  - `question_id` (sequential integer 0–4039)
  - `question_prompt` — "Given that an American {sex} has lived at least {age} years, estimate how old they will be when they die."
  - `confidence_prompt` — "How certain are you that your answer is within {radius} year(s) of the true value?"
  - `true_lifespan`, `min_age`, `sex`, `radius`, `best_answer`, `MAS`, `gold_response`
- **Scale:** 101 ages × 2 sexes × 20 radii = **4,040 rows**

### SPD Track: `build_spd_benchmark.py`

- **Input:** Life table (no precomputed best_answer or MAS needed)
- **Loop:** For each sex ∈ {male, female}, for each age ∈ 0–100, for each bin_width ∈ {2, 10, 20, 40}
- **Bin width → top-N mapping:** {2→10, 10→10, 20→5, 40→3}
- **`radius` column:** set to `bin_width / 2` (compatibility shim for shared scoring logic)
- **Output:** `benchmark_spd.csv` with columns:
  - `question_id` (e.g., `le_spd_50_male_10`)
  - `question_prompt` — same as DCE (estimate age at death)
  - `confidence_prompt` — "Using age ranges of exactly {bw} years, report your top {top_n} most likely ranges as JSON array [{min, max, confidence}, ...]"
  - `min_age`, `sex`, `radius`, `bin_width`, `top_n`
  - No `true_lifespan`, `best_answer`, `MAS`, or `gold_response` columns
- **Scale:** 101 ages × 2 sexes × 4 bin widths = **808 rows**

## Evaluation (`eval.py` → `executor.py` → `openrouter_client.py`)

### DCE Evaluation Path

1. `eval.py:load_domain("LifeEval")` loads `benchmark.csv`
2. `run_task(spd=False)` calls `_run_one()` per row
3. `_run_one()`:
   - Calls `openrouter_client.complete(model, question_prompt, confidence_prompt)`
   - Appends JSON format instruction: `{"Answer": "<your estimate>", "Confidence": "<probability>"}`
   - Parses response as `{"Answer": "83", "Confidence": "0.06"}`
4. Returns DataFrame: `question_id, Answer, Confidence, raw, error`

### SPD Evaluation Path

1. `eval.py:load_domain("LifeEval_SPD")` loads `benchmark_spd.csv` (from `domains/LifeEval/Data/`)
2. `run_task(spd=True)` calls `_run_one_spd()` per row
3. `_run_one_spd()`:
   - Calls `openrouter_client.complete_raw()` — sends prompts verbatim, no JSON format instruction appended
   - Model responds with JSON array: `[{min, max, confidence}, ...]`
4. `parse_spd_bins()` (routed because `bin_width` column exists):
   - Regex-extracts JSON array from raw response
   - Identifies the **modal bin** (highest confidence entry)
   - `Answer` = center of modal bin: `(min + max) / 2`
   - `Confidence` = modal bin's confidence value
5. Returns DataFrame with same schema as DCE

## Scoring (`analysis/scoring.py:score_lifeeval`)

Both DCE and SPD results are scored identically by `score_lifeeval()`:

1. **Merge** benchmark columns (`min_age`, `sex`, `radius`) onto results
2. **For each row**, call `lifeeval_true_probability(answer, min_age, sex, radius)`:
   - Retrieves cached Gompertz params for the sex
   - Computes `_window_probability(answer, min_age, radius, params)`:
     - `lo = max(answer - radius, min_age)` — clamp lower bound to conditioning age
     - `hi = answer + radius`
     - `true_probability = S(lo | min_age) - S(hi | min_age)`
     - where `S(x | a) = exp(-(b/c)(exp(cx) - exp(ca)))`
   - This is a **continuous** probability (not binary) — it's the mass the Gompertz distribution assigns to the window centered on the model's guess
3. **Brier score:** `(Confidence - true_probability)²`
4. Unparseable answers produce NaN scores

Key distinction from WGD: true_probability is continuous (Gompertz CDF integral), not binary (hit/miss).

## Analysis (`analysis/analysis.ipynb`)

- **RQ1 (Calibration):** Bins model confidences into 11 bins, computes mean `true_probability` per bin (not fraction correct — because true_probability is continuous), plots calibration curve. Computes ECE.
- **RQ2 (Difficulty):** Groups by `radius` (difficulty axis), computes mean overconfidence (`Confidence - true_probability`) per group. Larger radius = easier = wider window = higher true_probability.
- **RQ3 (SPD vs DCE):** Compares ECE between DCE baseline and SPD. Bootstrap significance test (n=2000). Delta ECE = ECE_baseline - ECE_SPD.
- **Murphy decomposition:** BS = Reliability - Resolution + Uncertainty
- **Illustration:** Overlays Gompertz conditional PDF against DCE tolerance window and SPD bin distribution for a specific question

## End-to-End Flow Summary

```
PeriodLifeTable_2022_RawData.csv
            │
            ▼
  fit_gompertz_to_life_table()
  MLE on ages 5–94
  h(x) = b·exp(c·x)
  → GompertzParams(b, c) × {male, female}
            │
            ├────────────────────────────────────────────┐
            ▼                                            ▼
  build_benchmark.py (DCE)                   build_spd_benchmark.py (SPD)
  101 ages × 2 sexes × 20 radii             101 ages × 2 sexes × 4 bin widths
  Precomputes best_answer & MAS              No precomputed optimal answers
  via minimize_scalar on                     radius = bin_width / 2
  window_probability                         top_n from {2→10,10→10,20→5,40→3}
            │                                            │
            ▼                                            ▼
     benchmark.csv (4040 rows)               benchmark_spd.csv (808 rows)
            │                                            │
            ▼                                            ▼
  ┌──────────────────────┐               ┌────────────────────────────┐
  │  DCE Evaluation       │               │  SPD Evaluation             │
  │  complete()           │               │  complete_raw()             │
  │  → {Answer: "83",     │               │  → [{min,max,confidence}]   │
  │     Confidence:"0.06"}│               │  → parse_spd_bins()         │
  │                       │               │  → modal bin center + conf  │
  └──────────┬───────────┘               └─────────────┬──────────────┘
             │                                          │
             ▼                                          ▼
  results/LifeEval/{model}.csv           results/LifeEval_SPD/{model}.csv
             │                                          │
             └──────────────┬───────────────────────────┘
                            ▼
                  score_lifeeval()
                  Gompertz params (cached, same fit)
                  true_prob = S(lo|a) - S(hi|a)
                    where S(x|a) = exp(-(b/c)(exp(cx)-exp(ca)))
                  brier = (Confidence - true_prob)²
                            │
                            ▼
                  analysis.ipynb
                  (calibration plots, ECE, Murphy,
                   RQ1/RQ2/RQ3, Gompertz PDF overlay)
```
