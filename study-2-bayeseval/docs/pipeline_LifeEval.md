# LifeEval — Pipeline Diagram Specification

Generate a pipeline/flowchart diagram for the LifeEval domain of the BayesEval benchmark. This is an actuarial mortality calibration benchmark scored directly against the SSA 2022 period life table (empirical rule — no parametric fit). The diagram should clearly show two parallel tracks: **DCE (Direct Confidence Elicitation)** and **SPD (Sampled Predictive Distribution)**, and highlight the life-table loading step that feeds into both benchmark construction and scoring.

## Data Source

**PeriodLifeTable_2022_RawData.csv** — 2022 US Period Life Table containing 119 rows (ages 0–118) with columns:
- `Age`
- `Death probability (MALE)` / `Death probability (FEMALE)` — 1-year mortality probability q_x
- `Life expectancy (MALE)` / `Life expectancy (FEMALE)` — remaining life expectancy e_x

## Life-Table Loading (`scoring.py:_get_life_table_qx` / `_death_mass`)

This step is shared by both benchmark construction and scoring. No parametric model is fitted — the table's per-year death probabilities are used directly (study 1's original empirical rule; an earlier Gompertz-fit version survives in git history).

- **`_get_life_table_qx()`:** parses the CSV once per process into per-sex `q_x` arrays (ages 0–118), dropping the junk trailing row and asserting the age grid is contiguous
- **`_death_mass(sex, min_age)`:** for conditioning age `m`, computes `d[x−m] = S_rel(x) · q_x` where `S_rel(x) = Π_{k=m}^{x−1}(1−q_k)` — the probability of dying in year `[x, x+1)` given survival to `m`
- **Caching:** `q_x` arrays and per-`(sex, m)` death-mass vectors are cached lazily in module-level dicts

## Benchmark Construction (two parallel tracks)

### DCE Track: `build_benchmark.py`

- **Input:** Life table (life-expectancy column for metadata; `q_x` via `analysis.scoring`)
- **Loop:** For each sex ∈ {male, female}, for each age ∈ 0–100, for each radius ∈ 1–20
- **Per-row computation:**
  - `true_lifespan` = age + life_expectancy (from life table, stored as metadata)
  - `best_answer` (y*): `scoring.lifeeval_best_answer_and_mas` — discrete argmax over integer guesses y ∈ [age, 118] of the empirical window probability; smallest age wins ties (study-1 convention)
  - `MAS` (Maximum Achievable Score): the window probability at y* — the theoretical ceiling for this question
  - `gold_response`: `{"Answer": y*, "Confidence": round(MAS, 2)}` — the ideal perfectly-calibrated response
  - window probability = P(death in [floor(y−r), ceil(y+r)) | survived to age a) = Σ over window of `S_rel(x) · q_x`, clamped to [a, 119)
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
   - Retrieves the cached death-mass vector for `(sex, min_age)`
   - `lo = max(floor(answer − radius), min_age)` — clamp lower bound to conditioning age
   - `hi = min(ceil(answer + radius), 119)` — clamp upper bound to the table's last age
   - `true_probability = Σ_{x=lo}^{hi−1} S_rel(x) · q_x` (0.0 if the window is empty)
   - This is a **graded** probability (not binary) — it's the empirical mass the life table assigns to the window centered on the model's guess
3. **Brier score:** `(Confidence - true_probability)²`
4. Unparseable answers produce NaN scores

Key distinction from WGD: true_probability is graded (a table probability mass), not binary (hit/miss).

## Analysis (`analysis/analysis.ipynb`)

- **RQ1 (Calibration):** Bins model confidences into 11 bins, computes mean `true_probability` per bin (not fraction correct — because true_probability is continuous), plots calibration curve. Computes ECE.
- **RQ2 (Difficulty):** Groups by `radius` (difficulty axis), computes mean overconfidence (`Confidence - true_probability`) per group. Larger radius = easier = wider window = higher true_probability.
- **RQ3 (SPD vs DCE):** Compares ECE between DCE baseline and SPD. Bootstrap significance test (n=2000). Delta ECE = ECE_baseline - ECE_SPD.
- **Murphy decomposition:** BS = Reliability - Resolution + Uncertainty
- **Illustration:** Overlays the empirical conditional death-probability curve against DCE tolerance window and SPD bin distribution for a specific question

## End-to-End Flow Summary

```
PeriodLifeTable_2022_RawData.csv
            │
            ▼
  _get_life_table_qx()
  per-sex q_x arrays (ages 0–118)
  d[x−m] = S_rel(x)·q_x  (per-year death mass,
  conditional on survival to m)
            │
            ├────────────────────────────────────────────┐
            ▼                                            ▼
  build_benchmark.py (DCE)                   build_spd_benchmark.py (SPD)
  101 ages × 2 sexes × 20 radii             101 ages × 2 sexes × 4 bin widths
  Precomputes best_answer & MAS              No precomputed optimal answers
  via discrete argmax over                   radius = bin_width / 2
  integer guesses (empirical rule)           top_n from {2→10,10→10,20→5,40→3}
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
                  cached death-mass vectors (same table)
                  true_prob = Σ_{x=lo}^{hi−1} S_rel(x)·q_x
                    window [floor(y−r), ceil(y+r)) clamped to [a, 119)
                  brier = (Confidence - true_prob)²
                            │
                            ▼
                  analysis.ipynb
                  (calibration plots, ECE, Murphy,
                   RQ1/RQ2/RQ3, empirical death-mass overlay)
```
