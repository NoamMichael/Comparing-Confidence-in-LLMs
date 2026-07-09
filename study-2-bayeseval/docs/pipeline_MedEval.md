# MedEval — Pipeline Diagram Specification

Generate a pipeline/flowchart diagram for the MedEval domain of the BayesEval benchmark. This is a medical differential diagnosis calibration benchmark using the DDXPlus dataset. The diagram should clearly show two parallel tracks: **DCE (Direct Confidence Elicitation)** and **SPD (Sampled Predictive Distribution)**, and highlight the candidate-removal mechanism that controls question difficulty.

## Data Sources

1. **release_evidences.json** — Evidence code → human-readable question/value mapping (translates DDXPlus evidence codes like `"E_42"` into English findings like `"Do you have a fever? — Yes"`)
2. **release_test_patients.zip** — DDXPlus test set patients CSV containing columns:
   - `PATHOLOGY` — true diagnosis
   - `EVIDENCES` — list of evidence codes present for this patient
   - `DIFFERENTIAL_DIAGNOSIS` — Python-literal list of `(pathology_name, probability)` tuples. This is the **analytical ground-truth probability distribution** over all candidate pathologies.
   - Patient demographics (age, sex)

## Benchmark Construction (two parallel tracks)

Both tracks share a common preprocessing step:

### Shared Preprocessing

1. Read patients from ZIP, shuffle with fixed seed, take first N patients
2. Optional filters: `--min-findings`, `--min-candidates`
3. **Candidate removal** (difficulty control): For each removal percentage in the configured set (e.g., 0%, 10%, 25%, 50%):
   - `reduce_candidates()` removes the least-likely candidates from the differential diagnosis
   - The true pathology is never removed
   - Remaining probabilities are renormalized to sum to 1.0
   - Higher removal % = fewer candidates = easier question
4. **Vignette construction** (`build_vignette()`): Creates a JSON structure `{patient: {age, sex}, findings: [{finding, value}, ...]}` using `render_evidence()` to translate evidence codes via `release_evidences.json`

### DCE Track: `build_benchmark.py`

- **Per-row construction:**
  - `question_prompt`: Clinical vignette + shuffled list of candidate pathologies + instruction to select the most likely diagnosis
  - `confidence_prompt`: "How confident are you?" with `n_candidates` and `uniform_prior` (1/n) provided as context
  - `differential_json`: The renormalized differential stored as JSON string `[[name, prob], ...]`
- **Output:** One CSV per removal percentage (`benchmark_remove{pct}.csv`) + a combined `benchmark_combined.csv` with `removal_pct` column
- **Columns:** `question_id` (e.g., `med_test_00042_c25`), `question_prompt`, `confidence_prompt`, `differential_json`, `removal_pct`, patient metadata
- **Question ID format:** `med_{split}_{index:05d}_c{removal_pct}`

### SPD Track: `build_spd_benchmark.py`

- **Same patient pool** and same candidate-removal logic as DCE
- **Key difference in prompting:**
  - `question_prompt`: Asks model to assign a probability to **each** candidate pathology (not just pick one)
  - `confidence_prompt`: "For each candidate pathology, report your estimated probability that it is the correct diagnosis. Probabilities should sum to 1.0. Respond with ONLY a JSON array: [{pathology, confidence}, ...]"
- **Iterates over** `--candidate-removal-pcts` (default: 0, 10, 25, 50)
- **Output:** Single `benchmark_spd.csv` with all removal percentages combined
- **Question ID format:** `med_spd_{split}_{index:05d}_c{removal_pct}`

## Evaluation (`eval.py` → `executor.py` → `openrouter_client.py`)

### DCE Evaluation Path

1. `eval.py:load_domain("MedEval")` loads `benchmark_combined.csv` (or a per-removal CSV)
2. `run_task(spd=False)` calls `_run_one()` per row
3. `_run_one()`:
   - Calls `openrouter_client.complete(model, question_prompt, confidence_prompt)`
   - Appends JSON format instruction: `{"Answer": "<your estimate>", "Confidence": "<probability>"}`
   - Model picks one pathology and states a scalar confidence
   - Parses response as `{"Answer": "Pneumonia", "Confidence": "0.75"}`
4. Returns DataFrame: `question_id, Answer, Confidence, raw, error`

### SPD Evaluation Path

1. `eval.py:load_domain("MedEval_SPD")` loads `benchmark_spd.csv`
2. `run_task(spd=True)` calls `_run_one_spd()` per row
3. `_run_one_spd()`:
   - Calls `openrouter_client.complete_raw()` — sends prompts verbatim, no JSON format instruction appended
   - Model responds with JSON array: `[{pathology, confidence}, ...]`
4. `parse_spd_candidates()` (routed because `bin_width` column is **absent**):
   - Regex-extracts JSON array from raw response
   - Identifies the **modal candidate** (highest confidence entry)
   - `Answer` = that pathology's name
   - `Confidence` = that pathology's confidence value
5. Returns DataFrame with same schema as DCE

Note: The SPD parser uses `parse_spd_candidates()` (not `parse_spd_bins()` used by WGD and LifeEval). The routing is determined by whether `bin_width` exists as a column — MedEval uses discrete pathology candidates rather than continuous numeric bins.

## Scoring (`analysis/scoring.py:score_medeval`)

Both DCE and SPD results are scored identically by `score_medeval()`:

1. **Merge** `differential_json` from benchmark onto results
2. **For each row**, call `medeval_true_probability(answer, differential_json)`:
   - Parses `differential_json` via `json.loads` → list of `[pathology_name, probability]` pairs
   - Normalizes both the model's answer and each candidate name via `_normalize_pathology()` (strip whitespace, lowercase)
   - Looks up the model's chosen pathology in the differential
   - Returns that pathology's ground-truth probability, or **0.0** if not found
3. **Brier score:** `(Confidence - true_probability)²`
4. Unparseable answers produce NaN scores

Key distinction: true_probability comes from a lookup in the DDXPlus differential distribution (a discrete probability distribution over diagnoses), not from a statistical model or binary check.

## Candidate Removal as Difficulty Axis

The candidate-removal mechanism is central to MedEval's experimental design:

- **0% removal:** Full differential — all candidate pathologies present (hardest)
- **10% removal:** Least-likely 10% of candidates removed, probabilities renormalized
- **25% removal:** Least-likely 25% removed
- **50% removal:** Least-likely 50% removed (easiest — fewer distractors)

The true pathology is **never** removed. After removal, remaining probabilities are renormalized to sum to 1.0. This means:
- The true pathology's ground-truth probability increases with more removal (its share of the pie grows)
- Models should increase confidence as candidates are removed (fewer alternatives)
- RQ2 tests whether models appropriately adjust confidence as difficulty changes

## Analysis (`analysis/analysis.ipynb`)

- **RQ1 (Calibration):** Bins model confidences into 11 bins, computes mean `true_probability` per bin, plots calibration curve. Computes ECE.
- **RQ2 (Difficulty):** Groups by `removal_pct` (difficulty axis), computes mean overconfidence (`Confidence - true_probability`) per removal level per model. Higher removal % = easier.
- **RQ3 (SPD vs DCE):** Compares ECE between DCE baseline and SPD. Bootstrap significance test (n=2000). Delta ECE = ECE_baseline - ECE_SPD (positive = SPD improved calibration).
- **Murphy decomposition:** BS = Reliability - Resolution + Uncertainty

## End-to-End Flow Summary

```
release_test_patients.zip          release_evidences.json
(DDXPlus patients with             (evidence code → English
 PATHOLOGY, EVIDENCES,              question/value translation)
 DIFFERENTIAL_DIAGNOSIS)
            │                                │
            ▼                                │
  Shuffle + sample N patients                │
  Optional: filter by                        │
  min-findings, min-candidates               │
            │                                │
            ▼                                ▼
  ┌──────────────────────────────────────────────┐
  │  For each removal_pct ∈ {0, 10, 25, 50}:    │
  │    reduce_candidates()                        │
  │    → remove least-likely candidates           │
  │    → never remove true pathology              │
  │    → renormalize probabilities                │
  │                                               │
  │  build_vignette()                             │
  │    → render_evidence() using evidences.json   │
  │    → {patient: {age,sex}, findings: [...]}    │
  └──────────────────┬───────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                      ▼
build_benchmark.py (DCE)   build_spd_benchmark.py (SPD)
"Pick the most likely       "Assign probability to
 diagnosis + confidence"     each candidate diagnosis"
→ {Answer, Confidence}      → [{pathology, confidence},...]
          │                      │
          ▼                      ▼
benchmark_combined.csv      benchmark_spd.csv
(+ per-pct CSVs)            (all removal pcts combined)
          │                      │
          ▼                      ▼
┌──────────────────┐    ┌─────────────────────────┐
│ DCE Evaluation    │    │ SPD Evaluation           │
│ complete()        │    │ complete_raw()           │
│ → {Answer:        │    │ → [{pathology,conf},...]  │
│   "Pneumonia",    │    │ → parse_spd_candidates() │
│    Confidence:    │    │ → modal candidate +      │
│    "0.75"}        │    │   its confidence         │
└────────┬─────────┘    └───────────┬─────────────┘
         │                          │
         ▼                          ▼
results/MedEval/          results/MedEval_SPD/
{model}.csv               {model}.csv
         │                          │
         └────────────┬─────────────┘
                      ▼
            score_medeval()
            Parse differential_json
            Normalize pathology names
            true_prob = lookup in DDXPlus
              differential distribution
            (0.0 if answer not found)
            brier = (Confidence - true_prob)²
                      │
                      ▼
            analysis.ipynb
            (calibration plots, ECE, Murphy,
             RQ1/RQ2/RQ3, difficulty by removal_pct)
```
