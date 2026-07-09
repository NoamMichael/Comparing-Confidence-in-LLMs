# WGD (Weight Guessing Dataset) — Pipeline Diagram Specification

Generate a pipeline/flowchart diagram for the WGD domain of the BayesEval benchmark. The diagram should clearly show two parallel tracks: **DCE (Direct Confidence Elicitation)** and **SPD (Sampled Predictive Distribution)**, which share infrastructure but diverge at the prompting and parsing stages.

## Data Sources (shared entry point)

1. **labels.csv** — Master participant table with columns: Participant Number, Age, Weight (lbs), sex, ethnicity
2. **Photos/*.JPG** — Raw photographs of participants (one per person)

These two sources are joined on Participant Number = photo filename stem. Participants without a photo or without a recorded weight are filtered out.

## Benchmark Construction (two parallel tracks diverge here)

### DCE Track: `build_benchmark.py`

- **Input:** Filtered (participant, weight) pairs + photo filenames
- **Cross-product expansion:** Each participant × 20 tolerance levels (`within_lbs` = 1, 2, ..., 20)
- **Output:** `benchmark.csv` with columns:
  - `question_id` (e.g., `wgd_136_05`)
  - `question_prompt` — "Look at this photo... estimate weight... how confident within {within_lbs} lbs?"
  - `confidence_prompt` — JSON format instruction
  - `photo`, `within_lbs`, `true_weight`
- **Scale:** ~46 participants × 20 tolerances = ~920 rows

### SPD Track: `build_spd_benchmark.py`

- **Input:** Same filtered participants
- **Cross-product expansion:** Each participant × 4 bin widths (2, 10, 20, 40 lbs)
- **Bin width → top-N mapping:** {2→10, 10→10, 20→5, 40→3} (wider bins = fewer requested)
- **Output:** `benchmark_spd.csv` with columns:
  - `question_id` (e.g., `wgd_spd_136_10`)
  - `question_prompt` — "Look at this photo... estimate weight..."
  - `confidence_prompt` — "Report your top-N most likely weight ranges of exactly {bw} lbs width as a JSON array [{min, max, confidence}, ...]"
  - `photo`, `within_lbs` (= bin_width / 2), `true_weight`, `bin_width`, `top_n`
- **Scale:** ~46 participants × 4 bin widths = ~184 rows

## Image Encoding (shared, lazy)

- **`image_encoder.py`** runs before evaluation
- Each JPEG in `Photos/` is base64-encoded → written to `Photos-encoded/{stem}.txt` as a `data:image/jpeg;base64,...` URI
- A manifest `image_sizes.json` tracks which photos have been encoded (idempotent — skips already-encoded photos)
- At eval time, each row's `photo` column resolves to an `image_path` pointing to the `.txt` encoded file

## Evaluation (`eval.py` → `executor.py` → `openrouter_client.py`)

The evaluator loads the benchmark CSV, reads config.yaml for model list and concurrency settings, and dispatches async API calls through OpenRouter.

### DCE Evaluation Path

1. `eval.py:load_domain("WGD")` loads `benchmark.csv`
2. `run_task(spd=False)` calls `_run_one()` per row
3. `_run_one()`:
   - Reads base64 image from `.txt` file
   - Calls `openrouter_client.complete(model, question_prompt, confidence_prompt, image_uri)`
   - Sends multimodal message: [image_url block, text block] + appended JSON format instruction
   - Parses response as `{"Answer": "<weight>", "Confidence": "0.XX"}`
4. Returns DataFrame: `question_id, Answer, Confidence, raw, error`

### SPD Evaluation Path

1. `eval.py:load_domain("WGD_SPD")` loads `benchmark_spd.csv`
2. `run_task(spd=True)` calls `_run_one_spd()` per row
3. `_run_one_spd()`:
   - Reads base64 image from `.txt` file
   - Calls `openrouter_client.complete_raw()` — sends question_prompt + confidence_prompt verbatim (no JSON format instruction appended)
   - Model responds with JSON array: `[{min, max, confidence}, ...]`
4. `parse_spd_bins()`:
   - Regex-extracts the JSON array from raw response
   - Identifies the **modal bin** (highest confidence entry)
   - `Answer` = center of modal bin: `(min + max) / 2`
   - `Confidence` = modal bin's confidence value
5. Returns DataFrame with same schema as DCE

## Scoring (`analysis/scoring.py:score_wgd`)

Both DCE and SPD results are scored identically by `score_wgd()`:

1. **True probability** (binary, answer-dependent):
   - `true_probability = 1.0` if `|Answer - true_weight| <= within_lbs`
   - `true_probability = 0.0` otherwise
   - Note: unlike other domains, ground truth depends on the model's own answer
2. **Brier score:** `(Confidence - true_probability)²`
3. Unparseable answers (NaN) produce NaN scores

For SPD: `within_lbs = bin_width / 2`, so the modal bin's center is checked against the half-width tolerance.

## Analysis (`analysis/analysis.ipynb`)

- **RQ1 (Calibration):** Bins model confidences into 11 bins (0–1.0), computes fraction correct per bin, plots calibration curve against the diagonal. Computes ECE (Expected Calibration Error).
- **RQ2 (Difficulty):** Groups by `within_lbs` (difficulty axis), computes mean overconfidence (`Confidence - true_probability`) per group per model. Wider tolerance = easier question.
- **RQ3 (SPD vs DCE):** Compares ECE between DCE baseline and SPD via two-sided bootstrap test (n=2000). Delta ECE = ECE_baseline - ECE_SPD (positive = SPD improved calibration).
- **Bias analysis:** Joins on participant demographics (sex) from labels.csv, computes directional weight estimation error by sex, runs Welch's t-test.
- **Murphy decomposition:** BS = Reliability - Resolution + Uncertainty

## End-to-End Flow Summary

```
labels.csv ──────┐
                 ├──[join on participant ID]──┐
Photos/*.JPG ────┘                            │
                                              ▼
                                    ┌─────────────────────┐
                                    │  Filter: has photo   │
                                    │  AND has weight       │
                                    └────────┬────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                              ▼
                    build_benchmark.py              build_spd_benchmark.py
                    (× 20 tolerances)              (× 4 bin widths)
                              │                              │
                              ▼                              ▼
                       benchmark.csv                 benchmark_spd.csv
                              │                              │
                              ▼                              ▼
                    image_encoder.py ◄──── lazy encoding ────►
                    (base64 photos)
                              │                              │
                              ▼                              ▼
                    ┌─────────────────┐          ┌──────────────────────┐
                    │  DCE Evaluation  │          │   SPD Evaluation      │
                    │  complete()      │          │   complete_raw()      │
                    │  → {Answer,      │          │   → [{min,max,conf}]  │
                    │     Confidence}  │          │   → parse_spd_bins()  │
                    │                  │          │   → modal bin center  │
                    └────────┬────────┘          └──────────┬───────────┘
                             │                              │
                             ▼                              ▼
                    results/WGD/{model}.csv       results/WGD_SPD/{model}.csv
                             │                              │
                             └──────────┬───────────────────┘
                                        ▼
                              score_wgd()
                              true_prob = 1 if |Answer - true_weight| ≤ within_lbs
                              brier = (Confidence - true_prob)²
                                        │
                                        ▼
                              analysis.ipynb
                              (calibration plots, ECE, Murphy,
                               RQ1/RQ2/RQ3, bias analysis)
```
