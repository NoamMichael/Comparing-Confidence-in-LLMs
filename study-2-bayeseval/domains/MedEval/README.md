# MedEval

Differential-diagnosis calibration benchmark built on [DDXPlus](https://arxiv.org/abs/2205.09148).

## Task

Each question presents a synthetic patient (age, sex, symptom list as JSON) alongside a shuffled list of candidate pathologies. The model picks a single diagnosis and reports a confidence in [0, 1].

Ground truth is the DDXPlus differential diagnosis — a per-patient probability distribution over pathologies — so `true_probability = differential[model_answer]`.

Scoring uses the Brier Score: `(confidence - true_probability)²`.

## Data

Source: `Data/release_test_patients.zip` (134,529 synthetic patients from DDXPlus).

Each patient has:
- Age, sex
- Evidence list (symptoms + antecedents as E-codes, translated via `release_evidences.json`)
- True pathology (ground-truth diagnosis)
- Differential diagnosis (probability distribution over candidate pathologies, pre-computed by DDXPlus)

49 pathologies total, 223 evidence types. Patients have 1–27 candidates in their differential (mean ~9).

## Difficulty Variants: Candidate Removal

To create difficulty levels with **exact ground-truth probabilities**, we remove candidate pathologies from the answer set rather than removing symptoms. This avoids the problem of needing to recompute posterior distributions under partial evidence (the DDXPlus generative model is proprietary and cannot be re-run).

**Why not symptom removal?** Removing symptoms changes the true posterior distribution over diagnoses. Without access to the original Bayesian network, we cannot recompute the correct ground truth — any approximation (naive Bayes, logistic regression) introduces ±0.05–0.13 noise per question on the scoring target, making fine-grained calibration claims unreliable.

**Candidate removal is exact.** If the original differential is P(d₁)=0.4, P(d₂)=0.3, P(d₃)=0.2, P(d₄)=0.1 and we remove d₄, the new ground truth is simply renormalized: P(d₁)=0.44, P(d₂)=0.33, P(d₃)=0.22. No approximation needed.

### How it works

1. Start with the full patient (all symptoms, all candidate pathologies)
2. Remove `ceil(n_candidates × pct / 100)` of the least-likely pathologies (the true pathology is never removed)
3. Renormalize remaining probabilities to sum to 1
4. Present the model with the same symptoms but fewer candidate options

### Difficulty gradient

| Variant | Mean candidates | Min candidates | Mean P(true) |
|---------|----------------|----------------|--------------|
| -0% (baseline) | 15.0 | 10 | 0.125 |
| -10% | 13.0 | 9 | 0.132 |
| -25% | 10.9 | 7 | 0.146 |
| -50% | 7.3 | 5 | 0.191 |

Fewer candidates = easier task = model should report higher confidence. A well-calibrated model adjusts its confidence to match the true probability at each difficulty level.

### Patient selection

Only patients with **10+ candidates** in their differential are used, ensuring that even at -50% removal there are at least 5 meaningful options. From the current 500-patient sample, 192 meet this criterion (768 total questions across 4 variants). The full DDXPlus test set has 54,647 patients with 10+ candidates if more are needed.

## Build

```bash
python build_benchmark.py --n 500 --min-candidates 10 \
    --removal-pcts 0,10,25,50
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--ddxplus` | `Data/` | Directory with `release_evidences.json` and `release_test_patients.zip` |
| `--split` | `test` | DDXPlus split (train/validate/test) |
| `--n` | 500 | Number of patients to sample before filtering |
| `--seed` | 0 | Random seed for patient sampling and shuffling |
| `--min-candidates` | 0 | Minimum candidates in differential (use 10 for candidate-removal) |
| `--min-findings` | 0 | Minimum symptoms per patient |
| `--removal-pcts` | `0` | Comma-separated candidate removal percentages |
| `--outdir` | `Data/` | Output directory |

### Output files

- `benchmark_remove{pct}.csv` — individual variant files
- `benchmark_combined.csv` — all variants merged with `removal_pct` column
- Columns: `question_id, question_prompt, confidence_prompt, age, sex, true_pathology, differential_json, removal_pct`

## Scoring

There is no single "correct" diagnosis — pathologies lie in a probability space defined by the DDXPlus differential. A model answering "Pneumonia" with 0.85 confidence when Pneumonia has 0.15 true probability is overconfident by 0.70, even if Pneumonia is the most likely diagnosis. For this reason, all calibration metrics (Brier Score, ECE, overconfidence) use `true_probability` from the differential rather than binary accuracy. Binary correct/incorrect would give full credit (1.0) for picking the top diagnosis regardless of how uncertain the differential actually is, masking systematic overconfidence.

Given a results CSV with `question_id, Answer, Confidence`:

```python
from analysis.scoring import score_medeval, murphy_decomposition

scored = score_medeval(results_df, benchmark_df)
print(f"Brier Score: {scored['brier'].mean():.4f}")
print(murphy_decomposition(scored))
```

The Murphy decomposition breaks Brier Score into:
- **Reliability** — how close confidence is to observed frequency (lower = better calibrated)
- **Resolution** — ability to discriminate between easy and hard questions (higher = better)
- **Uncertainty** — irreducible difficulty of the question set

## Expected model response format

```json
{"Answer": "Pulmonary neoplasm", "Confidence": "0.35"}
```

Answer matching is case-insensitive and whitespace-insensitive.
