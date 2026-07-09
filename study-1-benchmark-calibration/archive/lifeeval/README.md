# LifeEval Archive (Study 1)

LifeEval was introduced in Study 1 as an estimation benchmark with ground-truth
probabilities derived from the U.S. SSA 2022 Period Life Tables: 808 questions
(2 genders × 101 ages × 4 tolerance radii r ∈ {1, 5, 10, 20}), scored as
P(death within [estimate − r, estimate + r] | survived to current age).

LifeEval is **no longer part of Study 1's active pipeline**. It has been expanded
and is actively developed in **Study 2 (BayesEval)** — see
[`../../../study-2-bayeseval/domains/LifeEval/`](../../../study-2-bayeseval/domains/LifeEval/),
which extends the task to 20 radii (4,040 questions) and scores it with proper
scoring rules (Brier + Murphy decomposition).

This directory preserves the complete preregistered record
([OSF preregistration](https://osf.io/y8rqv/)) so Study 1's original results
remain reproducible.

## Contents

| Path | What it is |
|------|------------|
| `data/life_eval_formatted.csv` | The 808 formatted benchmark questions |
| `data/PeriodLifeTable_2022_RawData.csv` | SSA 2022 period life table (ground truth) |
| `data/fin_le_data.csv` | Final LifeEval analysis data |
| `prompts/life_eval_prompts.csv` | Prompts with confidence-elicitation instructions |
| `batches/<model>/` | Batch API request files (the models tracked in git) |
| `parsed-results/` | Parsed model responses, one CSV per model (11 models) |
| `plots/` | LifeEval figures (calibration plots, gender differentials, radii analysis) |
| `r-processed/life_eval.csv` | Output of the R processing pipeline |
| `best_results_le.csv` | Best-age / MAS table computed in analysis.ipynb |
| `results_metadata_lifeeval.json` | Batch job IDs (extracted from `../../results_metadata.json`) |
| `ssa-contamination/` | Contamination analysis (see below) |

## SSA contamination analysis (`ssa-contamination/`)

A two-stage check (contributed by [@ddanie1](https://github.com/ddanie1)) of
whether models' LifeEval reasoning shows evidence of having memorized the actual
SSA life tables, which would inflate apparent accuracy/calibration:

1. **Keyword flagging** — `ssa_flag_viewer.ipynb`, `analyze_ssa_references.py`,
   `life_eval_ssa_analysis.csv`, `ssa_reference_examples.csv`,
   `ssa_reference_report.txt`: flags responses whose reasoning mentions
   SSA / life tables / actuarial terms (~75.6% of 8,261 responses).
2. **LLM-judge verification** — `LLM_SSA_Analysis/`: a Claude judge classifies
   each flagged response as `no_evidence` / `weak_evidence` / `strong_evidence`
   of table memorization (submission scripts, raw batch outputs, result CSVs),
   and `generate_subset_table.py` / `analyze_lifeeval_subset.py` recompute the
   calibration metrics excluding contaminated subsets (`subset_*.tex`,
   `plots/LifeEval_Subset*/`). See `SSA_APENDIX.md` for the write-up.

## Reproducibility notes

- `Combined Results/combined_raw.csv` and `combined_clean.csv` (in the study-1
  root) **intentionally still contain LifeEval rows** — they are the canonical
  preregistered record. Active notebooks filter them out at load time with
  `df[df["Question Set"] != "LifeEval"]`.
- Re-running the stripped pipeline (`combine.py` → `clean.py`) produces
  LifeEval-free CSVs; this divergence from the canonical files is expected.
- The LifeEval scoring code removed from the active pipeline (`compute_prob`,
  `score_life_eval`, etc. in `combine.py`; `clean_LifeEval` in `clean.py`;
  `life_eval_prompts` in `Workflow/batch_processing.py`) is preserved in git
  history — see the "Strip LifeEval" commits on this branch, or Study 2's
  `analysis/scoring.py` for the successor implementation.
