# paper/ — consolidated paper resources

Everything here is a **copy**, collected from the two study folders by
`scripts/retrieve_paper_resources.py`. Originals stay in place; to refresh after
regenerating any figure or table upstream, re-run:

```bash
python3 paper/scripts/retrieve_paper_resources.py
```

## Selection criteria

Keep figures/tables that summarize **across models and domains** — the things a paper
would actually show (aggregate calibration curves, per-model metric comparisons,
RQ headline panels, dataset descriptives). Skip per-model × per-question-set granular
plots (e.g. study 1's 55 individual `cal_plot_*` diagrams), poster presentation copies,
and intermediate data files.

## Contents

### `tex/` — LaTeX tables

| File | Source | Contents |
|---|---|---|
| `study1_table1.tex` | `study-1-benchmark-calibration/R/analysis/table1-similar-layout.csv` (converted via pandas) | Table 1: N / ECE / accuracy / overconfidence for 6 question sets × 11 models, from the preregistered R pipeline |
| `study1_summary_reasoning.tex` | extracted from `study-1-benchmark-calibration/analysis.ipynb` output | Accuracy / confidence / ECE / % rounded-confidence per question set, reasoning models |
| `study1_summary_chat.tex` | extracted from `study-1-benchmark-calibration/analysis.ipynb` output | Same, chat (non-reasoning) models |
| `study2_rq1_summary.tex` | `study-2-bayeseval/analysis/rq1_summary_table.txt` | Study 2 master results: accuracy, mean conf., overconfidence, β₁, ECE, ECE_SPD, ΔECE% per domain × model |
| `study2_sensitivity_ymax.tex` | `study-2-bayeseval/analysis/sensitivity_ymax_table.txt` | Y_max robustness of the LifeEval difficulty metric |

Notes:
- The two `study1_summary_*` tables are pulled from notebook cell outputs; the script
  keys on model-name markers (not cell indices) and picks the complete variant when the
  notebook holds several. Captions/labels are normalized by the script.
- The notebook outputs and the R Table 1 CSV predate LifeEval being dropped from
  study 1; the script **removes the LifeEval rows** from all three study-1 tables at
  retrieval time.

### `plots/study1/` — from `study-1-benchmark-calibration/Plots/`

| File | Contents |
|---|---|
| `aggregate_extended_{all_qsets,mcq,2afc}_cal_plot.png` | Aggregate reliability curves pooling all models (all sets / MCQ only / 2AFC only) |
| `reasoning_vs_chat_by_qset_cal_plot.png` | Reasoning vs chat model calibration, per benchmark |
| `calibration_grid_all_models_all_qsets.png` | 11×5 composite grid of every model × benchmark reliability diagram (appendix) |
| `ece_all.png`, `oc_all.png`, `acc_all.png` | ECE / overconfidence / accuracy per model, faceted by question set |
| `stated_vs_token_ece.png` | Stated-confidence ECE vs token-probability ECE (token-capable models) |

### `plots/study2/` — from `study-2-bayeseval/analysis/figs/`

| File | Contents |
|---|---|
| `calibration_combined.png` | RQ1: calibration curves, 3-domain panel, all models (DCE) |
| `calibration_spd_combined.png` | RQ3: calibration curves under SPD prompting |
| `overconfidence_by_difficulty.png`, `rq2_overconfidence_by_percentile.png` | RQ2: hard–easy effect |
| `rq3_overconfidence_by_percentile_spd.png`, `rq3_ece_spd_improvement.png`, `rq3_ece_pct_change.png` | RQ3: SPD effect on calibration |
| `sensitivity_ymax.png` | Y_max robustness figure |
| `posthoc_sex_bias_wgd.png` | Post-hoc: sex bias in WGD weight estimates |
| `wgd_demographics.png`, `wgd_weight_age.png`, `medeval_pathologies.png` | Dataset descriptives (medeval one sourced from `figs/poster/`) |
| `human_supplement/*.png` | Human-vs-LLM LifeEval comparison: calibration, hard–easy effect, overconfidence by radius and by age |
