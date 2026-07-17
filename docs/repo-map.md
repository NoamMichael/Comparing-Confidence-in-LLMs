# Repository Map

This repository is a two-study monorepo on LLM confidence and calibration, run by the
same research group (Don Moore's lab, UC Berkeley URAP):

- **Study 1 — Benchmark Calibration** (`study-1-benchmark-calibration/`): do LLMs know
  what they know? Eleven models state answers *and* confidence on five public
  benchmarks; we measure calibration (ECE, overconfidence, Gini).
- **Study 2 — BayesEval** (`study-2-bayeseval/`): an honors-thesis follow-up that
  builds benchmarks with *analytically computable* ground-truth probabilities
  (weight-guessing photos, actuarial life expectancy, differential diagnosis), scored
  with strictly proper rules (Brier + Murphy decomposition).

The bridge between them is **LifeEval** (life-expectancy estimation): it began as study
1's sixth benchmark, was archived there after an SSA-table contamination analysis, and
was redesigned as a study-2 domain. See [lifeeval-history.md](lifeeval-history.md).

## Top level

```
├── README.md                        Project overview + git-history notes for the merge
├── docs/                            ← you are here (cross-study orientation)
├── thoughts/sessions/               Dated session notes (plans, results, next steps)
├── study-1-benchmark-calibration/
└── study-2-bayeseval/
```

## Study 1 (`study-1-benchmark-calibration/`)

Source of truth for each stage is the directory the previous stage writes into
(details: [study-1-workflow.md](study-1-workflow.md)).

```
├── README.md                  Thorough study README: metrics, data dictionary, models
├── Workflow/                  Pipeline notebooks/scripts (retrieve → batch → parse)
│   ├── Retrieve_Benchmarks.ipynb   Downloads/format benchmarks from HuggingFace
│   ├── DatasetFormatting.ipynb     Formatting helpers
│   ├── batch_processing.py         Builds prompts, submits API batch jobs (needs keys)
│   ├── get_results_analysis.ipynb  Parses raw API responses → Parsed Results/
│   ├── LlamaEvaluation.ipynb       Llama models run separately (local/cloud GPU)
│   └── terminate_instance.py
├── Formatted Benchmarks/      INPUT  standardized benchmark CSVs (5 benchmarks)
├── Prompts/                   GENERATED  per-benchmark prompt CSVs
├── Batches/                   GENERATED  API batch request files (payloads gitignored)
├── Parsed Results/            GENERATED  per-family/model parsed response CSVs
├── Combined Results/          GENERATED  combined_raw.csv (keeps archived LifeEval rows
│                              as the preregistered record), combined_clean.csv (filtered)
├── Plots/                     GENERATED  entire tree is rebuilt by plots.py
├── combine.py                 Parsed Results → combined_raw.csv (merges gold, grades)
├── clean.py                   combined_raw → combined_clean (drops LifeEval, exclusions,
│                              normalization, display names)
├── plots.py                   combined_clean → the full Plots/ tree in one pass
├── analysis.ipynb             Exploratory analysis notebook (ECE, overconfidence, tables)
├── compare_analysis.ipynb     Cross-researcher validation
├── R/                         Parallel R pipeline (1process-data.Rmd → 2analyze.Rmd)
├── Calibration Comparison Data/  External comparison data
├── MISC/                      Small shared tables (table1-info.csv, …)
├── scripts/                   One-shot maintenance scripts
├── results_metadata.json      Batch job IDs per model/benchmark
└── archive/lifeeval/          The archived LifeEval record: data, prompts, batches,
                               parsed results, plots, R outputs, and the SSA
                               contamination analysis (ssa-contamination/)
```

**Benchmarks (5):** SciQ, BoolQ, SAT-EN, LSAT-AR, HaluEval-QA.
**Models (11):** GPT-4o, GPT-o3, Claude Sonnet 3.7/4, Claude Haiku 3, Gemini 2.5
Pro/Flash, DeepSeek-R1/V3, Llama-3.1-70B/8B.

## Study 2 (`study-2-bayeseval/`)

Details: [study-2-workflow.md](study-2-workflow.md) and the
[study-2 README](../study-2-bayeseval/README.md).

```
├── README.md                  Domains, prompts, models, scoring, research questions
├── CLAUDE.md                  Working conventions for the study
├── eval.py + config.yaml      Config-driven async evaluation runner (OpenRouter)
├── src/runner/                Executor (DCE + SPD modes), client, live dashboard
├── src/features/              Cost estimator
├── domains/<name>/            build_benchmark.py, build_spd_benchmark.py, Data/
│   ├── WGD/                   Weight guessing from photos (photos stored externally)
│   ├── LifeEval/              Actuarial life expectancy (4,040 DCE + 808 SPD questions)
│   └── MedEval/               Differential diagnosis from DDXPlus
├── results/<Domain>[_SPD]/    COMMITTED  raw model outputs (the thesis's evidence)
├── analysis/
│   ├── scoring.py             Unified scoring: Brier, Murphy, empirical SSA life-table rule
│   ├── evaluate_diff.py       Difficulty percentile (see difficulty-scoring.md)
│   ├── analysis.ipynb         Main RQ1–3 analysis
│   ├── human_supplement.ipynb Human-vs-LLM LifeEval comparison
│   ├── sensitivity_ymax.py    Robustness of difficulty to the Y_max parameter
│   └── figs/                  Generated figures
├── human-data/                Preregistered human LifeEval study (AsPredicted #267677)
├── docs/                      Per-domain pipeline walkthroughs
└── thoughts/                  Research notes (gitignored)
```

**Domains (3 × 2 modes):** WGD, LifeEval, MedEval — each in DCE (point estimate +
scalar confidence) and SPD (distribution over bins) mode.
**Models (4, via OpenRouter):** Claude Haiku 4.5, Gemini 2.5 Flash, Llama 4 Maverick,
GPT-5.4 Mini.

## What is generated vs. source

Regenerate rather than hand-edit anything under: study 1's `Prompts/`, `Batches/`,
`Parsed Results/`, `Combined Results/`, `Plots/`; study 2's `analysis/figs/`. Model
outputs (`Parsed Results/`, `results/`) are generated but *committed*, since they are
the studies' primary evidence and cost real money to reproduce.

## Git history notes

Study 2 was merged in from the standalone BayesEval repo with full history; the root
[README](../README.md#git-history) documents how to browse across the merge boundary.
