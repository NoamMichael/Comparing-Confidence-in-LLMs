# Study 2 Workflow: BayesEval

End-to-end data flow; paths relative to `study-2-bayeseval/`. For prompts, models,
scoring definitions, and research questions see the
[study-2 README](../study-2-bayeseval/README.md); the verbatim DCE/SPD prompts for
every question set are in
[study-2-bayeseval/docs/prompts.md](../study-2-bayeseval/docs/prompts.md), and each
domain has a step-by-step pipeline doc in [`docs/`](../study-2-bayeseval/docs/).

```
domains/<Domain>/build_benchmark.py        (DCE: point estimate + scalar confidence)
domains/<Domain>/build_spd_benchmark.py    (SPD: distribution over bins)
        │  computes question grids and analytic metadata
        │  (true probabilities, best answers, MAS)
        ▼
domains/<Domain>/Data/benchmark.csv  +  benchmark_spd.csv
        │
eval.py  +  config.yaml                    (OpenRouter key in .env)
        │  async runner (src/runner/executor.py) queries each model,
        │  parses the JSON response, live dashboard, cost estimator
        ▼
results/<Domain>[_SPD]/<provider>_<model>.csv     (committed: the thesis's evidence)
        │
analysis/analysis.ipynb                    main RQ1–RQ3 analysis
        │  scoring.py     — true_probability + Brier per response,
        │                   Murphy decomposition (BS = REL − RES + UNC)
        │  evaluate_diff.py — difficulty rank percentile
        │                   (see docs/difficulty-scoring.md)
        ▼
analysis/figs/  +  rq*_summary tables
```

## The three domains

| Domain | Task | Ground truth |
|---|---|---|
| WGD | Guess a person's weight from a photo | Binary: within `within_lbs` of measured weight |
| LifeEval | Estimate age at death given survival to `min_age` | Gompertz conditional survival CDF over `[y−r, y+r)`, fitted to the SSA 2022 period life table |
| MedEval | Name the diagnosis from a DDXPlus case | Probability of the answer in the case's differential distribution |

Each domain runs in two elicitation modes: **DCE** (direct confidence estimate — one
point answer plus a scalar confidence) and **SPD** (stated probability distribution —
top-N bins with probabilities; scored on the modal bin).

## Models

Four models via OpenRouter: `anthropic/claude-haiku-4.5`, `google/gemini-2.5-flash`,
`meta-llama/llama-4-maverick`, `openai/gpt-5.4-mini`. Configured in `config.yaml`;
`src/features/estimate_costs.py` estimates spend before a run.

## Analysis entry points

- `analysis/analysis.ipynb` — RQ1 (are models calibrated?), RQ2 (difficulty and the
  hard-easy effect), RQ3 (DCE vs SPD elicitation), post-hoc analyses
- `analysis/human_supplement.ipynb` — humans vs. LLMs on the 88 preregistered human
  LifeEval cells ([human-data/](../study-2-bayeseval/human-data/README.md))
- `analysis/fast_facts.ipynb` — auditable recomputation of every number cited in the
  write-up
- `analysis/sensitivity_ymax.py` — robustness of the LifeEval difficulty metric to
  the `Y_max` parameter

## Re-running

```bash
cd study-2-bayeseval
pip install -r requirements.txt
python eval.py                 # re-query models (costs money; needs .env key)
# analysis only (no API):
jupyter execute analysis/analysis.ipynb analysis/human_supplement.ipynb
```

Analysis is CPU-only and reads the committed `results/`, so it reproduces without any
API access.
