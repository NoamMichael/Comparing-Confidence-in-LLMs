# Study 1 Workflow: Benchmark Calibration

End-to-end data flow. Each stage reads the previous stage's output directory; all
paths are relative to `study-1-benchmark-calibration/`. For per-step detail (prompt
wording, parsing rules, metric definitions, data dictionary) see the
[study-1 README](../study-1-benchmark-calibration/README.md).

```
Workflow/Retrieve_Benchmarks.ipynb
        │  downloads from HuggingFace, standardizes columns
        ▼
Formatted Benchmarks/*.csv                (5 benchmarks, gold answers)
        │
Workflow/batch_processing.py              (needs API keys; Llama via LlamaEvaluation.ipynb)
        │  wraps questions in confidence-elicitation prompts,
        │  submits OpenAI/Anthropic/Google batch jobs
        ▼
Prompts/  +  Batches/  +  results_metadata.json   (job IDs)
        │
Workflow/get_results_analysis.ipynb
        │  parses raw JSONL API responses (lenient json5), extracts
        │  answer / stated confidence / token logprobs
        ▼
Parsed Results/<Family>/<model>/*.csv
        │
combine.py                                 python combine.py
        │  walks Parsed Results/, merges gold answers, grades
        ▼
Combined Results/combined_raw.csv          (keeps archived LifeEval rows — preregistered record)
        │
clean.py                                   python clean.py
        │  drops LifeEval, bad QIDs, uncoerced rows; normalizes
        │  confidences to sum 1; maps model IDs to display names
        ▼
Combined Results/combined_clean.csv        (61,017 rows: 5 benchmarks × 11 models)
        │
plots.py                                   python plots.py
        │  one deterministic pass; also writes Plots/summary_stats.csv
        ▼
Plots/                                     (entire tree regenerated)
```

## Regenerating everything from parsed results

```bash
cd study-1-benchmark-calibration
python combine.py && python clean.py && python plots.py
```

No API keys needed from `combine.py` onward. Dependencies: `requirements.txt`
(pandas, numpy, scipy, matplotlib, seaborn).

## What plots.py produces

- `Plots/<Benchmark>/Calibration Plots/` — per-model reliability diagrams
  (stated confidence; `_tokens` variants for GPT-4o and Llama, which expose logprobs)
- `Plots/<Benchmark>/summary_bars_*.png` — 2×2 ECE / overconfidence / accuracy / n
- `Plots/Main Plots/` — aggregate calibration plots (all question sets, MCQ-only,
  2AFC-only, reasoning-vs-chat overlay, full model×benchmark grid)
- `Plots/Summary Plots/` — cross-benchmark ECE/accuracy/overconfidence bars,
  stated-vs-token ECE scatter, per-model token-plot strips
- `Plots/summary_stats.csv` — tidy per-(benchmark, model) table backing the bar plots

## Notebooks

`analysis.ipynb` (root) is the exploratory notebook: ECE/Gini calculations, reliability
diagrams, LaTeX summary tables. It recomputes the combined data itself, so it can run
standalone after `Parsed Results/` exists. `compare_analysis.ipynb` cross-checks
results between researchers, and `R/1process-data.Rmd` → `R/2analyze.Rmd` is a parallel
R pipeline producing its own processed CSV and tables.

Notes for running the notebooks headlessly on Linux: paths are forward-slash and
relative to the study-1 root (run Jupyter from there), and the archived-LifeEval cells
were removed in July 2026 (`scripts/strip_lifeeval_from_notebooks.py` documents that
edit), so a fresh top-to-bottom run needs no archived files.

## Where LifeEval went

`clean.py` drops the LifeEval rows that remain in `combined_raw.csv`; the full LifeEval
record (data, prompts, responses, plots, contamination analysis) lives in
`archive/lifeeval/`. Background: [lifeeval-history.md](lifeeval-history.md).
