# Study 1 — Measuring LLM Calibration and the Hard-Easy Effect

This study evaluates whether large language models (LLMs) say "how sure" they are in a way that matches reality. We study calibration across five benchmarks spanning different cognitive demands. Across 11 models we observe a consistent **hard-easy effect**: models are overconfident on difficult tasks and underconfident on easy ones. The code and protocol are designed to help you reproduce these findings, extend them to new models, and audit confidence in your own applications.

**Preregistration:** [OSF](https://osf.io/y8rqv/)

> **Looking for LifeEval?** This study originally introduced **LifeEval**, an estimation task with ground-truth probabilities derived from SSA actuarial life tables. Its complete preregistered record — data, results, plots, and an SSA-contamination analysis — is preserved in [`archive/lifeeval/`](archive/lifeeval/), and the benchmark is actively developed in [Study 2 (BayesEval)](../study-2-bayeseval/).

---

## Table of Contents

1. [Why Calibration Matters](#why-calibration-matters)
2. [What We Tested](#what-we-tested)
3. [How We Quantify Calibration](#how-we-quantify-calibration)
4. [What We Found](#what-we-found)
5. [Repository Structure](#repository-structure)
6. [Setup](#setup)
7. [Full Workflow](#full-workflow)
8. [Data Dictionary](#data-dictionary)
9. [Models Evaluated](#models-evaluated)
10. [Archived: LifeEval](#archived-lifeeval)
11. [Practical Guidance](#practical-guidance)
12. [Limitations and Notes](#limitations-and-notes)

---

## Why Calibration Matters

Accuracy alone is insufficient for safe deployment. A model that is right 60% of the time but **claims** 90% confidence creates operational risk; a model that is right 95% of the time but **claims** 70% confidence leaves performance untapped because users discount correct answers. We evaluate both **first-order confidence** (the probability assigned to the chosen answer) and **second-order confidence** (the decisiveness of the full option distribution) and relate these to observed correctness. This lets you see not just whether a model is right, but whether its probabilities are trustworthy and how that trust varies by task.

---

## What We Tested

We measure calibration on five datasets that span different cognitive demands:

| Dataset | N | Task Type | Description |
|---------|---|-----------|-------------|
| **SciQ** | 1,000 | 4-option MCQ | Science knowledge questions |
| **BoolQ** | 3,270 | True/False | Factual yes/no questions |
| **SAT-EN** | 206 | 4-option MCQ | Passage-based reading comprehension |
| **LSAT-AR** | 230 | 4-5 option MCQ | Multi-step logical/analytical reasoning |
| **HaluEval-QA** | 2,000 | Confidence only | Self-monitoring: rate confidence in a provided answer (1,000 correct + 1,000 hallucinated) |

A sixth benchmark, **LifeEval** (808 estimation questions scored against U.S. SSA Period Life Tables), was part of the original preregistered study; see [`archive/lifeeval/`](archive/lifeeval/) for its full record and [Study 2](../study-2-bayeseval/) for its successor.

---

## How We Quantify Calibration

For each question we collect the model's chosen answer and a probability distribution over all options. We then compute:

- **Accuracy**: fraction of correct answers.
- **Confidence**: the probability the model assigns to its chosen answer.
- **Expected Calibration Error (ECE)**: the average gap between accuracy and confidence across confidence bins.
- **Overconfidence**: mean(confidence) - accuracy (positive means the model overstates certainty).
- **Second-order confidence (Gini)**: summarizes how sharply the model distinguishes among options.

When token-level probabilities are available, we also compare **stated** probabilities to **token-derived** probabilities to understand how verbalized confidence relates to the model's internal scoring.

---

## What We Found

Across models and tasks, calibration tracks task difficulty.

- On **hard reasoning** tasks (LSAT-AR), models are **overconfident**. They keep assigning high probabilities even as accuracy falls, which inflates ECE and positive overconfidence.
- On **easy knowledge** and **reading** tasks (SciQ, SAT-EN), models are often **underconfident**. Accuracy is high, yet reported confidence lags behind, yielding negative overconfidence.
- On **self-evaluation** (HaluEval), many models struggle to lower confidence on incorrect or hallucinated content, reflecting weak self-monitoring.
- **Stated vs token probabilities** are broadly aligned, with stated values sometimes slightly better calibrated. This suggests verbalized confidence can capture broader uncertainty than raw next-token scores.
- Confidence values are **"lumpy."** Many models round to coarse steps (e.g., 0.5, 0.6, 1.0), which limits resolution and likely contributes to residual ECE.

These patterns replicate the **hard-easy effect** known from human judgment: overconfidence grows as difficulty rises, while underconfidence emerges when the task becomes trivial.

---

## Repository Structure

```
study-1-benchmark-calibration/
├── Workflow/                          # Data pipeline scripts
│   ├── Retrieve_Benchmarks.ipynb      # Step 1: Download datasets from HuggingFace
│   ├── DatasetFormatting.ipynb        # Step 1: Additional formatting utilities
│   ├── batch_processing.py            # Step 2: Format prompts + submit batch API jobs
│   ├── get_results_analysis.ipynb     # Step 3: Parse raw API responses
│   ├── LlamaEvaluation.ipynb          # Step 2 (alt): Run Llama models locally
│   └── terminate_instance.py          # Utility: clean up cloud instances
│
├── Formatted Benchmarks/              # Standardized benchmark CSVs (output of Step 1)
├── Prompts/                           # Formatted prompts with system instructions
├── Batches/                           # API batch request files (gitignored)
├── Parsed Results/                    # Per-model per-dataset CSVs (output of Step 3)
├── Combined Results/                  # Aggregated data (output of Step 4)
│   ├── combined_raw.csv               # All parsed results merged and graded
│   ├── combined_clean.csv             # Filtered and normalized for analysis
│   └── llm-confidence-correct.csv     # Confidence-correctness analysis
├── Plots/                             # Generated visualizations (output of Step 5)
├── R/                                 # Parallel analysis pipeline in R
│   ├── 1process-data.Rmd
│   └── 2analyze.Rmd
│
├── archive/lifeeval/                  # Preserved LifeEval record (see its README)
│   └── ssa-contamination/             # SSA table-memorization analysis (@ddanie1)
│
├── combine.py                         # Step 4a: Merge all parsed results + grade answers
├── clean.py                           # Step 4b: Apply exclusion criteria + normalize
├── analysis.ipynb                     # Step 5: Main analysis notebook (plots, ECE, tables)
├── compare_analysis.ipynb             # Validation: cross-check between researchers
├── results_metadata.json              # Batch job IDs for API result retrieval
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab
- R and RStudio (optional, for R-based analysis in `R/`)

### Install dependencies

```bash
pip install -r requirements.txt
```

### API keys

Create a `.env` file in this directory with the API keys for whichever providers you want to run:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

You only need keys for the providers whose models you plan to evaluate.

---

## Full Workflow

The pipeline has five stages. Each stage reads from the previous stage's output directory. If you only want to analyze our existing results, skip to **Step 5**.

```
HuggingFace Datasets
    │  Step 1: Retrieve_Benchmarks.ipynb
    ▼
Formatted Benchmarks/  (standardized CSVs)
    │  Step 2: batch_processing.py
    ▼
Prompts/ + Batches/  (formatted prompts → API batch requests → submitted jobs)
    │  Step 3: get_results_analysis.ipynb
    ▼
Parsed Results/  (per-model per-dataset CSVs)
    │  Step 4: combine.py → clean.py
    ▼
Combined Results/  (combined_raw.csv → combined_clean.csv)
    │  Step 5: analysis.ipynb
    ▼
Plots/ + summary tables + LaTeX output
```

### Step 1: Retrieve and Format Benchmarks

**Script:** `Workflow/Retrieve_Benchmarks.ipynb`

Downloads raw datasets from HuggingFace and formats them into standardized CSVs.

| Dataset | Source | Formatting |
|---------|--------|------------|
| LSAT-AR | `hails/agieval-lsat-ar` | Extracts query, parses choices array into Option A-E columns, converts gold index to letter |
| SciQ | `allenai/sciq` | Shuffles correct answer position among distractors (seed=42), creates 4-option format |
| BoolQ | `google/boolq` (validation split) | Renames columns, drops passage |
| HaluEval | `shunk031/HaluEval` (qa subset) | Samples 1,000 rows (seed=42), retains knowledge/right_answer/hallucinated_answer |
| SAT-EN | `hails/agieval-sat-en` | Extracts passage+question from query field, parses 4 options |

**Output:** CSVs saved to `Formatted Benchmarks/` with standardized columns.

> **Note:** The formatted benchmarks are already included in this repo, so you can skip this step unless you want to regenerate them or modify the datasets.

### Step 2: Generate Prompts and Run Models

**Script:** `Workflow/batch_processing.py`

Reads the formatted benchmarks, wraps each question in a prompt with confidence-elicitation instructions, and submits batch jobs to the OpenAI, Anthropic, and Google APIs.

| Dataset | Answer Format | Confidence Format |
|---------|--------------|-------------------|
| BoolQ | True/False + reasoning | Single confidence float (0.0-1.0) |
| HaluEval | Confidence only (answer pre-provided) | Single confidence float (0.0-1.0) |
| LSAT-AR | Letter answer (A-E) + reasoning | Probability for each option, summing to 1.0 |
| SciQ | Letter answer (A-D) + reasoning | Probability for each option, summing to 1.0 |
| SAT-EN | Letter answer (A-D) + reasoning | Probability for each option, summing to 1.0 |

All prompts request JSON-formatted responses. Temperature is set to 0 for deterministic output (except o3, which requires temperature=1). GPT models request top-5 log probabilities for token-level confidence.

Configure which models to run via the `models` dictionary and which datasets via the `skip_datasets` list near the bottom of the script.

```bash
cd Workflow
python batch_processing.py
```

Batch jobs run asynchronously; job IDs are recorded in `results_metadata.json`. **For Llama models:** use `Workflow/LlamaEvaluation.ipynb` instead, which runs inference locally or on a cloud instance.

### Step 3: Parse Raw Results

**Script:** `Workflow/get_results_analysis.ipynb`

Takes the raw batch API responses (JSONL) and extracts structured data: text content from nested API structures, lenient JSON parsing (`json5`), token probabilities where available. Saves per-model CSVs to `Parsed Results/{Model Type}/{Model Name}/`.

> **Note:** The parsed results for all 11 models are already included under `Parsed Results/`.

### Step 4: Combine and Clean Results

**Scripts:** `combine.py` then `clean.py` (run from this directory)

`combine.py` merges all per-model per-dataset parsed CSVs and grades each response:

| Dataset | Scoring Method |
|---------|---------------|
| LSAT-AR, SAT-EN, SciQ | Binary: 1.0 if answer matches correct letter, 0.0 otherwise |
| BoolQ | Binary: 1.0 if answer matches True/False, 0.0 otherwise |
| HaluEval | Binary by suffix: `_r` (real answer) = 1.0, `_h` (hallucinated) = 0.0 |

`clean.py` applies exclusion criteria (incomplete questions, unparseable responses, zero-sum MCQ confidences) and normalizes confidence distributions to sum to 1.0.

```bash
python combine.py   # -> Combined Results/combined_raw.csv
python clean.py     # -> Combined Results/combined_clean.csv
```

> **Note on LifeEval rows:** the canonical `Combined Results/*.csv` shipped in this repo still contain LifeEval rows — they are the preregistered record. `clean.py` and the analysis notebooks drop those rows at load time; re-running the pipeline from `Parsed Results/` produces LifeEval-free CSVs. See [`archive/lifeeval/`](archive/lifeeval/).

### Step 5: Run Analysis

**Script:** `analysis.ipynb`

Loads the combined results and produces all plots, tables, and statistical results: ECE, overconfidence, Gini coefficients, calibration (reliability) diagrams comparing stated confidence vs token probability, reasoning vs non-reasoning model comparisons, and LaTeX summary tables. Plots are saved to `Plots/`.

**Additional analysis:**

- `compare_analysis.ipynb`: validates methodology by cross-checking results between two researchers
- `R/1process-data.Rmd` -> `R/2analyze.Rmd`: parallel analysis pipeline in R

---

## Data Dictionary

The combined CSV files (`combined_raw.csv` and `combined_clean.csv`) contain these columns:

```
METADATA
  Question Set (str) -------------- Dataset name (BoolQ, HaluEval, LSAT-AR, SAT-EN, SciQ; canonical
                                    files also contain archived LifeEval rows)
  Question ID (str) --------------- Unique question identifier
  Model (str) --------------------- Model name (e.g., GPT-4o, Claude-Sonnet-4)
  Model Type (str) ---------------- Model family (GPT, Claude, Gemini, Llama, Deepseek)
  Coerce (bool) ------------------- Whether the JSON response was successfully parsed

RESPONSE
  Question (str) ------------------ The question posed to the model
  Correct Answer (str) ------------ Ground truth (format varies by dataset)
  Content (str) ------------------- Raw model response text
  Reasoning (str) ----------------- Extracted reasoning (NA if Coerce=False)
  Answer (str) -------------------- Extracted answer (NA if Coerce=False)
  Score (float) ------------------- Correctness score

STATED CONFIDENCE
  Stated Confidence Answer (float)  Confidence in chosen answer
  Stated Confidence A-E (float) --- Per-option confidence (MCQ datasets only)

TOKEN PROBABILITIES
  Token Probability Answer (float)  Token probability for answer (models with logprobs only)
  Token Probability A-E (float) --- Per-option token probability (MCQ, models with logprobs only)
  Token Probability True (float) -- Token probability for True (BoolQ, models with logprobs only)
  Token Probability False (float) - Token probability for False (BoolQ, models with logprobs only)
```

Many columns contain NA values because certain fields only apply to specific datasets or models (e.g., token probabilities are unavailable for reasoning models like o3 and DeepSeek-R1).

---

## Models Evaluated

| Model | Family | Type | Token Probs Available |
|-------|--------|------|----------------------|
| GPT-4o | GPT | General | Yes (top-5) |
| GPT-o3 | GPT | Reasoning | No |
| Claude Sonnet 3.7 | Claude | Reasoning | No |
| Claude Sonnet 4 | Claude | Reasoning | No |
| Claude Haiku 3 | Claude | General | No |
| Gemini 2.5 Pro | Gemini | Reasoning | No |
| Gemini 2.5 Flash | Gemini | General | No |
| DeepSeek-R1 | DeepSeek | Reasoning | No |
| DeepSeek-V3 | DeepSeek | General | No |
| Llama 3.1 70B Instruct | Llama | General | Yes |
| Llama 3.1 8B Instruct | Llama | General | Yes |

---

## Archived: LifeEval

LifeEval asked models to predict age at death given a person's current age and gender, scored as P(death within ±r years | survived to current age) against the SSA 2022 Period Life Tables (808 questions; r ∈ {1, 5, 10, 20}). It was the estimation-task arm of the preregistered study and is the direct ancestor of Study 2's expanded LifeEval domain (4,040 questions, 20 radii, proper scoring rules).

Preserved in [`archive/lifeeval/`](archive/lifeeval/):

- benchmark data, prompts, batch files, and parsed model responses
- all LifeEval plots and summary tables
- the **SSA contamination analysis** (contributed by [@ddanie1](https://github.com/ddanie1)): a keyword flag + LLM-judge pipeline testing whether models' reasoning shows evidence of memorized life tables, and how the calibration findings hold up when contaminated responses are excluded

---

## Practical Guidance

If you plan to gate actions or escalate reviews based on model confidence:

- Treat high confidence on **hard** reasoning tasks with caution. Use thresholds, secondary checks, or require corroborating signals.
- Expect mild **underconfidence** on **easy** tasks; correct answers may deserve more trust than the reported number implies.
- Prefer models or wrappers that expose token probabilities and make both **stated** and **token-derived** confidence auditable.

---

## Limitations and Notes

Some benchmark items contain minor defects (typos, truncated text, missing figures). We keep them intentionally. A calibrated system should express **lower confidence** under ambiguity, and ECE should remain robust to a small fraction of noisy items. Certain APIs restrict decoding settings or hide log-probabilities; in those cases we analyze stated confidence only and document the constraint. For full details on exclusions, hedged responses, and API limitations, see the deviations and constraints notes in the paper.

**Raw batch results** are not included in the GitHub repo due to size. Visit our [OSF site](https://osf.io/y8rqv/) for all raw data.
