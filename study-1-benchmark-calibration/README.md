# LifeEval: Measuring LLM Calibration and the Hard-Easy Effect

This repository evaluates whether large language models (LLMs) say "how sure" they are in a way that matches reality. We study calibration across six benchmarks and introduce **LifeEval**, a new estimation task with ground-truth probabilities derived from actuarial life tables. Across 11 models we observe a consistent **hard-easy effect**: models are overconfident on difficult tasks and underconfident on easy ones. The code and protocol are designed to help you reproduce these findings, extend them to new models, and audit confidence in your own applications.

**Preregistration:** [OSF](https://osf.io/y8rqv/)

---

## Table of Contents

1. [Why Calibration Matters](#why-calibration-matters)
2. [What We Tested](#what-we-tested)
3. [How We Quantify Calibration](#how-we-quantify-calibration)
4. [What We Found](#what-we-found)
5. [Repository Structure](#repository-structure)
6. [Setup](#setup)
7. [Full Workflow](#full-workflow)
   - [Step 1: Retrieve and Format Benchmarks](#step-1-retrieve-and-format-benchmarks)
   - [Step 2: Generate Prompts and Run Models](#step-2-generate-prompts-and-run-models)
   - [Step 3: Parse Raw Results](#step-3-parse-raw-results)
   - [Step 4: Combine and Clean Results](#step-4-combine-and-clean-results)
   - [Step 5: Run Analysis](#step-5-run-analysis)
8. [Data Dictionary](#data-dictionary)
9. [Models Evaluated](#models-evaluated)
10. [Practical Guidance](#practical-guidance)
11. [Limitations and Notes](#limitations-and-notes)

---

## Why Calibration Matters

Accuracy alone is insufficient for safe deployment. A model that is right 60% of the time but **claims** 90% confidence creates operational risk; a model that is right 95% of the time but **claims** 70% confidence leaves performance untapped because users discount correct answers. We evaluate both **first-order confidence** (the probability assigned to the chosen answer) and **second-order confidence** (the decisiveness of the full option distribution) and relate these to observed correctness. This lets you see not just whether a model is right, but whether its probabilities are trustworthy and how that trust varies by task.

---

## What We Tested

We measure calibration on six datasets that span different cognitive demands:

| Dataset | N | Task Type | Description |
|---------|---|-----------|-------------|
| **SciQ** | 1,000 | 4-option MCQ | Science knowledge questions |
| **BoolQ** | 3,270 | True/False | Factual yes/no questions |
| **SAT-EN** | 206 | 4-option MCQ | Passage-based reading comprehension |
| **LSAT-AR** | 230 | 4-5 option MCQ | Multi-step logical/analytical reasoning |
| **HaluEval-QA** | 2,000 | Confidence only | Self-monitoring: rate confidence in a provided answer (1,000 correct + 1,000 hallucinated) |
| **LifeEval** | 808 | Estimation | Predict age at death given current age/gender; scored against U.S. SSA Period Life Tables with tolerance radii r in {1, 5, 10, 20} years |

---

## How We Quantify Calibration

For each question we collect the model's chosen answer and a probability distribution over all options (or, for LifeEval, a probability for each radius). We then compute:

- **Accuracy**: fraction of correct answers.
- **Confidence**: the probability the model assigns to its chosen answer.
- **Expected Calibration Error (ECE)**: the average gap between accuracy and confidence across confidence bins.
- **Overconfidence**: mean(confidence) - accuracy (positive means the model overstates certainty).
- **Second-order confidence (Gini)**: summarizes how sharply the model distinguishes among options.

When token-level probabilities are available, we also compare **stated** probabilities to **token-derived** probabilities to understand how verbalized confidence relates to the model's internal scoring.

---

## What We Found

Across models and tasks, calibration tracks task difficulty.

- On **hard reasoning** tasks (LSAT-AR) and **tight LifeEval radii**, models are **overconfident**. They keep assigning high probabilities even as accuracy falls, which inflates ECE and positive overconfidence.
- On **easy knowledge** and **reading** tasks (SciQ, SAT-EN), models are often **underconfident**. Accuracy is high, yet reported confidence lags behind, yielding negative overconfidence.
- On **self-evaluation** (HaluEval), many models struggle to lower confidence on incorrect or hallucinated content, reflecting weak self-monitoring.
- **Stated vs token probabilities** are broadly aligned, with stated values sometimes slightly better calibrated. This suggests verbalized confidence can capture broader uncertainty than raw next-token scores.
- Confidence values are **"lumpy."** Many models round to coarse steps (e.g., 0.5, 0.6, 1.0), which limits resolution and likely contributes to residual ECE.

These patterns replicate the **hard-easy effect** known from human judgment: overconfidence grows as difficulty rises, while underconfidence emerges when the task becomes trivial.

---

## Repository Structure

```
Comparing-Confidence-in-LLMs/
├── Workflow/                          # Data pipeline scripts
│   ├── Retrieve_Benchmarks.ipynb      # Step 1: Download datasets from HuggingFace
│   ├── DatasetFormatting.ipynb        # Step 1: Additional formatting utilities
│   ├── batch_processing.py           # Step 2: Format prompts + submit batch API jobs
│   ├── get_results_analysis.ipynb    # Step 3: Parse raw API responses
│   ├── LlamaEvaluation.ipynb         # Step 2 (alt): Run Llama models locally
│   └── terminate_instance.py         # Utility: clean up cloud instances
│
├── Formatted Benchmarks/              # Standardized benchmark CSVs (output of Step 1)
│   ├── boolq_valid_formatted.csv
│   ├── halu_eval_qa_formatted.csv
│   ├── life_eval_formatted.csv
│   ├── lsat_ar_test_formatted.csv
│   ├── sat_en_formatted.csv
│   ├── sciq_test_formatted.csv
│   └── PeriodLifeTable_2022_RawData.csv   # SSA life tables for LifeEval scoring
│
├── Prompts/                           # Formatted prompts with system instructions (output of Step 2a)
│   └── {dataset}_prompts.csv
│
├── Batches/                           # API batch request files (output of Step 2b)
│   └── {model_name}/{dataset}_batch.jsonl
│
├── Parsed Results/                    # Per-model per-dataset CSVs (output of Step 3)
│   ├── Claude/{model_name}/{dataset}_{model}.csv
│   ├── Deepseek/{model_name}/{dataset}_{model}.csv
│   ├── Gemini/{model_name}/{dataset}_{model}.csv
│   ├── GPT/{model_name}/{dataset}_{model}.csv
│   └── Llama/{model_name}/{dataset}_{model}.csv
│
├── Combined Results/                  # Aggregated data (output of Step 4)
│   ├── combined_raw.csv               # All parsed results merged and graded
│   ├── combined_clean.csv             # Filtered and normalized for analysis
│   └── llm-confidence-correct.csv     # Confidence-correctness analysis
│
├── Plots/                             # Generated visualizations (output of Step 5)
│   ├── {Dataset}/Calibration Plots/   # Per-model calibration curves
│   ├── Main Plots/                    # Aggregated analysis figures
│   └── Summary Plots/                 # Consolidated comparisons
│
├── R/                                 # R analysis scripts
│   ├── 1process-data.Rmd             # Data processing in R
│   └── 2analyze.Rmd                  # Statistical analysis in R
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

The core dependencies are:

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scipy` | Statistical tests (Pearson correlation) |
| `openai` | OpenAI batch API |
| `anthropic` | Anthropic Messages Batch API |
| `google-genai` | Google Gemini batch API |
| `json5` | Lenient JSON parsing of model responses |
| `datasets` | HuggingFace dataset loading |
| `huggingface_hub` | HuggingFace model access |
| `python-dotenv` | Environment variable loading |

### API keys

Create a `.env` file in the project root with the API keys for whichever providers you want to run:

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

This notebook downloads raw datasets from HuggingFace and formats them into standardized CSVs.

**What it does for each dataset:**

| Dataset | Source | Formatting |
|---------|--------|------------|
| LSAT-AR | `hails/agieval-lsat-ar` | Extracts query, parses choices array into Option A-E columns, converts gold index to letter |
| SciQ | `allenai/sciq` | Shuffles correct answer position among distractors (seed=42), creates 4-option format |
| BoolQ | `google/boolq` (validation split) | Renames columns, drops passage |
| HaluEval | `shunk031/HaluEval` (qa subset) | Samples 1,000 rows (seed=42), retains knowledge/right_answer/hallucinated_answer |
| SAT-EN | `hails/agieval-sat-en` | Extracts passage+question from query field, parses 4 options |
| LifeEval | Generated from SSA life tables | Creates 808 questions: 2 genders x 101 ages x 4 radii; computes ground-truth life expectancy |

**Output format:** CSVs saved to `Formatted Benchmarks/` with standardized columns:

- MCQ datasets: `Question ID, Question, Option A, Option B, Option C, Option D, [Option E], Correct Answer Letter`
- BoolQ: `Question, Correct Answer, Question ID`
- HaluEval: `knowledge, Question, right_answer, hallucinated_answer, Question ID`
- LifeEval: `Question Prompt, Confidence Prompt, True Lifespan, Question ID`

**To run:**

```bash
jupyter notebook Workflow/Retrieve_Benchmarks.ipynb
# Run all cells. Output goes to Formatted Benchmarks/
```

> **Note:** The formatted benchmarks are already included in this repo, so you can skip this step unless you want to regenerate them or modify the datasets.

### Step 2: Generate Prompts and Run Models

**Script:** `Workflow/batch_processing.py`

This script handles the full prompt generation and batch submission pipeline. It reads the formatted benchmarks, wraps each question in a prompt with confidence-elicitation instructions, and submits batch jobs to the OpenAI, Anthropic, and Google APIs.

**Prompt design by dataset:**

| Dataset | Answer Format | Confidence Format |
|---------|--------------|-------------------|
| BoolQ | True/False + reasoning | Single confidence float (0.0-1.0) |
| HaluEval | Confidence only (answer pre-provided) | Single confidence float (0.0-1.0) |
| LifeEval | Integer age prediction + reasoning | Single confidence float (0.0-1.0) for given radius |
| LSAT-AR | Letter answer (A-E) + reasoning | Probability for each option, summing to 1.0 |
| SciQ | Letter answer (A-D) + reasoning | Probability for each option, summing to 1.0 |
| SAT-EN | Letter answer (A-D) + reasoning | Probability for each option, summing to 1.0 |

All prompts request JSON-formatted responses. Temperature is set to 0 for deterministic output (except o3, which requires temperature=1). GPT models request top-5 log probabilities for token-level confidence.

**How to configure which models to run:**

Edit the `models` dictionary near the bottom of `batch_processing.py` (around line 977). Uncomment or add models to each provider's list:

```python
models = {
    'GPT': {
        'class': GPTModels,
        'api_key_name': 'OPENAI_API_KEY',
        'models': ['gpt-4o', 'o3-2025-04-16']  # Add/remove models here
    },
    'Claude': {
        'class': ClaudeModels,
        'api_key_name': 'ANTHROPIC_API_KEY',
        'models': ['claude-3-7-sonnet-20250219']  # Add/remove models here
    },
    'Gemini': {
        'class': GeminiModels,
        'api_key_name': 'GOOGLE_API_KEY',
        'models': ['gemini-2.5-pro']  # Add/remove models here
    }
}
```

**How to configure which datasets to run:**

Edit the `skip_datasets` list (around line 1008) to control which benchmarks are processed:

```python
skip_datasets = [
    # Comment out datasets you WANT to run:
    # 'boolq_valid',
    # 'halu_eval_qa',
    # 'life_eval',
    # 'lsat_ar_test',
    # 'sat_en',
    # 'sciq_test',
]
```

**To run:**

```bash
cd Workflow
python batch_processing.py
```

**What happens:**

1. Reads formatted CSVs from `Formatted Benchmarks/`
2. Applies dataset-specific prompt formatting functions
3. Saves prompts to `Prompts/{dataset}_prompts.csv`
4. Creates API-specific batch files in `Batches/{model_name}/`
5. Uploads and submits batch jobs to each provider's API
6. Records batch job IDs in `results_metadata.json`

Batch jobs run asynchronously. Check status via each provider's API or dashboard. Results must be downloaded separately before Step 3.

**For Llama models:** Use `Workflow/LlamaEvaluation.ipynb` instead, which runs inference locally or on a cloud instance using HuggingFace.

### Step 3: Parse Raw Results

**Script:** `Workflow/get_results_analysis.ipynb`

This notebook takes the raw batch API responses (JSONL files) and extracts structured data from the JSON-formatted model outputs.

**What it does:**

1. **Reads raw batch output files** from each provider (different JSON structures for GPT, Claude, Gemini, Llama, DeepSeek)
2. **Extracts text content** from nested API response structures using `get_content()`
3. **Parses JSON responses** using a two-stage process:
   - `quick_parse()`: regex-based JSON extraction + `json5.loads()` for lenient parsing
   - `parse_response()`: extracts expected fields (Reasoning, Answer, Confidence, A-E probabilities)
4. **Extracts token probabilities** from GPT/Llama logprobs when available
5. **Maps token probabilities to answer options** using `field_probs()`
6. **Saves parsed CSVs** to `Parsed Results/{Model Type}/{Model Name}/{dataset}_{model}.csv`

**Output columns per dataset:**

| Column | BoolQ | HaluEval | LifeEval | MCQ (LSAT/SciQ/SAT) |
|--------|-------|----------|----------|---------------------|
| Question ID | yes | yes | yes | yes |
| Reasoning | yes | - | yes | yes |
| Answer | yes | - | yes | yes |
| Confidence | yes | yes | yes | - |
| A, B, C, D, [E] | - | - | - | yes (stated probs) |
| {option}_prob | - | - | - | yes (token probs) |
| True_prob, False_prob | yes | - | - | - |
| coerce | yes | yes | yes | yes |
| content | yes | yes | yes | yes |

**To run:**

```bash
jupyter notebook Workflow/get_results_analysis.ipynb
# Edit the file paths and model names at the top of the notebook
# Run all cells for each model you want to parse
```

> **Note:** The parsed results for all 11 models are already included in this repo under `Parsed Results/`.

### Step 4: Combine and Clean Results

**Scripts:** `combine.py` then `clean.py` (run from the project root)

#### Step 4a: Combine (`combine.py`)

Merges all per-model per-dataset parsed CSVs into a single file and grades each response against ground truth.

**Grading logic by dataset:**

| Dataset | Scoring Method |
|---------|---------------|
| LSAT-AR, SAT-EN, SciQ | Binary: 1.0 if answer matches correct letter, 0.0 otherwise |
| BoolQ | Binary: 1.0 if answer matches True/False, 0.0 otherwise |
| LifeEval | Probabilistic: P(death in [estimate-R, estimate+R] \| survived to current age) using SSA life tables |
| HaluEval | Binary by suffix: `_r` (real answer) = 1.0, `_h` (hallucinated) = 0.0 |

```bash
python combine.py
# Output: Combined Results/combined_raw.csv
```

#### Step 4b: Clean (`clean.py`)

Applies exclusion criteria and normalization to create the analysis-ready dataset.

**Exclusion criteria applied:**

1. **Incomplete questions:** Removes questions that don't appear for all models (batch job failures)
2. **Unparseable responses:** Removes rows where `Coerce == False` (JSON parsing failed)
3. **Missing MCQ confidences:** Removes MCQ rows where stated confidence sums to 0
4. **Invalid LifeEval confidence:** Removes LifeEval rows with non-numeric confidence values

**Normalization:**

- MCQ stated confidence distributions are normalized to sum to 1.0
- MCQ token probability distributions are normalized to sum to 1.0
- 2AFC (True/False) token probabilities are normalized to sum to 1.0
- Model names are mapped to display names (e.g., `gpt-4o` -> `GPT-4o`)

```bash
python clean.py
# Output: Combined Results/combined_clean.csv
```

**Filtering impact (approximate):**

| Dataset | Original | After Cleaning | Retention |
|---------|----------|----------------|-----------|
| SciQ | 1,000 | ~995 | ~99% |
| BoolQ | 3,270 | ~2,503 | ~77% |
| LifeEval | 808 | ~751 | ~93% |
| HaluEval | 1,999 | ~1,790 | ~90% |
| SAT-EN | 206 | ~173 | ~84% |
| LSAT-AR | 230 | ~86 | ~37% |

LSAT-AR has the most aggressive filtering because many models struggle with the complex JSON format required for 5-option probability distributions.

### Step 5: Run Analysis

**Script:** `analysis.ipynb`

The main analysis notebook. Loads `Combined Results/combined_clean.csv` and produces all plots, tables, and statistical results.

**What it computes:**

| Metric | Description |
|--------|-------------|
| **ECE** | Expected Calibration Error: average gap between accuracy and confidence across 10 bins |
| **Overconfidence** | mean(confidence) - accuracy; positive = overconfident |
| **Gini coefficient** | Measures inequality in probability distribution (second-order confidence) |
| **Pearson r** | Correlation between stated confidence and actual probability (especially for LifeEval) |
| **Regression coefficient** | Slope of confidence ~ ideal probability (sensitivity of stated confidence) |

**Key visualizations generated:**

- Calibration plots (reliability diagrams) for each model x dataset, comparing stated confidence (blue) vs token probability (red)
- ECE bar charts by model and dataset
- Overconfidence strip plots across all model-dataset combinations
- LifeEval-specific: score vs. confidence scatter plots, probability by age line plots, gender differential analysis
- Reasoning vs. non-reasoning model comparisons
- Summary bar charts and LaTeX tables for publication

**To run:**

```bash
jupyter notebook analysis.ipynb
# Run all cells. Plots are saved to Plots/
```

**Additional analysis notebooks:**

- `compare_analysis.ipynb`: Validates methodology by cross-checking results between two researchers
- `R/1process-data.Rmd` -> `R/2analyze.Rmd`: Parallel analysis pipeline in R

---

## Data Dictionary

The combined CSV files (`combined_raw.csv` and `combined_clean.csv`) contain these columns:

```
METADATA
  Question Set (str) -------------- Dataset name (BoolQ, HaluEval, LifeEval, LSAT-AR, SAT-EN, SciQ)
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
  Score (float) ------------------- Correctness score (binary for most; probability for LifeEval)

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

## Practical Guidance

If you plan to gate actions or escalate reviews based on model confidence:

- Treat high confidence on **hard** reasoning tasks with caution. Use thresholds, secondary checks, or require corroborating signals.
- Expect mild **underconfidence** on **easy** tasks; correct answers may deserve more trust than the reported number implies.
- Prefer models or wrappers that expose token probabilities and make both **stated** and **token-derived** confidence auditable.
- For estimation tasks like **LifeEval**, verify that confidence **scales with tolerance**. Flat scaling by radius signals miscalibration.

---

## Limitations and Notes

Some benchmark items contain minor defects (typos, truncated text, missing figures). We keep them intentionally. A calibrated system should express **lower confidence** under ambiguity, and ECE should remain robust to a small fraction of noisy items. Certain APIs restrict decoding settings or hide log-probabilities; in those cases we analyze stated confidence only and document the constraint. For full details on exclusions, hedged responses, and API limitations, see the deviations and constraints notes in the paper.

**Raw batch results** are not included in the GitHub repo due to size. Visit our [OSF site](https://osf.io/y8rqv/) for all raw data.
