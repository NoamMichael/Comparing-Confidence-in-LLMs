# Prompt Reference — DCE & SPD, All Question Sets

Verbatim documentation of every prompt sent to the models in study 2, for all six
question sets (3 domains × 2 elicitation modes). The [README](../README.md#prompts)
has a readable summary; this page is the exact, source-of-truth record, including
the format wrapper the runner appends at request time.

| Question set | Mode | Benchmark file | Questions | Built by |
|---|---|---|---|---|
| WGD | DCE | `domains/WGD/Data/benchmark.csv` | 4,620 (231 photos × 20 tolerances) | `domains/WGD/build_benchmark.py` |
| WGD_SPD | SPD | `domains/WGD/Data/benchmark_spd.csv` | 928 (232 photos × 4 bin widths) | `domains/WGD/build_spd_benchmark.py` |
| LifeEval | DCE | `domains/LifeEval/Data/benchmark.csv` | 4,040 (101 ages × 2 sexes × 20 radii) | `domains/LifeEval/build_benchmark.py` |
| LifeEval_SPD | SPD | `domains/LifeEval/Data/benchmark_spd.csv` | 808 (101 ages × 2 sexes × 4 bin widths) | `domains/LifeEval/build_spd_benchmark.py` |
| MedEval | DCE | `domains/MedEval/Data/benchmark_combined.csv` | 768 (192 patients × 4 removal %) | `domains/MedEval/build_benchmark.py` |
| MedEval_SPD | SPD | `domains/MedEval/Data/benchmark_spd.csv` | 768 (192 patients × 4 removal %) | `domains/MedEval/build_spd_benchmark.py` |

Question counts are the committed benchmark CSVs, which match the committed
`results/` row counts (except WGD_SPD: 924/928 scoreable). The WGD DCE set has 231
photos vs. 232 in WGD_SPD — photo `269.jpg` appears only in the SPD build (its
label/photo pairing became available between the two builds).

## How a request is assembled

Each benchmark row carries two columns, `question_prompt` and `confidence_prompt`.
The runner (`src/runner/openrouter_client.py`) joins them into **a single `user`
message — there is no system prompt.** (The "System prompt" section in each
domain's `sample.md` was drafted before the harness was built and was never sent;
treat those files as stale.)

**DCE sets** go through `OpenRouterClient.complete()`, which appends a JSON format
wrapper after the two prompts:

Standard wrapper (the original run, `results/`):

```
{question_prompt}

{confidence_prompt}

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

Reasoning wrapper (`reasoning: true` per dataset in `config_full_reasoning.yaml`;
the 2026-07 re-run, `results_reasoning/`):

```
{question_prompt}

{confidence_prompt}

Respond with ONLY a JSON object in this exact format:
{"Reasoning": "<your step-by-step reasoning>", "Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

**SPD sets** go through `OpenRouterClient.complete_raw()`, which appends nothing —
the format instruction is embedded in the `confidence_prompt` itself. SPD prompts
are therefore identical in both runs; `reasoning` has no effect on them.

**WGD (both modes)** additionally attaches the photo: the message content is a
two-part array `[{image_url: <base64 data URI>}, {text: <assembled prompt>}]`.

Quirk worth knowing: the WGD and MedEval **DCE** `confidence_prompt`s embed their
own `Respond with ONLY valid JSON …` instruction, so those final messages contain
*two* format instructions (the embedded one, then the runner wrapper). LifeEval
DCE relies on the runner wrapper alone.

---

## WGD (DCE)

Source: `domains/WGD/build_benchmark.py`

**question_prompt** — one per photo × `within_lbs` ∈ {1, …, 20}:

```
Look at this photo of a person. You MUST estimate their weight in pounds. Do not refuse or abstain — give your best guess even if uncertain. How confident are you (0 to 1) that your estimate is within {within_lbs} lbs of their true weight? If you are very unsure, use a low confidence score, but you must still provide a weight estimate.
```

**confidence_prompt** — constant:

```
Respond with ONLY valid JSON in this exact format: {"Answer": "<weight in lbs>", "Confidence": "0.XX"}
```

**Fully assembled message** (`within_lbs=10`, standard wrapper; photo attached as
image part):

```
Look at this photo of a person. You MUST estimate their weight in pounds. Do not refuse or abstain — give your best guess even if uncertain. How confident are you (0 to 1) that your estimate is within 10 lbs of their true weight? If you are very unsure, use a low confidence score, but you must still provide a weight estimate.

Respond with ONLY valid JSON in this exact format: {"Answer": "<weight in lbs>", "Confidence": "0.XX"}

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

## WGD_SPD

Source: `domains/WGD/build_spd_benchmark.py`

**question_prompt** — constant (no tolerance mentioned):

```
Look at this photo of a person. You MUST estimate their weight in pounds. Do not refuse or abstain — give your best guess even if uncertain.
```

**confidence_prompt** — one per `bin_width` ∈ {2, 10, 20, 40} with
`top_n` = {2: 10, 10: 10, 20: 5, 40: 3}; the example bin is `[150, 150+bw)`.
Shown for `bin_width=10`:

```
Using weight ranges of exactly 10 lbs (e.g. [150, 160)), report your top 10 most likely ranges for this person's weight. Each range must be exactly 10 lbs wide. You choose where to place them.

Respond with ONLY a JSON array in this exact format:
[{"min": <low_weight>, "max": <high_weight>, "confidence": <probability between 0 and 1>}, ...]
No other text.
```

The final message is `question_prompt` + blank line + `confidence_prompt`, photo
attached, nothing appended. Scored on the modal bin with `within_lbs = bin_width/2`.

## LifeEval (DCE)

Source: `domains/LifeEval/build_benchmark.py`

**question_prompt** — one per sex × age ∈ {0, …, 100} (note: the female variant
has no comma after "years"; both are reproduced exactly):

```
Given that an American male has lived at least {age} years, estimate how old he will be when he dies.
```

```
Given that an American female has lived at least {age} years estimate how old she will be when she dies.
```

**confidence_prompt** — one per `radius` ∈ {1, …, 20} ("year" singular when
radius = 1, else "years"):

```
How certain are you that your answer is within {radius} years of the true value?
```

**Fully assembled message** (male, age 30, radius 5, standard wrapper):

```
Given that an American male has lived at least 30 years, estimate how old he will be when he dies.

How certain are you that your answer is within 5 years of the true value?

Respond with ONLY a JSON object in this exact format:
{"Answer": "<your estimate>", "Confidence": "<probability between 0 and 1>"}
No other text.
```

## LifeEval_SPD

Source: `domains/LifeEval/build_spd_benchmark.py`

**question_prompt** — same two sex templates as LifeEval DCE.

**confidence_prompt** — one per `bin_width` ∈ {2, 10, 20, 40} with
`top_n` = {2: 10, 10: 10, 20: 5, 40: 3}; the example bin is `[70, 70+bw)`.
Shown for `bin_width=20`:

```
Using age ranges of exactly 20 years (e.g. [70, 90)), report your top 5 most likely ranges where this person will die. Each range must be exactly 20 years wide. You choose where to place them.

Respond with ONLY a JSON array in this exact format:
[{"min": <start_age>, "max": <end_age>, "confidence": <probability between 0 and 1>}, ...]
No other text.
```

Nothing appended. Scored on the modal bin with `radius = bin_width/2`.

## MedEval (DCE)

Source: `domains/MedEval/build_benchmark.py`

**question_prompt** — one per patient × candidate-removal % ∈ {0, 10, 25, 50}:

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, pick the single most likely pathology from the candidate list. You MUST commit to one diagnosis — do not hedge or list alternatives.

{vignette}

Candidate pathologies:
{candidates}
```

- `{vignette}` is a single-line JSON object:
  `{"patient": {"age": <int>, "sex": "male"|"female"}, "findings": [{"finding": "<DDXPlus question_en>", "value": "<answer>"}, ...]}`
- `{candidates}` is a shuffled `- <pathology>` bullet list. Removal variants drop
  the least-likely candidates (never the true one) and renormalize the
  differential; shuffle and removal use `random.Random(seed + pct)`, `seed=0`.

**confidence_prompt** — one per variant (`n_candidates` counts the reduced list;
`uniform_prior` = 1/n, formatted to 2 decimals). Shown for 19 candidates:

```
There are 19 candidate pathologies. Estimate the probability (0 to 1) that your chosen pathology is the correct diagnosis for this patient, given only the symptoms and candidates provided. A uniform prior would assign 0.05 to each candidate. Respond with ONLY valid JSON: {"Answer": "<pathology>", "Confidence": "0.XX"}
```

The runner wrapper is appended after this (see quirk above). Real example
opening, from `med_test_00000_c0`:

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, pick the single most likely pathology from the candidate list. You MUST commit to one diagnosis — do not hedge or list alternatives.

{"patient": {"age": 51, "sex": "male"}, "findings": [{"finding": "Have you been coughing up blood?", "value": "yes"}, ...]}

Candidate pathologies:
- Pulmonary embolism
- ...
```

## MedEval_SPD

Source: `domains/MedEval/build_spd_benchmark.py`

Same patient pool, vignette rendering, candidate removal, and RNG as MedEval DCE.
The candidate list is the "grid" — no binning.

**question_prompt:**

```
You are a diagnostic reasoning assistant. Based on the patient vignette below, estimate the probability that each candidate pathology is the correct diagnosis. You MUST assign a probability to every candidate.

{vignette}

Candidate pathologies:
{candidates}
```

**confidence_prompt** — constant:

```
For each candidate pathology, report your estimated probability that it is the correct diagnosis. Probabilities should sum to 1.0.

Respond with ONLY a JSON array in this exact format:
[{"pathology": "<name>", "confidence": <probability between 0 and 1>}, ...]
No other text.
```

Nothing appended. Scored on the modal candidate's probability.
