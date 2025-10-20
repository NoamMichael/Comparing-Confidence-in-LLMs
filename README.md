# LifeEval: Measuring LLM Calibration and the Hard–Easy Effect

This repository evaluates whether large language models (LLMs) say “how sure” they are in a way that matches reality. We study calibration across six benchmarks and introduce **LifeEval**, a new estimation task with ground-truth probabilities derived from actuarial life tables. Across 11 models we observe a consistent **hard–easy effect**: models are overconfident on difficult tasks and underconfident on easy ones. The code and protocol are designed to help you reproduce these findings, extend them to new models, and audit confidence in your own applications.

---

## Why calibration matters

Accuracy alone is insufficient for safe deployment. A model that is right 60% of the time but **claims** 90% confidence creates operational risk; a model that is right 95% of the time but **claims** 70% confidence leaves performance untapped because users discount correct answers. We evaluate both **first-order confidence** (the probability assigned to the chosen answer) and **second-order confidence** (the decisiveness of the full option distribution) and relate these to observed correctness. This lets you see not just whether a model is right, but whether its probabilities are trustworthy and how that trust varies by task.

---

## What we tested

We measure calibration on six datasets that span different cognitive demands:

* **SciQ (1,000)** and **BoolQ (3,270)** probe ground-truth knowledge via multiple-choice and true/false questions.
* **SAT-EN (206)** tests contextual understanding with passage-based comprehension.
* **LSAT-AR (230)** stresses multi-step logical reasoning.
* **HaluEval-QA (2,000)** evaluates self-monitoring by asking models to assess the correctness of provided answers, including hallucinated ones.
* **LifeEval (808)** is our new estimation task: given a person’s current age and gender, the model estimates age at death and reports the probability that the true age lies within a tolerance radius ( r \in {1,5,10,20} ) of its guess. We score against **U.S. SSA Period Life Tables**, which provide the true conditional probability of the event. This gives a rare setting where probabilistic forecasts can be judged against known base rates.

---

## How we quantify calibration

For each question we collect the model’s chosen answer and a probability distribution over all options (or, for LifeEval, a probability for each radius). We then compute:

* **Accuracy**: fraction of correct answers.
* **Confidence**: the probability the model assigns to its chosen answer.
* **Expected Calibration Error (ECE)**: the average gap between accuracy and confidence across confidence bins.
* **Overconfidence**: mean(confidence) − accuracy (positive means the model overstates certainty).
* **Second-order confidence (Gini)**: (1 - \sum_k p_k^2), which summarizes how sharply the model distinguishes among options.

When token-level probabilities are available, we also compare **stated** probabilities to **token-derived** probabilities to understand how verbalized confidence relates to the model’s internal scoring.

---

## Methodology at a glance

We evaluate **11 models** spanning proprietary and open-source families. We use one-shot prompts and request JSON formatted outputs with per-option probabilities for MCQ tasks and per-radius probabilities for LifeEval. We set temperature to 0 and use greedy decoding wherever APIs allow. We exclude unparseable or incomplete outputs and, for cross-model comparisons, analyze the **intersection** of questions successfully answered by all models. We **preregistered** our plan before data collection to constrain analytic flexibility (OSF: `https://osf.io/y8rqv/`). Any deviations and API constraints are documented in the paper and mirrored here for transparency.

---

## Data

You can find all of our data in the `Parsed Results` folder. There you will find two files (`combined_raw.csv` & `combined_clean.csv`) with all data combined for easy comparisons across models/datasets. There are many NA values in this dataset as certain fields are not relevent to certain datasets or certain models and so are left blank. The columns of these files are:

```text
Question Set (str) ---------------- Required: The display name of the question set
Question ID (str) ----------------- Required: The Question ID
Model (str) ----------------------- Required: The model that provided the response (e.g. Llama-3.1-8B-Instruct)
Model Type (str) ------------------ Required: The family of models which  the model (e.g. Llama)
Coerce (Bool) --------------------- Required: Whether the parser was able to understand the response

Question (str) -------------------- Required: The question posed to the model
Correct Answer (str) -------------- Optional: Depends on Question Set (LifeEval is different than others)
Content (str) --------------------- Optional: Depends on Coerce value (NA if Coerce == False)
Reasoning (str) ------------------- Optional: Depends on Coerce value (NA if Coerce == False)
Answer (str) ---------------------- Optional: Depends on Coerce value (NA if Coerce == False)
Score (float) --------------------- Optional: Depends on Coerce value (NA if Coerce == False)

Stated Confidence Answer (float) -- Optional: Depends on Question Set (NA if not available)
Stated Confidence A (float) ------- Optional: Depends on Question Set (NA if not available)
Stated Confidence B (float) ------- Optional: Depends on Question Set (NA if not available)
Stated Confidence C (float) ------- Optional: Depends on Question Set (NA if not available)
Stated Confidence D (float) ------- Optional: Depends on Question Set (NA if not available)
Stated Confidence E (float)-------- Optional: Depends on Question Set (NA if not available)

Token Probability Answer (float) -- Optional: Depends on Model Type (NA if not available)
Token Probability A (float) ------- Optional: Depends on Model Type (NA if not available)
Token Probability B (float) ------- Optional: Depends on Model Type (NA if not available)
Token Probability C (float) ------- Optional: Depends on Model Type (NA if not available)
Token Probability D (float) ------- Optional: Depends on Model Type (NA if not available)
Token Probability E (float) ------- Optional: Depends on Model Type (NA if not available)

```

Additionally you will find specific results organized by `Model Type` $\rightarrow $ `Specific Model`  $\rightarrow $ `Question Set`

---

## What we found

Across models and tasks, calibration tracks task difficulty.

* On **hard reasoning** tasks (LSAT-AR) and **tight LifeEval radii**, models are **overconfident**. They keep assigning high probabilities even as accuracy falls, which inflates ECE and positive overconfidence.
* On **easy knowledge** and **reading** tasks (SciQ, SAT-EN), models are often **underconfident**. Accuracy is high, yet reported confidence lags behind, yielding negative overconfidence.
* On **self-evaluation** (HaluEval), many models struggle to lower confidence on incorrect or hallucinated content, reflecting weak self-monitoring.
* **Stated vs token probabilities** are broadly aligned, with stated values sometimes slightly better calibrated. This suggests verbalized confidence can capture broader uncertainty than raw next-token scores.
* Confidence values are **“lumpy.”** Many models round to coarse steps (e.g., 0.5, 0.6, 1.0), which limits resolution and likely contributes to residual ECE.

These patterns replicate the **hard–easy effect** known from human judgment: overconfidence grows as difficulty rises, while underconfidence emerges when the task becomes trivial.

---

## Practical guidance

If you plan to gate actions or escalate reviews based on model confidence:

* Treat high confidence on **hard** reasoning tasks with caution. Use thresholds, secondary checks, or require corroborating signals.
* Expect mild **underconfidence** on **easy** tasks; correct answers may deserve more trust than the reported number implies.
* Prefer models or wrappers that expose token probabilities and make both **stated** and **token-derived** confidence auditable.
* For estimation tasks like **LifeEval**, verify that confidence **scales with tolerance**. Flat scaling by ( r ) signals miscalibration.

---

## Limitations and notes

Some benchmark items contain minor defects (typos, truncated text, missing figures). We keep them intentionally. A calibrated system should express **lower confidence** under ambiguity, and ECE should remain robust to a small fraction of noisy items. Certain APIs restrict decoding settings or hide log-probabilities; in those cases we analyze stated confidence only and document the constraint. For full details on exclusions, hedged responses, and API limitations, see the deviations and constraints notes in the paper.

---

