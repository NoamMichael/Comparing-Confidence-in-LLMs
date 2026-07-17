# LifeEval Across the Project

LifeEval — "given that an American {male|female} has lived at least {age} years,
estimate how old {he|she} will be when {he|she} dies, and state your confidence that
you're within {radius} years" — is the thread connecting both studies and the human
experiment. This page tells the story and records the study-1 vs. study-2 comparison.

## 1. Study 1: the original benchmark (2025)

Preregistered ([OSF y8rqv](https://osf.io/y8rqv/)): **808 questions** = 101 ages
(0–100) × 2 sexes × radii {1, 5, 10, 20}, answered by all 11 study-1 models using the
study's "Directions:" reasoning scaffold plus a system prompt. Responses were scored
against the **SSA 2022 Period Life Table** directly: the empirical probability of
death inside `[estimate − r, estimate + r]` conditional on surviving to the given age.
The record lives in `study-1-benchmark-calibration/archive/lifeeval/`.

## 2. The SSA contamination finding

Because the ground truth is a public government table, models may have memorized it.
The analysis in `archive/lifeeval/ssa-contamination/` (contributed by @ddanie1) ran two
stages over 8,261 responses:

1. **Keyword flagging** of the models' reasoning text for SSA/actuarial/life-table
   references — 75.6% flagged, with known over-flagging.
2. **LLM-judge verification** (Claude Sonnet 4) classifying each flagged response as
   no / weak / strong evidence of memorizing specific table values.

Findings (see `archive/lifeeval/ssa-contamination/SSA_APENDIX.md`): some models showed
heavy contamination (strong-evidence rates: DeepSeek-R1 71.5%, Gemini-2.5-Pro 71.0%,
GPT-o3 50.1%), but **overconfidence and the hard–easy effect persisted on the
uncontaminated subset** (4,188 no-evidence responses), so the study's calibration
conclusions are not artifacts of memorization. Recommendation carried forward:
contamination-screen any benchmark built on public tables.

## 3. Archival in study 1

Given the contamination concern, LifeEval was archived out of study 1's active
pipeline: `clean.py` drops its rows at load (the canonical `combined_raw.csv` retains
them as the preregistered record), the notebooks no longer analyze it, and all
LifeEval materials moved to `archive/lifeeval/`.

## 4. Study 2: the redesign

BayesEval rebuilt LifeEval as a proper-scoring domain
(`study-2-bayeseval/domains/LifeEval/`):

- **Grid:** 4,040 DCE questions (101 ages × 2 sexes × radii 1–20 — a superset of study
  1's radii) plus 808 SPD questions (4 bin widths).
- **Prompt:** a single bare user message, no reasoning scaffold; JSON answer
  `{"Answer": …, "Confidence": …}`.
- **Scoring:** originally a Gompertz survival model MLE-fitted to the *same* SSA 2022
  table (smooth analytic `true_probability`); **reverted in July 2026** to study 1's
  empirical table rule — `true_probability` read directly from the table's per-year
  death probabilities, exactly as `combine.py:compute_prob` computed it (the port
  reproduces study-1 scores bit-for-bit on the human data). The two rules agreed at
  r ≈ 0.99; the Gompertz implementation survives in git history. Scored with Brier +
  Murphy decomposition.

## 5. The human study

A preregistered MTurk study (AsPredicted #267677, Jan 2026) put 88 of the same
conditions to 98 human participants; materials, cleaned data, and the human-vs-LLM
analysis live in `study-2-bayeseval/human-data/` and
`study-2-bayeseval/analysis/human_supplement.ipynb`. Humans are overconfident (+0.13)
and show the same hard–easy slope as the models.

## 6. Study 1 vs. study 2: are the responses interchangeable?

Assessed July 2026 (structural comparison; the empirical deep-dive was deliberately
set aside — see §7):

| Dimension | Study 1 (archived) | Study 2 (BayesEval) |
|---|---|---|
| Question stem | "Given that an American {sex} has lived at least {age} years…" | **identical** |
| Confidence prompt | "How certain that your answer is within {r} years…" + Directions scaffold + system prompt | "How certain are you that your answer is within {r} year(s)…", bare message |
| Questions | 808 (radii {1,5,10,20}) | 4,040 DCE (radii 1–20, superset) + 808 SPD |
| Source table | SSA 2022 period life table | **same file** |
| Ground truth | Empirical table probability | **same rule** (reverted from a Gompertz fit, July 2026) |
| Response format | Free text with a `Reasoning` field | JSON only — **no reasoning text** |
| Models | 11 (2024–25 frontier) | 4 (late-2025, via OpenRouter) |
| Model overlap | — | **only `gemini-2.5-flash`** |

**Verdict: not drop-in replaceable.** The two runs share the task, the source data,
and (since the July 2026 revert) the scoring rule, but only one model appears in both
rosters and the prompt harness differs (scaffolded reasoning vs. bare JSON). Study-2 responses can't
substitute for study-1's LifeEval rows in any per-model analysis; at most the
gemini-2.5-flash cells are comparable across studies, confounded by the prompt change.

**Contamination blocker:** study 1's SSA screen operated on the models' *reasoning
text*. Study-2 responses contain none (the `raw` column is just the JSON answer), so
that screen **cannot run** on study 2 as collected.

## 7. Open question (as of 2026-07-10)

Whether study 2's LifeEval results need a contamination assessment is unresolved, and
the empirical study-1-vs-study-2 comparison is **set aside for now**. If it becomes
necessary, the options are:

- **Re-run study 2's LifeEval with reasoning capture** (chain-of-thought or a
  "explain your answer" field), then reuse the study-1 keyword + LLM-judge pipeline.
- **Answer-level screen without re-running** (no API needed): match study-2 point
  estimates against SSA-table-derived values (e.g. age + e(age), rounded variants)
  and compare the exact/near-match rate to a permutation baseline; a spike at zero in
  |answer − table value| would signal memorization. Weaker evidence than the
  reasoning-based screen.

None of this has been executed. If study-2 LifeEval is re-run, prefer capturing
reasoning so the original screen applies.
