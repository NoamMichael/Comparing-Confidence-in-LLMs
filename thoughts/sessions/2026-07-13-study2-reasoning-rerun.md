# Session 2026-07-10 → 07-13: study-2 reasoning re-run, scoring fix

Sessions spanning the full study-2 re-run with reasoning capture (commits
`c80c596`, `ff11fe3`, `e7f6d8b`; also `7099693` fixing plots.py CWD paths).

## Why

`docs/lifeeval-history.md` §7: study-2 responses were JSON-only, so study-1's
SSA contamination screen (keyword flag + LLM judge on reasoning text) couldn't
run. Re-ran all of study 2 with a `Reasoning` field on DCE prompts.

## What happened

1. **Cost estimates** (live OpenRouter pricing 2026-07-10): full re-run with
   reasoning ~$51 projected; frontier roster (gemini-3.1-pro, gpt-5.5,
   sonnet-5, deepseek-r1) would be ~$870, GPT-5.5 alone ~$730. Stayed with the
   original 4 models. `estimate_costs.py` pricing table refreshed.
2. **Pilot** (400 calls, ~$0.60, `results_pilot_reasoning/`): measured output
   lengths (gemini ~940 tok on LifeEval, gpt-5.4-mini ~150) and caught two
   parser bugs before the full run.
3. **Full run** (47,728 calls, ~$60, `results_reasoning/`): three error waves,
   all diagnosed — (a) json.loads rejecting literal newlines in Reasoning
   strings → strict=False; (b) Haiku markdown fences + unescaped inch marks
   (5'10") → fence stripping + regex fallback; (c) **Parasail** serving
   llama-4-maverick image calls with empty content (tokens billed!) →
   `ignore_providers` config. Final: 47,671/47,728 scoreable; 52 residual =
   gemini stably returning Reasoning-only JSON on certain LifeEval questions.
4. **Scoring bug found**: `pd.to_numeric(Answer)` silently dropped answers
   like "81 years old" from mean Brier — 48% of Haiku's LifeEval rows in the
   ORIGINAL run. Fixed with regex extraction; both runs' `summary.csv`
   regenerated with an `n_scored` coverage column.

## Findings

- Reasoning effect (paired per-question, SPD sets as null control — no
  significant SPD drift): **llama LifeEval Brier 0.123→0.043** (conf
  0.82→0.72), gpt/haiku modest improvements, **gemini worse everywhere**
  (LifeEval 0.083→0.134, conf 0.77→0.90 — reasoning-induced overconfidence).
- Keyword-stage contamination screen on LifeEval reasoning: llama 97%,
  gemini 91%, haiku 59%, gpt-5.4-mini 37% mention SSA/actuarial/life tables.
  **LLM-judge stage still to run** (study-1 pipeline in
  `study-1-benchmark-calibration/archive/lifeeval/`).

## Loose ends

- LLM-judge contamination stage on `results_reasoning/LifeEval/*.csv`.
- 52 gemini Reasoning-only LifeEval rows (retry more or document).
- `domains/WGD/Data/Photos-encoded` is a local symlink to ~/git/BayesEval
  (now gitignored); WGD runs need it.
- OpenRouter key lives in repo-root `.env` (old study-2 `.env` deleted —
  its key was revoked).
