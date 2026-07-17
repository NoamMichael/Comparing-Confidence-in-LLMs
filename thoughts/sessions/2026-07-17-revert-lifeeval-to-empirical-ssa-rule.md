# 2026-07-17 — Revert LifeEval scoring: Gompertz → empirical SSA rule

Co-author decision: drop the Gompertz-fitted scoring rule and revert to study 1's
original empirical SSA life-table rule (`combine.py:compute_prob`).

## What changed

- `analysis/scoring.py` — Gompertz block (fit, CDF, params cache) replaced by
  `_get_life_table_qx` / `_death_mass` / empirical `lifeeval_true_probability`
  (same signature) + new `lifeeval_best_answer_and_mas` (discrete argmax over
  integer ages, smallest age wins ties — study-1 convention).
- `domains/LifeEval/build_benchmark.py` — uses the shared argmax; `benchmark.csv`
  rebuilt (prompts/question_ids byte-identical; only best_answer/MAS/gold_response
  changed; best_answer is now an integer). `benchmark_spd.csv` untouched.
- `analysis/evaluate_diff.py` `_eu_lifeeval` and `analysis/sensitivity_ymax.py` —
  closed-form Gompertz Z (exp1) replaced by numeric mean/cumsum of empirical window
  probabilities over integer guesses; EU convention = mean over count (y_max − a + 1).
- Notebooks re-executed (analysis, fast_facts, human_supplement); docs updated
  (lifeeval-history, study-2-workflow, difficulty-scoring, repo-map, READMEs,
  pipeline_LifeEval, explainer, sample.md).

## Verification

- New rule vs old `compute_prob` (main checkout): bit-for-bit on 39,660 grid points.
- Benchmark diff: 4,040 rows, identical prompts/qids; MAS mean Δ +0.0027 (max 0.064,
  worst at ages 94–98 r∈{1,2} where the Gompertz fit—ages 5–94—was weakest).
- Gompertz vs empirical true_probability on actual model answers: Pearson ≥ 0.9956
  on every model × {DCE, SPD}; mean |Δ| ≈ 0.017–0.02.
- All RQ directions unchanged (see below).

## Finding: `fin_le_data.csv` Score sex bug (study-1 era)

Study 1's `score_life_eval` assigned sex by row position (`female iff index >= 404`).
Fine for qid-sorted 808-row LLM files; wrong for the 980-row human file → `Score`
is computed with the wrong sex on **498/980 rows**. Proof: the ported rule reproduces
`Score` to 1e-16 on all 980 rows once that assignment is replicated (human_supplement
cell 5). All study-2 human analyses use the declared sex (correct — matches the qid
scheme exactly). Flag for anything study-1-side that consumed `fin_le_data.csv`
(`study-1-benchmark-calibration/analysis.ipynb`, `MISC/LE_human_data/` on old main).
The old "Gompertz vs empirical r = 0.991" comparison was polluted by this bug; true
rule-vs-rule agreement on correct inputs is r = 0.9976 (human answers, declared sex).

## Direction check (Gompertz → empirical)

- RQ1 LifeEval: overconf signs unchanged (Haiku +0.029→+0.038, Gemini +0.116→+0.120,
  Llama +0.174→+0.185, GPT −0.042→−0.033); β₁ all positive*** (0.73–0.80 → 0.75–0.82);
  ECE ordering unchanged.
- RQ3 LifeEval ΔECE: Haiku SPD-worse*** both; Gemini/Llama SPD-better, now stronger
  (−11.6% p=.051 → −18.1% p=.001; −12.9% p=.004 → −17.8% p<.001); GPT n.s. both but
  sign flips (−11.9% p=.19 → +5.6% p=.55) — only directional change anywhere, n.s.
- Humans: overconf +0.127→+0.136; slopes ~same (human 0.60→0.61); Brier: humans no
  longer uniquely worst — now tied with Llama (0.1366 vs 0.1373).
- sensitivity_ymax: ρ curve nearly identical; conclusion (ranking insensitive) holds.

Not committed yet — worktree also carries earlier uncommitted changes (269.jpg
exclusion etc.); baseline diffs vs HEAD conflate the two for non-LifeEval rows.
