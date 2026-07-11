# Session 2026-07-10: docs, LifeEval-free plots, cross-study note, human data

Branch: `merge-studies` (worktree). Five commits: `3a0ef05`, `295df80`, `d069de2`,
`d845ac7`, plus this note. Nothing pushed to origin; local `main` untouched.

## What was asked

1. A `docs/` folder explaining the repo layout, workflows, and processes (e.g. study-2
   difficulty scoring).
2. Study-1 plots regenerated without LifeEval — amended mid-session to: **a `plots.py`
   script that generates everything in one go** instead of notebook re-execution.
3. Compare study-1 vs study-2 LifeEval responses (replaceable?) — amended mid-session
   to: **set the empirical work aside, just write the note**; study-2 LifeEval may
   need re-running.
4. Add the human MTurk data as a supplement to the study-2 analysis.
5. Create `thoughts/sessions/` and leave this note.

## What was done

### 1. Notebook cleanup (`3a0ef05`)

- `scripts/strip_lifeeval_from_notebooks.py` (one-shot, idempotent, verifies by AST):
  deleted the 38 archived-LifeEval cells from `analysis.ipynb` (176 → 138) — including
  two that hard-errored on files now in `archive/lifeeval/` — and the dead LifeEval
  branches/helpers in `Workflow/get_results_analysis.ipynb`.
- Kept `clean.py`'s and the notebook's `!= "LifeEval"` filter-at-load as guards.
- Normalized Windows backslash path literals to forward slashes in both notebooks
  (they were unrunnable on Linux).
- Gotcha preserved for the record: the rounding cell defined `percent_rounded`/
  `cc_r`/`cc_c` used by the later summary tables — it was rewritten, not deleted.

### 2. plots.py + regenerated tree (`295df80`)

- New `study-1-benchmark-calibration/plots.py`: single deterministic pass over
  `Combined Results/combined_clean.csv` → the full `Plots/` tree (85 files: per-model
  reliability diagrams + `_tokens` variants, per-benchmark 2×2 summary bars, aggregate
  Main Plots, cross-benchmark Summary Plots) + `Plots/summary_stats.csv`.
- Re-ran `combine.py` + `clean.py`. `combined_raw.csv` was **restored to HEAD** after
  the re-run: it intentionally keeps the archived LifeEval rows as the preregistered
  record (clean.py filters at load). `combined_clean.csv` is now LifeEval-free
  (61,017 rows; it previously still carried 8,261 LifeEval rows).
- Deleted the old 120-file Plots tree wholesale: it mixed several naming generations,
  and all nine `Main Plots/*.png` had been blank 2,398-byte files since commit
  `a32e4c4` — the old code called `plt.savefig` *after* `plt.show()`, saving a closed
  figure. Two of them were LifeEval-derived.
- Sanity numbers (stated confidence): mean ECE by benchmark — SciQ 0.067, BoolQ 0.113,
  SAT-EN 0.147, HaluEval 0.181, LSAT-AR 0.300; LSAT-AR most overconfident (+0.26),
  SciQ/SAT-EN slightly underconfident.

### 3. Human data supplement (`d069de2`)

- `study-2-bayeseval/human-data/`: raw Qualtrics export + codebook + original R
  scripts + prereg PDF (AsPredicted #267677) + the canonical cleaned data
  (980 obs × 98 participants; copied from `~/git/LifeEval_Mturk_Analysis`, which also
  fills the role of the missing `260113LifeEvalQuestions.csv` — the raw→clean R step
  is not re-runnable without it, documented in the README). MTurk worker IDs were
  never exported, so the raw file is anonymous. Item attributes verify exactly
  against the study-1 QID scheme.
- `analysis/human_supplement.ipynb` (executed, outputs committed): scores humans with
  study-2's Gompertz rule (r = 0.991 vs. study-1's empirical rule → rule choice
  immaterial) and compares them to the 4 DCE models on the matched 88 cells.
- Results: humans overconfident +0.127 and *inside* the LLM range (GPT-5.4 Mini −0.04,
  Claude Haiku 4.5 +0.03, Gemini 2.5 Flash +0.15, Llama 4 Maverick +0.18); everyone
  overconfident at r=1 and underconfident at r=20; hard-easy slopes all positive
  (humans 0.60, models 0.38–0.70); humans have the worst Brier (0.132).
- Known data wart: `results/LifeEval/meta-llama_llama-4-maverick.csv` carries stale
  `_x`/`_y` merge-suffix columns; the notebook joins attributes from `benchmark.csv`
  on `question_id` instead of trusting result-file columns.

### 4. docs/ (`d845ac7`)

`docs/{README,repo-map,study-1-workflow,study-2-workflow,difficulty-scoring,lifeeval-history}.md`
— orientation + links, no duplication of study READMEs. `lifeeval-history.md` carries
the cross-study comparison (task 3's deliverable). Root README points to `docs/`;
study-2 README gained a Human Supplement section and structure fixes (results/ is
committed, thoughts/ is gitignored).

## Cross-study LifeEval verdict (task 3, set aside per Noam)

**Not drop-in replaceable.** Same stem, same SSA 2022 table, but only
`gemini-2.5-flash` appears in both model rosters, study 1 used a Directions scaffold +
system prompt vs study 2's bare JSON message, and scoring differs (empirical table vs
Gompertz). Full table in `docs/lifeeval-history.md` §6.

**Blocker:** study-2 responses have no reasoning text (`raw` = the JSON answer), so
study-1's SSA keyword + LLM-judge contamination screen cannot run on them.

## Next steps / open items

- **Decide whether to re-run study-2 LifeEval with reasoning capture** so the
  contamination screen applies; the no-API fallback (answer-level exact-match vs SSA
  values with a permutation baseline) is specced in `docs/lifeeval-history.md` §7 but
  not executed.
- Merge `merge-studies` → `main` (fast-forward) and push when ready; then delete the
  `ddanie1`/`bayeseval` remotes (deferred from the merge session).
- `MISC/LE_human_data/` in the old main checkout is now redundant (copied into
  `study-2-bayeseval/human-data/`) — delete after the merge lands.
- Consider nbstripout to keep notebook outputs from going stale (analysis.ipynb still
  carries pre-strip outputs in untouched cells; a fresh top-to-bottom run would clear
  them).
- Archive NoamMichael/BayesEval on GitHub with a pointer (deferred from merge session).
- `study-2-bayeseval/thoughts/` is gitignored, so `measuring_difficulty.md` (the
  difficulty-metric spec) exists only locally — consider tracking it or moving its
  content into `docs/`.
