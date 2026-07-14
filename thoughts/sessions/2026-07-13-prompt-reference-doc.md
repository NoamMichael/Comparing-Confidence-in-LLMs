# Session 2026-07-13 (b): verbatim DCE/SPD prompt reference

Wrote `study-2-bayeseval/docs/prompts.md` — the exact prompts for all six
question sets (WGD, LifeEval, MedEval × DCE/SPD), reconstructed from the
builder scripts, the committed benchmark CSVs, and the runner.

## What it records that the README summary didn't

- The exact runner-appended DCE wrapper, both variants: standard
  (`results/`) and reasoning (`results_reasoning/`,
  `config_full_reasoning.yaml`). SPD prompts are identical in both runs.
- Fully assembled final user messages per question set (what the model
  actually saw), including the double-format-instruction quirk: WGD and
  MedEval DCE `confidence_prompt`s embed their own JSON instruction *and*
  get the runner wrapper appended.
- No system prompt is ever sent — the "System prompt" sections in
  `domains/*/sample.md` predate the harness and are stale (left as-is,
  flagged in the doc).
- Verified question counts from the committed CSVs: WGD 4,620 / WGD_SPD 928
  (SPD has one extra photo, `269.jpg`), LifeEval 4,040 / 808, MedEval
  768 / 768 (192 patients × 4 removal pcts). eval.py's hardcoded
  per-domain count table (`WGD: 928`, `MedEval: 312`, line ~347) is stale —
  it's only used for cost estimation, not correctness.
- Template oddities preserved verbatim: LifeEval female question template
  is missing the comma after "years".

Linked from the study-2 README Prompts section and `docs/study-2-workflow.md`.
