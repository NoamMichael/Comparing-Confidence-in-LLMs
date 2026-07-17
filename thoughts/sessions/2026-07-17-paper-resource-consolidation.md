# 2026-07-17 — Consolidate paper resources into paper/

Branch: `paper-resource-consolidation` (off merge-studies).

New `paper/` folder collecting copies of all publication-ready resources, built by
`paper/scripts/retrieve_paper_resources.py` (idempotent; never touches sources):

- `tex/` — 5 LaTeX tables: study-2 `rq1_summary_table.txt` / `sensitivity_ymax_table.txt`
  renamed to .tex; study-1 reasoning + chat summary tables extracted from
  `analysis.ipynb` cell outputs (marker-based, picks the complete variant — the
  captioned reasoning-table output in the notebook has corrupted LSAT/SAT/SciQ rows);
  Table 1 converted from `R/analysis/table1-similar-layout.csv` via pandas.
- `plots/study1/` — 9 aggregate figures (Main Plots + Summary Plots families).
- `plots/study2/` — 12 headline figures + `human_supplement/` (4).

Selection rule: aggregate/summary only; excluded per-model×qset cal plots, per-benchmark
summary bars, poster copies (except `medeval_pathologies.png`, which exists only there),
and `per_cell_confidence.png` (per Noam).

Update: LifeEval rows are now stripped from all three study-1 tables by the script.

## Study-1 analysis.ipynb re-run (same session)

The notebook couldn't run top-to-bottom (cleaning moved to clean.py; `combined_clean`
was a ghost) and its saved outputs mixed old kernel sessions — the captioned reasoning
summary table had corrupted MCQ rows (all-NaN "Stated Confidence Answer (MCQ)" in that
old run; `stack()` silently dropped the Confidence rows). Fixes: cell 15 now loads
`Combined Results/combined_clean.csv`, cell 28 no longer re-writes it; per Noam, the
gini expectation was removed (sum_df has no Gini/GiniECE; "Gini ECE Plots" section
deleted). Re-executed with `--allow-errors`: 5 cells of the dead LifeEval-era
"Additional Analysis" block error honestly (ghost `grouped`/`bar_data`/`has_tokens`,
missing `MISC/fin_le_data.csv`) — candidates for deletion, Noam's call. Fresh table
outputs are byte-identical to the previously extracted paper/tex versions.
`combined_raw.csv` restored after the run (notebook rewrites it without LifeEval rows).
