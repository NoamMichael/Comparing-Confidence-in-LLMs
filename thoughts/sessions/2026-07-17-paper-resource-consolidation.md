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

Caveats noted in `paper/README.md`: study-1 notebook tables and Table 1 still contain
LifeEval rows (pre-drop); trim when placing in the paper.
