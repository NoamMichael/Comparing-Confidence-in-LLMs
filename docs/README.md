# Documentation

Orientation docs for the whole repository. These pages explain how everything fits
together and link to the detailed study-level READMEs rather than duplicating them.

| Doc | What it covers |
|---|---|
| [repo-map.md](repo-map.md) | Annotated directory map: what lives where, what is source vs. generated output, and how the two studies relate |
| [study-1-workflow.md](study-1-workflow.md) | Study 1 end-to-end: benchmark retrieval → prompting/batching → parsing → combine/clean → plots |
| [study-2-workflow.md](study-2-workflow.md) | Study 2 (BayesEval) end-to-end: benchmark building → eval runner → results → scoring/analysis |
| [difficulty-scoring.md](difficulty-scoring.md) | Plain-language explainer of study 2's uniform-guesser difficulty metric, with the per-domain formulas |
| [lifeeval-history.md](lifeeval-history.md) | The LifeEval story across the project: study-1 design, SSA contamination finding, archival, study-2 redesign, human study, and the study-1 vs study-2 comparison |

## Deeper references

- [Root README](../README.md) — project overview and git-history notes for the BayesEval merge
- [Study 1 README](../study-1-benchmark-calibration/README.md) — full pipeline detail, data dictionary, model table, metric definitions
- [Study 2 README](../study-2-bayeseval/README.md) — domains, prompts, models, scoring framework, research questions
- Study 2 per-domain pipelines: [LifeEval](../study-2-bayeseval/docs/pipeline_LifeEval.md) · [MedEval](../study-2-bayeseval/docs/pipeline_MedEval.md) · [WGD](../study-2-bayeseval/docs/pipeline_WGD.md)
- [Human LifeEval study](../study-2-bayeseval/human-data/README.md) — preregistered MTurk supplement
- [Archived LifeEval (study 1)](../study-1-benchmark-calibration/archive/lifeeval/README.md) — the preserved record, including the SSA contamination analysis
