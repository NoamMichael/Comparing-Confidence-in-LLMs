# Comparing Confidence in LLMs

This repository holds two related studies of **LLM calibration** — whether the
confidence a language model reports matches how often it is actually right —
conducted at UC Berkeley (URAP calibration project and an honors thesis that
grew out of it).

## Study 1 — Benchmark Calibration (`study-1-benchmark-calibration/`)

Measures first- and second-order confidence across five benchmarks (SciQ,
BoolQ, SAT-EN, LSAT-AR, HaluEval-QA) for 11 models, and finds a consistent
**hard–easy effect**: models are overconfident on hard tasks and underconfident
on easy ones. Preregistered on [OSF](https://osf.io/y8rqv/).

Study 1 originally introduced **LifeEval**, an estimation task scored against
SSA period life tables. LifeEval's preregistered record — data, results, plots,
and an SSA-contamination analysis contributed by
[@ddanie1](https://github.com/ddanie1) — is preserved in
[`study-1-benchmark-calibration/archive/lifeeval/`](study-1-benchmark-calibration/archive/lifeeval/),
and the benchmark is now actively developed in Study 2.

## Study 2 — BayesEval (`study-2-bayeseval/`)

An honors thesis benchmark suite evaluating calibration on **Bayesian inference
tasks with continuous ground-truth probabilities**, scored with strictly proper
rules (Brier score + Murphy decomposition). Three domains:

- **LifeEval** — actuarial mortality estimation (successor of Study 1's task,
  expanded to 4,040 questions across 20 tolerance radii)
- **WGD** — weight estimation from photos
- **MedEval** — differential diagnosis (DDXPlus)

Two elicitation modes are compared: direct confidence (DCE) and stated
probability distributions (SPD).

## Repository map

```
├── docs/                            # Cross-study orientation: repo map, workflows,
│                                    # difficulty-scoring explainer, LifeEval history
├── thoughts/sessions/               # Dated session notes (plans, results, next steps)
├── study-1-benchmark-calibration/   # Study 1: pipeline, results, analysis, R scripts
│   └── archive/lifeeval/            # preserved LifeEval record + SSA contamination analysis
└── study-2-bayeseval/               # Study 2: runner, domains, analysis, human data
```

Each study is self-contained with its own README, `requirements.txt`, and
`.gitignore` — see the study READMEs for setup and full workflows. New to the repo?
Start with [`docs/README.md`](docs/README.md).

## Git history notes

- Study 2 was merged in from the standalone
  [BayesEval](https://github.com/NoamMichael/BayesEval) repository with its full
  history (second parent of the merge commit). If `git log --follow` on a
  `study-2-bayeseval/` file does not cross the merge boundary, use
  `git log <merge-commit>^2 -- <original path>`.
- The SSA contamination analysis was contributed by
  [@ddanie1](https://github.com/ddanie1) and merged from their fork.
