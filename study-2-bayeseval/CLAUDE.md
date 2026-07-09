# CLAUDE.md — BayesEval

## Project Overview

BayesEval is a multi-domain benchmark suite for evaluating LLM calibration on Bayesian inference tasks. Each domain provides questions with analytically computable ground-truth probabilities, scored using the Brier Score (a strictly proper scoring rule).

## Repository Structure

- `analysis/scoring.py` — Unified scoring module for all domains (Brier, Murphy decomposition)
- `analysis/analysis.ipynb` — Calibration analysis and plots
- `eval.py` — Cross-domain evaluation runner (config-driven)
- `domains/` — Domain-specific benchmarks and data (WGD, LifeEval, MedEval)
- `thoughts/` — Research notes, experiment logs, and design decisions
- `requirements.txt` — Python dependencies

## Key Concepts

- **Brier Score**: `(confidence - true_probability)^2` — primary evaluation metric
- **Proper scoring rule**: Model minimizes expected score only by reporting true beliefs
- **Murphy decomposition**: BS = Reliability - Resolution + Uncertainty
- **MAS (Maximum Achievable Score)**: Theoretical ceiling for a given question's true probability

## Conventions

- Domain benchmarks produce CSVs with columns: `question_prompt`, `confidence_prompt`, `true_probability`, `question_id`, plus domain-specific metadata
- Scoring uses strictly proper scoring rules throughout — never use accuracy-only metrics for calibration evaluation
- Gold responses are JSON: `{"Answer": "...", "Confidence": "0.XX"}`
- Results files are git-ignored; benchmark definitions are tracked

## Development Notes

- Python 3.10+
- GPU evaluation uses vLLM for batched inference
- Fine-tuning uses LoRA via HuggingFace TRL (DPO, SFT)
- Local analysis is CPU-only (pandas, scipy, matplotlib)
