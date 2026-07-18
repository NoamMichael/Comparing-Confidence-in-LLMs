"""
Load Study-2 domain results with humans treated as an extra "model".

The four LLMs live in results/<DOMAIN>/<slug>.csv. A human replication is passed
in as a single CSV with the same shape (at minimum question_id, Answer,
Confidence). Ground-truth metadata missing from a human upload is back-filled
from the domain benchmark, so a collaborator can upload just their raw answers.

The returned frame is "long": one row per (model, question), with a `model`
column, numeric `Confidence`, and a `true_probability` outcome column produced by
the shared scoring rules in scoring.py (WGD hit / LifeEval SSA window /
MedEval differential lookup). `overconfidence` = Confidence - true_probability.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scoring import _merge_missing, lifeeval_true_probability, medeval_true_probability

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = ANALYSIS_DIR.parent
RESULTS_PATH = ROOT_DIR / "results"
DOMAINS_PATH = ROOT_DIR / "domains"

HUMAN_MODEL = "human"

MODEL_SLUGS = [
    "anthropic_claude-haiku-4.5",
    "google_gemini-2.5-flash",
    "meta-llama_llama-4-maverick",
    "openai_gpt-5.4-mini",
]

# Benchmark file used to back-fill missing ground-truth columns for a domain.
# MedEval results already carry differential_json/removal_pct inline, so its
# benchmark is only a safety net.
_BENCHMARKS = {
    "WGD": DOMAINS_PATH / "WGD" / "Data" / "benchmark.csv",
    "WGD_SPD": DOMAINS_PATH / "WGD" / "Data" / "benchmark_spd.csv",
    "LifeEval": DOMAINS_PATH / "LifeEval" / "Data" / "benchmark.csv",
    "LifeEval_SPD": DOMAINS_PATH / "LifeEval" / "Data" / "benchmark_spd.csv",
    "MedEval": DOMAINS_PATH / "MedEval" / "Data" / "benchmark_combined.csv",
    "MedEval_SPD": DOMAINS_PATH / "MedEval" / "Data" / "benchmark_combined.csv",
}

# Column used to bucket questions by difficulty in the overconfidence plots.
DIFFICULTY_COL = {
    "WGD": "within_lbs",
    "LifeEval": "radius",
    "MedEval": "removal_pct",
}


def normalize_merge_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse mangled _x/_y columns from prior double-merges.

    Some results files (e.g. LifeEval meta-llama) carry ``min_age_x``/``min_age_y``
    instead of ``min_age``. Rename each ``<col>_x`` to ``<col>`` when the base name
    is missing and drop the redundant ``_y`` duplicates so every file shares one
    clean schema.
    """
    rename = {}
    drop = []
    for col in df.columns:
        if col.endswith("_x") and col[:-2] not in df.columns:
            rename[col] = col[:-2]
        elif col.endswith("_y"):
            drop.append(col)
    if drop:
        df = df.drop(columns=drop)
    if rename:
        df = df.rename(columns=rename)
    return df


def _base_domain(domain: str) -> str:
    """WGD_SPD -> WGD, etc. Used to pick the scorer/difficulty column."""
    return domain[:-4] if domain.endswith("_SPD") else domain


def _score(domain: str, df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Add a `true_probability` outcome column, reproducing analysis.ipynb.

    Deliberately mirrors the notebook that generated the paper's Study-2 tables
    so model rows match published numbers exactly: answers are parsed with plain
    ``pd.to_numeric`` (not scoring.py's more lenient regex extractor), and the
    LifeEval/MedEval ground-truth rules are reused from scoring.py. Missing
    metadata is back-filled from the benchmark for human uploads.
    """
    base = _base_domain(domain)
    if base == "WGD":
        df = _merge_missing(df, benchmark, ["within_lbs", "true_weight"])
        if "photo" in df.columns:
            # 269.jpg is a labeling error present only in the SPD build.
            df = df[df["photo"] != "269.jpg"].reset_index(drop=True)
        answer = pd.to_numeric(df["Answer"], errors="coerce")
        within = df["within_lbs"].astype(float)
        true_w = df["true_weight"].astype(float)
        tp = ((answer - true_w).abs() <= within).astype(float)
        tp[answer.isna()] = np.nan
        df["true_probability"] = tp
    elif base == "LifeEval":
        df = _merge_missing(df, benchmark, ["min_age", "sex", "radius"])
        answer = pd.to_numeric(df["Answer"], errors="coerce")
        df["true_probability"] = [
            lifeeval_true_probability(a, age, sex, r) if isinstance(sex, str) else np.nan
            for a, age, sex, r in zip(answer, df["min_age"], df["sex"], df["radius"])
        ]
    elif base == "MedEval":
        df = _merge_missing(df, benchmark, ["true_pathology", "differential_json"])
        df["true_probability"] = [
            medeval_true_probability(a, d)
            for a, d in zip(df["Answer"], df["differential_json"])
        ]
    else:
        raise KeyError(f"Unknown domain {domain!r}")
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Confidence, drop unscoreable rows, add overconfidence."""
    df = df.copy()
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    df = df.dropna(subset=["Confidence", "true_probability"]).reset_index(drop=True)
    df["overconfidence"] = df["Confidence"] - df["true_probability"]
    return df


def load_domain(
    domain: str,
    human_csv: str | Path | None = None,
    results_root: str | Path = RESULTS_PATH,
    benchmark_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return the scored long frame for one domain (models + optional humans).

    domain: one of WGD, WGD_SPD, LifeEval, LifeEval_SPD, MedEval, MedEval_SPD.
    human_csv: path to a human replication CSV, or None to load models only.
    """
    results_root = Path(results_root)

    bench_path = _BENCHMARKS[domain]
    if benchmark_root is not None:
        bench_path = Path(benchmark_root) / bench_path.name
    benchmark = pd.read_csv(bench_path)

    # Models: metadata already present, so scoring is a no-op merge.
    model_frames = []
    for slug in MODEL_SLUGS:
        df = normalize_merge_columns(pd.read_csv(results_root / domain / f"{slug}.csv"))
        df["model"] = slug
        model_frames.append(df)
    models = pd.concat(model_frames, ignore_index=True)
    frames = [_score(domain, models, benchmark)]

    # Humans: scored on their own so missing metadata is back-filled from the
    # benchmark even when the models already carry that column.
    if human_csv is not None:
        human = normalize_merge_columns(pd.read_csv(human_csv))
        human["model"] = HUMAN_MODEL
        frames.append(_score(domain, human, benchmark))

    long = pd.concat(frames, ignore_index=True)
    return _finalize(long)


def difficulty_column(domain: str) -> str:
    """Metadata column that buckets questions by difficulty for a domain."""
    return DIFFICULTY_COL[_base_domain(domain)]
