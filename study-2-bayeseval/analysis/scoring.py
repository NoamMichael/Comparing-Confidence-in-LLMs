"""
Unified scoring module for all BayesEval domains.

Each domain exposes a score(results, benchmark) -> DataFrame function that
computes true_probability and brier columns. The dispatch function
get_scorer(domain_name) returns the appropriate scorer.

Domains:
    WGD       — binary hit: |answer - true_weight| <= within_lbs
    LifeEval  — empirical SSA life-table window probability (study-1 rule)
    MedEval   — lookup in DDXPlus differential distribution
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+\.?\d*")


def _numeric_answer(s: pd.Series) -> pd.Series:
    """Like pd.to_numeric but tolerates units and prose ("81 years old",
    "185 lbs", "98-100" -> 98). Plain to_numeric silently dropped ~48% of
    some models' answers from mean Brier."""
    def _extract(v):
        if pd.isna(v):
            return np.nan
        m = _NUM_RE.search(str(v))
        return float(m.group(0)) if m else np.nan
    return s.map(_extract)


def _merge_missing(results: pd.DataFrame, benchmark: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Merge only benchmark columns not already present in results."""
    missing = [c for c in cols if c not in results.columns]
    if not missing:
        return results.copy()
    return results.merge(
        benchmark[["question_id"] + missing],
        on="question_id",
        how="left",
    )


def murphy_decomposition(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """BS = Reliability - Resolution + Uncertainty."""
    conf = pd.to_numeric(df["Confidence"], errors="coerce").to_numpy()
    p = df["true_probability"].to_numpy(dtype=float)
    mask = ~(np.isnan(conf) | np.isnan(p))
    conf, p = conf[mask], p[mask]
    if conf.size == 0:
        return {"reliability": np.nan, "resolution": np.nan,
                "uncertainty": np.nan, "brier": np.nan}
    p_bar = p.mean()
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    reliability = resolution = 0.0
    for b in range(n_bins):
        idx = bins == b
        if not idx.any():
            continue
        w = idx.sum() / conf.size
        c_b = conf[idx].mean()
        p_b = p[idx].mean()
        reliability += w * (c_b - p_b) ** 2
        resolution += w * (p_b - p_bar) ** 2
    uncertainty = float((p * (1 - p)).mean())
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": uncertainty,
        "brier": float(((conf - p) ** 2).mean()),
    }


# ---------------------------------------------------------------------------
# WGD — Weight Guessing Dataset
# ---------------------------------------------------------------------------

def score_wgd(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Ground truth is deterministic: 1.0 if within tolerance, else 0.0."""
    df = _merge_missing(results, benchmark, ["within_lbs", "true_weight"])
    if "photo" in df.columns:
        # Drop photo 269.jpg: present only in the SPD build (labeling error), not in the DCE set
        df = df[df["photo"] != "269.jpg"].reset_index(drop=True)
    answer = _numeric_answer(df["Answer"])
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    within = df["within_lbs"].astype(float)
    true_w = df["true_weight"].astype(float)

    df["true_probability"] = ((answer - true_w).abs() <= within).astype(float)
    df.loc[answer.isna(), "true_probability"] = np.nan
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


# ---------------------------------------------------------------------------
# LifeEval — empirical SSA life-table rule (study-1 preregistered rule)
# ---------------------------------------------------------------------------

_LIFE_TABLE = Path(__file__).resolve().parent.parent / "domains" / "LifeEval" / "Data" / "PeriodLifeTable_2022_RawData.csv"
_LIFE_TABLE_QX: dict[str, tuple[np.ndarray, int]] | None = None  # sex -> (q_x, table_min)
_DEATH_MASS: dict[tuple[str, int], np.ndarray] = {}              # (sex, m) -> d array


def _get_life_table_qx() -> dict[str, tuple[np.ndarray, int]]:
    """Parse and cache per-sex death probabilities q_x from the SSA life table."""
    global _LIFE_TABLE_QX
    if _LIFE_TABLE_QX is None:
        df = pd.read_csv(_LIFE_TABLE)
        out = {}
        for sex in ("male", "female"):
            col = f"Death probability ({sex.upper()})"
            tab = df[["Age", col]].dropna().sort_values("Age")
            ages = tab["Age"].astype(int).to_numpy()
            q = tab[col].astype(float).to_numpy()
            if not np.array_equal(ages, np.arange(ages[0], ages[-1] + 1)):
                raise ValueError("life table ages not contiguous")
            out[sex] = (q, int(ages[0]))
        _LIFE_TABLE_QX = out
    return _LIFE_TABLE_QX


def _death_mass(sex: str, min_age: int) -> tuple[np.ndarray, int, int]:
    """d[x-m] = P(die in [x, x+1) | survived to m) = S_rel(x)*q_x for x in [m, table_max]."""
    q, table_min = _get_life_table_qx()[sex]
    table_max = table_min + len(q) - 1
    m = max(int(min_age), table_min)
    key = (sex, m)
    if key not in _DEATH_MASS:
        qs = q[m - table_min:]
        s_rel = np.concatenate(([1.0], np.cumprod(1.0 - qs[:-1])))
        _DEATH_MASS[key] = s_rel * qs
    return _DEATH_MASS[key], m, table_max


def lifeeval_true_probability(answer: float, min_age: float, sex: str, radius: float) -> float:
    """P(death in integer-age window [floor(answer-r), ceil(answer+r)) | survived
    to min_age), read directly from the SSA 2022 period life table."""
    if answer is None or not np.isfinite(answer) or not np.isfinite(min_age) or not np.isfinite(radius):
        return np.nan
    d, m, table_max = _death_mass(sex.strip().lower(), int(min_age))
    lo = max(int(np.floor(answer - radius)), m)
    hi = min(int(np.ceil(answer + radius)), table_max + 1)
    if hi <= lo:
        return 0.0
    return float(min(max(d[lo - m: hi - m].sum(), 0.0), 1.0))


def lifeeval_best_answer_and_mas(min_age: int, sex: str, radius: float) -> tuple[int, float]:
    """Discrete argmax of the empirical window probability over integer answers
    y in [min_age, table_max]; smallest y wins ties (study-1 convention)."""
    d, m, table_max = _death_mass(sex.strip().lower(), int(min_age))
    ys = np.arange(m, table_max + 1)
    probs = np.array([lifeeval_true_probability(float(y), min_age, sex, radius) for y in ys])
    i = int(np.argmax(probs))
    return int(ys[i]), float(probs[i])


def score_lifeeval(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Ground truth from the empirical SSA life-table rule."""
    df = _merge_missing(results, benchmark, ["min_age", "sex", "radius"])
    answer = _numeric_answer(df["Answer"])
    df["true_probability"] = [
        lifeeval_true_probability(a, age, sex, r)
        for a, age, sex, r in zip(answer, df["min_age"], df["sex"], df["radius"])
    ]
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


# ---------------------------------------------------------------------------
# MedEval — DDXPlus differential diagnosis
# ---------------------------------------------------------------------------

def _normalize_pathology(s: str) -> str:
    return "".join(c.lower() for c in str(s) if not c.isspace())


def medeval_true_probability(answer: str, differential_json: str) -> float:
    """Lookup answer in the DDXPlus differential distribution."""
    try:
        differential = json.loads(differential_json)
    except (TypeError, json.JSONDecodeError):
        return 0.0
    norm = _normalize_pathology(answer)
    for pathology, prob in differential:
        if _normalize_pathology(pathology) == norm:
            return float(prob)
    return 0.0


def score_medeval(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Ground truth from DDXPlus differential probability distribution."""
    df = _merge_missing(results, benchmark, ["true_pathology", "differential_json"])
    df["true_probability"] = [
        medeval_true_probability(a, d)
        for a, d in zip(df["Answer"], df["differential_json"])
    ]
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SCORERS = {
    "WGD": score_wgd,
    "LifeEval": score_lifeeval,
    "MedEval": score_medeval,
    "WGD_SPD": score_wgd,
    "LifeEval_SPD": score_lifeeval,
    "MedEval_SPD": score_medeval,
}


def get_scorer(domain: str):
    """Return the score(results, benchmark) function for a domain."""
    if domain not in _SCORERS:
        raise KeyError(f"Unknown domain {domain!r}. Available: {list(_SCORERS)}")
    return _SCORERS[domain]
