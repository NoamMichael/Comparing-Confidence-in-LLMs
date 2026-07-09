"""
Unified scoring module for all BayesEval domains.

Each domain exposes a score(results, benchmark) -> DataFrame function that
computes true_probability and brier columns. The dispatch function
get_scorer(domain_name) returns the appropriate scorer.

Domains:
    WGD       — binary hit: |answer - true_weight| <= within_lbs
    LifeEval  — Gompertz conditional survival CDF over [y-r, y+r)
    MedEval   — lookup in DDXPlus differential distribution
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

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
    answer = pd.to_numeric(df["Answer"], errors="coerce")
    conf = pd.to_numeric(df["Confidence"], errors="coerce")
    within = df["within_lbs"].astype(float)
    true_w = df["true_weight"].astype(float)

    df["true_probability"] = ((answer - true_w).abs() <= within).astype(float)
    df.loc[answer.isna(), "true_probability"] = np.nan
    df["brier"] = (conf - df["true_probability"]) ** 2
    return df


# ---------------------------------------------------------------------------
# LifeEval — Gompertz mortality model
# ---------------------------------------------------------------------------

@dataclass
class GompertzParams:
    """Gompertz hazard parameters: h(x) = b * exp(c * x)"""
    b: float
    c: float


def fit_gompertz_to_life_table(life_table_path: str, sex: str = "male") -> GompertzParams:
    """Fit Gompertz hazard via MLE on ages 5-94 of the period life table."""
    df = pd.read_csv(life_table_path)
    col_prefix = "MALE" if sex.lower() == "male" else "FEMALE"
    q_col = f"Death probability ({col_prefix})"

    ages = df["Age"].values.astype(float)
    qx = df[q_col].values.astype(float)

    mask = (ages >= 5) & (ages <= 94)
    fit_ages = ages[mask]
    fit_qx = qx[mask]

    def neg_log_likelihood(params):
        log_b, c = params
        b = np.exp(log_b)
        if c <= 0:
            return 1e12
        exponent = -(b / c) * (np.exp(c * (fit_ages + 1)) - np.exp(c * fit_ages))
        predicted_qx = 1.0 - np.exp(exponent)
        predicted_qx = np.clip(predicted_qx, 1e-15, 1 - 1e-15)
        ll = fit_qx * np.log(predicted_qx) + (1 - fit_qx) * np.log(1 - predicted_qx)
        return -np.sum(ll)

    result = minimize(
        neg_log_likelihood,
        x0=[np.log(1e-5), 0.085],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12},
    )
    return GompertzParams(b=np.exp(result.x[0]), c=result.x[1])


def _conditional_survival(x: float, a: float, params: GompertzParams) -> float:
    """S(x | X >= a) = exp(-(b/c)(exp(cx) - exp(ca)))"""
    b, c = params.b, params.c
    return float(np.exp(-(b / c) * (np.exp(c * x) - np.exp(c * a))))


def _window_probability(y: float, a: float, r: float, params: GompertzParams) -> float:
    """P(death in [y-r, y+r) | survived to a), closed-form via conditional survival CDF."""
    lo = max(y - r, a)
    hi = y + r
    if lo >= hi:
        return 0.0
    return _conditional_survival(lo, a, params) - _conditional_survival(hi, a, params)


_LIFE_TABLE = Path(__file__).resolve().parent.parent / "domains" / "LifeEval" / "Data" / "PeriodLifeTable_2022_RawData.csv"
_GOMPERTZ_PARAMS: dict[str, GompertzParams] | None = None


def _get_gompertz_params() -> dict[str, GompertzParams]:
    global _GOMPERTZ_PARAMS
    if _GOMPERTZ_PARAMS is None:
        _GOMPERTZ_PARAMS = {
            "male": fit_gompertz_to_life_table(str(_LIFE_TABLE), "male"),
            "female": fit_gompertz_to_life_table(str(_LIFE_TABLE), "female"),
        }
    return _GOMPERTZ_PARAMS


def lifeeval_true_probability(answer: float, min_age: float, sex: str, radius: float) -> float:
    """P(death in [answer-r, answer+r) | survived to min_age) via Gompertz."""
    params = _get_gompertz_params()[sex.lower()]
    return _window_probability(answer, min_age, radius, params)


def score_lifeeval(results: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Ground truth from Gompertz conditional survival CDF."""
    df = _merge_missing(results, benchmark, ["min_age", "sex", "radius"])
    answer = pd.to_numeric(df["Answer"], errors="coerce")
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
