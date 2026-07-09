"""
Difficulty metric for BayesEval questions.

Computes the rank-percentile difficulty score defined in
thoughts/measuring_difficulty.md: the expected reward of a uniform
baseline guesser, converted to a within-domain rank percentile
(higher = harder).
"""

import json

import numpy as np
import pandas as pd
from scipy.special import exp1
from scipy.stats import rankdata

from scoring import _get_gompertz_params


def _eu_wgd(df: pd.DataFrame) -> np.ndarray:
    w_lo = df["true_weight"].min() - df["within_lbs"].max()
    w_hi = df["true_weight"].max() + df["within_lbs"].max()
    return (2 * df["within_lbs"].astype(float) / (w_hi - w_lo)).to_numpy()


def _eu_lifeeval(df: pd.DataFrame) -> np.ndarray:
    y_max = df["min_age"].max() + df["radius"].max()
    params_map = _get_gompertz_params()
    results = np.empty(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        a = float(row["min_age"])
        r = float(row["radius"])
        sex = row["sex"].lower()
        p = params_map[sex]
        alpha, beta = p.b, p.c
        A = (alpha / beta) * np.exp(beta * a)
        Z = r + (np.exp(A) / beta) * (exp1(A) - exp1(A * np.exp(beta * r)))
        results[i] = Z / (y_max - a) if y_max > a else 0.0
    return results


def _eu_medeval(df: pd.DataFrame) -> np.ndarray:
    def _n_candidates(dj):
        try:
            return len(json.loads(dj))
        except (TypeError, json.JSONDecodeError):
            return np.nan
    n = df["differential_json"].apply(_n_candidates).to_numpy(dtype=float)
    return 1.0 / n


_DOMAIN_FNS = {
    "WGD": _eu_wgd,
    "LifeEval": _eu_lifeeval,
    "MedEval": _eu_medeval,
}


def add_difficulty(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Add a 'diff' column with rank-percentile difficulty (higher = harder)."""
    if domain not in _DOMAIN_FNS:
        raise KeyError(f"Unknown domain {domain!r}. Available: {list(_DOMAIN_FNS)}")
    df = df.copy()
    raw = _DOMAIN_FNS[domain](df)
    df["diff"] = rankdata(-raw, method="average") / len(raw)
    return df
