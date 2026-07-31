"""
Difficulty metric for BayesEval questions.

Two quantities per question:

- reward E(R): the expected reward of a *uniform baseline guesser*
  (higher = easier). Each domain defines its own baseline; all three
  return a probability in [0, 1].
- raw difficulty ``1 - E(R)`` (higher = harder), also in [0, 1] — the
  probability the uniform baseline gets the question wrong.

``add_difficulty`` exposes both, plus ``diff``, the within-domain rank
percentile of the raw difficulty (higher = harder). See
thoughts/measuring_difficulty.md.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scoring import lifeeval_true_probability


def _reward_wgd(df: pd.DataFrame) -> np.ndarray:
    """P(uniform-random weight guess lands within ±within_lbs)."""
    w_lo = df["true_weight"].min() - df["within_lbs"].max()
    w_hi = df["true_weight"].max() + df["within_lbs"].max()
    return (2 * df["within_lbs"].astype(float) / (w_hi - w_lo)).to_numpy()


def _reward_lifeeval(df: pd.DataFrame) -> np.ndarray:
    """Mean empirical window probability over integer guesses y in [a, y_max]."""
    y_max = int(df["min_age"].max() + df["radius"].max())
    cache: dict[tuple[int, str, float], float] = {}
    results = np.empty(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        a = int(row["min_age"])
        r = float(row["radius"])
        sex = row["sex"].lower()
        key = (a, sex, r)
        if key not in cache:
            w = [lifeeval_true_probability(float(y), a, sex, r) for y in range(a, y_max + 1)]
            cache[key] = float(np.mean(w)) if w else 0.0
        results[i] = cache[key]
    return results


def _reward_medeval(df: pd.DataFrame) -> np.ndarray:
    """Uniform prior over candidate pathologies: 1 / n_candidates."""
    def _n_candidates(dj):
        try:
            return len(json.loads(dj))
        except (TypeError, json.JSONDecodeError):
            return np.nan
    n = df["differential_json"].apply(_n_candidates).to_numpy(dtype=float)
    return 1.0 / n


_REWARD_FNS = {
    "WGD": _reward_wgd,
    "LifeEval": _reward_lifeeval,
    "MedEval": _reward_medeval,
}


def add_difficulty(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Add difficulty columns for a domain.

    - ``reward``: expected reward E(R) of a uniform baseline guesser (higher = easier).
    - ``diff_raw``: raw difficulty ``1 - E(R)`` (higher = harder), in [0, 1].
    - ``diff``: within-domain rank percentile of ``diff_raw`` (higher = harder).

    The percentile ranks ascending, so higher raw difficulty maps to a higher
    percentile — identical to the old ``rankdata(-reward)`` since ``1 - reward``
    is a monotone transform of ``-reward``.
    """
    if domain not in _REWARD_FNS:
        raise KeyError(f"Unknown domain {domain!r}. Available: {list(_REWARD_FNS)}")
    df = df.copy()
    reward = _REWARD_FNS[domain](df)
    df["reward"] = reward
    df["diff_raw"] = 1.0 - reward
    df["diff"] = rankdata(df["diff_raw"].to_numpy(), method="average") / len(df)
    return df
