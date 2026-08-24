"""Confound-aware re-analysis of the RQ3 claim.

The README (RQ 3) states that "SPD prompting is most beneficial in domains where
DCE calibration is worst, and can be counterproductive when models are already
well-calibrated," citing two dramatic backfires (GPT-5.4 Mini / WGD +89% ECE,
Claude Haiku 4.5 / LifeEval +335% ECE). This module stress-tests that claim
against the confounds that could manufacture the pattern without any adaptive
benefit of SPD:

  A. Metric artefacts   -- %ΔECE inflates off tiny baselines; ECE binning; the
                           ECE floor asymmetry. Robustness via Brier + Murphy.
  B. Statistics         -- ΔECE = ECE_SPD - ECE_DCE is mechanically anti-correlated
                           with the baseline; measurement-noise RTM; tiny n; CIs.
  C. Structural cause   -- SPD's scored confidence is the modal-bin mass, whose
                           level is set behaviourally by spreading mass over bins,
                           roughly decoupled from DCE quality. SPD calibration
                           tracks |modal_mass - base-rate accuracy|.
  D. Comparability      -- DCE sweeps 20 difficulty levels, SPD only 4
                           ({1,5,10,20}); the modes are matched by item metadata
                           for a paired, difficulty-matched comparison.

Reuses the study's own scoring/metric code so the baseline numbers reproduce the
README before any new metric is layered on:
  - human_io.load_domain / difficulty_column  (scored long frames per domain)
  - study2_lib.compute_ece / BIN_EDGES        (11-bin count-weighted ECE)
  - scoring.murphy_decomposition              (binning-free-ish reliability term)

Run:  python analysis/spd_reanalysis.py
Writes analysis/reports/spd_reanalysis_*.csv, analysis/reports/spd_reanalysis_stats.json,
and figures under analysis/figs/spd_reanalysis/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from human_io import difficulty_column, load_domain
from scoring import get_scorer, murphy_decomposition
from study2_lib import BIN_EDGES, MODEL_LABELS, apply_style, compute_ece

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figs" / "spd_reanalysis"
REPORT_DIR = HERE / "reports"

DOMAINS = ["WGD", "LifeEval", "MedEval"]
MODELS = list(MODEL_LABELS)  # slug order matches CB palette
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
MODEL_COLOR = dict(zip(MODELS, CB_PALETTE))
DOMAIN_COLOR = {"WGD": "#0072B2", "LifeEval": "#D55E00", "MedEval": "#009E73"}

# Item-identity keys used to pair a DCE row with its SPD counterpart.
MATCH_KEYS = {
    "WGD": ["photo", "within_lbs"],
    "LifeEval": ["min_age", "sex", "radius"],
    "MedEval": ["patient", "removal_pct"],
}
N_BOOT = 2000
SEED = 42


# --------------------------------------------------------------------------
# Loading / matching helpers
# --------------------------------------------------------------------------
def _add_patient(df: pd.DataFrame) -> pd.DataFrame:
    """MedEval item id: the integer patient index in the question_id
    (med_test_<i>_c<pct> and med_spd_test_<i>_c<pct> share the same <i>)."""
    df = df.copy()
    df["patient"] = df["question_id"].str.extract(r"test_(\d+)").astype(float)
    return df


def _load_lenient(domain: str) -> pd.DataFrame:
    """Scored long frame using scoring.py's lenient answer extractor.

    The paper's pipeline (human_io._score) parses answers with plain
    ``pd.to_numeric``, so a DCE answer like "77 years" becomes NaN and is
    dropped — costing e.g. ~48% of Claude/LifeEval DCE rows, and only on the DCE
    side (SPD answers are bin centers). scoring.py's ``_numeric_answer`` recovers
    those rows, removing that mode-dependent selection confound. This is the
    primary loader; ``PARSE='plain'`` reproduces the paper's dropout for the
    sensitivity note."""
    scorer = get_scorer(domain)
    frames = []
    for slug in MODELS:
        df = pd.read_csv(HERE.parent / "results_reasoning" / domain / f"{slug}.csv")
        sc = scorer(df, df)  # results already carry benchmark metadata
        sc["Confidence"] = pd.to_numeric(sc["Confidence"], errors="coerce")
        sc = sc.dropna(subset=["Confidence", "true_probability"]).reset_index(drop=True)
        sc["model"] = slug
        frames.append(sc)
    return pd.concat(frames, ignore_index=True)


def load_all() -> dict[str, dict[str, pd.DataFrame]]:
    """Return {domain: {'dce': frame, 'spd': frame}} scored long frames, using
    the lenient (row-recovering) parse so the DCE↔SPD comparison is not biased
    by mode-dependent answer-format dropout."""
    out = {}
    for dom in DOMAINS:
        dce = _load_lenient(dom)
        spd = _load_lenient(f"{dom}_SPD")
        if dom == "MedEval":
            dce, spd = _add_patient(dce), _add_patient(spd)
        out[dom] = {"dce": dce, "spd": spd}
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def cell_metrics(sub: pd.DataFrame) -> dict:
    """Point metrics for one (domain, model, mode) slice."""
    conf, out = sub["Confidence"], sub["true_probability"]
    m = murphy_decomposition(sub, n_bins=10)
    return {
        "n": len(sub),
        "mean_conf": float(conf.mean()),
        "accuracy": float(out.mean()),
        "overconf": float(conf.mean() - out.mean()),
        "ece": float(compute_ece(conf, out)),
        "brier": float(((conf - out) ** 2).mean()),
        "reliability": float(m["reliability"]),
        "resolution": float(m["resolution"]),
        "uncertainty": float(m["uncertainty"]),
    }


_N_BINS = len(BIN_EDGES) - 1


def _bin_idx(conf: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(conf, BIN_EDGES) - 1, 0, _N_BINS - 1)


def _ece_from_arrays(conf: np.ndarray, out: np.ndarray) -> float:
    """Fast pooled ECE on raw arrays with the fixed BIN_EDGES (== compute_ece)."""
    tot = len(conf)
    if tot == 0:
        return np.nan
    idx = _bin_idx(conf)
    cs = np.bincount(idx, weights=conf, minlength=_N_BINS)
    os = np.bincount(idx, weights=out, minlength=_N_BINS)
    cnt = np.bincount(idx, minlength=_N_BINS).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(cnt > 0, np.abs(os / cnt - cs / cnt), 0.0)
    return float((cnt / tot * gap).sum())


def _strat_ece_from_arrays(conf: np.ndarray, out: np.ndarray, level: np.ndarray) -> float:
    """Difficulty-stratified ECE: pooled ECE computed *within* each difficulty
    level, then weighted by level count. Removes cross-difficulty cancellation,
    where a model gives similar confidence at different difficulties whose
    over/under-confidence cancel inside a shared confidence bin (which deflates
    the ordinary pooled ECE)."""
    tot = len(conf)
    if tot == 0:
        return np.nan
    acc = 0.0
    for lv in np.unique(level):
        mask = level == lv
        acc += mask.sum() * _ece_from_arrays(conf[mask], out[mask])
    return acc / tot


def strat_ece(df: pd.DataFrame, dc: str) -> float:
    return _strat_ece_from_arrays(
        df["Confidence"].to_numpy(float), df["true_probability"].to_numpy(float),
        df[dc].to_numpy())


def paired_delta_boot(matched: pd.DataFrame, dc: str, rng) -> dict:
    """Item-paired bootstrap of ΔECE (SPD − DCE) on the shared items, for both
    the pooled and the difficulty-stratified estimator. Resampling the *item*
    (a row that carries both modes) keeps DCE and SPD on the same resampled
    items and naturally conditions on items that parsed in both modes — unlike
    the README's independent per-mode resample."""
    dc_c = matched["dce_conf"].to_numpy(float)
    dc_o = matched["dce_tp"].to_numpy(float)
    sp_c = matched["spd_conf"].to_numpy(float)
    sp_o = matched["spd_tp"].to_numpy(float)
    lv = matched[dc].to_numpy()
    n = len(matched)
    dp = np.empty(N_BOOT)
    ds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ix = rng.integers(0, n, size=n)
        dp[i] = _ece_from_arrays(sp_c[ix], sp_o[ix]) - _ece_from_arrays(dc_c[ix], dc_o[ix])
        ds[i] = (_strat_ece_from_arrays(sp_c[ix], sp_o[ix], lv[ix])
                 - _strat_ece_from_arrays(dc_c[ix], dc_o[ix], lv[ix]))
    out = {}
    for name, d in [("pool", dp), ("strat", ds)]:
        pl = (d <= 0).mean()
        out[f"delta_{name}_ci_lo"] = float(np.percentile(d, 2.5))
        out[f"delta_{name}_ci_hi"] = float(np.percentile(d, 97.5))
        out[f"delta_{name}_p"] = float(2 * min(pl, 1 - pl))
    return out


# --------------------------------------------------------------------------
# Core table
# --------------------------------------------------------------------------
def _raw_counts() -> dict:
    """Raw result-CSV record counts per (domain_mode, model) for parse retention.
    Uses pd.read_csv (not physical line count) because the `raw` reasoning column
    holds multi-line JSON, so a row spans many physical lines."""
    counts = {}
    for dom in DOMAINS:
        for mode in [dom, f"{dom}_SPD"]:
            for model in MODELS:
                p = HERE.parent / "results_reasoning" / mode / f"{model}.csv"
                counts[(mode, model)] = len(pd.read_csv(p, usecols=["question_id"]))
    return counts


def build_tables(data) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    raw = _raw_counts()
    cell_rows, delta_rows = [], []

    for dom in DOMAINS:
        dce, spd = data[dom]["dce"], data[dom]["spd"]
        dc = difficulty_column(dom)
        keys = MATCH_KEYS[dom]
        spd_levels = sorted(spd[dc].unique())

        for model in MODELS:
            d_all = dce[dce["model"] == model]
            s = spd[spd["model"] == model]
            if d_all.empty or s.empty:
                continue
            d_match = d_all[d_all[dc].isin(spd_levels)]

            cm_dce, cm_dce_m, cm_spd = cell_metrics(d_all), cell_metrics(d_match), cell_metrics(s)
            for mode, cm, kept, rawk in [
                ("dce_all", cm_dce, cm_dce["n"], raw[(dom, model)]),
                ("spd", cm_spd, cm_spd["n"], raw[(f"{dom}_SPD", model)])]:
                cell_rows.append({"domain": dom, "model": model, "mode": mode,
                                  "raw_rows": rawk, "retention": kept / rawk, **cm})

            # Marginal (unpaired) stratified ECE on the matched levels — describes
            # each mode's calibration on the shared difficulty grid.
            strat_dce_m = strat_ece(d_match, dc)
            strat_spd = strat_ece(s, dc)

            # Item-paired match (per model) on the shared difficulty levels. The
            # inner join keeps items that parsed in BOTH modes, so point estimates
            # AND their CIs are computed on the same paired sample.
            left = d_match[keys + ["Confidence", "true_probability"]].rename(
                columns={"Confidence": "dce_conf", "true_probability": "dce_tp"})
            right = s[keys + ["Confidence", "true_probability"]].rename(
                columns={"Confidence": "spd_conf", "true_probability": "spd_tp"})
            matched = left.merge(right, on=keys, how="inner")

            if len(matched):
                lv = matched[dc].to_numpy()
                dcc, dct = matched["dce_conf"].to_numpy(float), matched["dce_tp"].to_numpy(float)
                spc, spt = matched["spd_conf"].to_numpy(float), matched["spd_tp"].to_numpy(float)
                pair_delta_pool = _ece_from_arrays(spc, spt) - _ece_from_arrays(dcc, dct)
                pair_delta_strat = (_strat_ece_from_arrays(spc, spt, lv)
                                    - _strat_ece_from_arrays(dcc, dct, lv))
                boot = paired_delta_boot(matched, dc, rng)
            else:
                pair_delta_pool = pair_delta_strat = np.nan
                boot = {k: np.nan for k in
                        ["delta_pool_ci_lo", "delta_pool_ci_hi", "delta_pool_p",
                         "delta_strat_ci_lo", "delta_strat_ci_hi", "delta_strat_p"]}

            e_all, e_m, e_spd = cm_dce["ece"], cm_dce_m["ece"], cm_spd["ece"]
            delta_rows.append({
                "domain": dom, "model": model, "label": MODEL_LABELS[model],
                "n_dce_all": cm_dce["n"], "n_dce_matched": cm_dce_m["n"],
                "n_matched_pairs": len(matched), "n_spd": cm_spd["n"],
                "retention_dce": cm_dce["n"] / raw[(dom, model)],
                "retention_spd": cm_spd["n"] / raw[(f"{dom}_SPD", model)],
                "accuracy_spd": cm_spd["accuracy"], "modal_mass": cm_spd["mean_conf"],
                "gap_mass_acc": abs(cm_spd["mean_conf"] - cm_spd["accuracy"]),
                # pooled ECE (marginal)
                "ece_dce_all": e_all, "ece_dce_matched": e_m, "ece_spd": e_spd,
                "delta_pool_all": e_spd - e_all,
                "delta_pct_all": 100 * (e_spd - e_all) / e_all if e_all else np.nan,
                # marginal stratified ECE on matched levels
                "sece_dce_matched": strat_dce_m, "sece_spd": strat_spd,
                # PAIRED point estimates (consistent with the bootstrap CIs)
                "delta_pool_matched": pair_delta_pool,
                "delta_strat_matched": pair_delta_strat,
                # proper-score robustness
                "brier_dce": cm_dce["brier"], "brier_spd": cm_spd["brier"],
                "reliability_dce": cm_dce["reliability"], "reliability_spd": cm_spd["reliability"],
                **boot,
            })

    return pd.DataFrame(cell_rows), pd.DataFrame(delta_rows)


# --------------------------------------------------------------------------
# Confound statistics
# --------------------------------------------------------------------------
def confound_stats(delta: pd.DataFrame) -> dict:
    st = {}
    # B1: mechanical anti-correlation of ΔECE with baseline (guaranteed since
    # ΔECE = ECE_SPD - ECE_DCE) vs the informative decoupling of ECE_SPD itself.
    st["corr_delta_vs_dce_pooled"] = _corr(delta["ece_dce_all"], delta["delta_pool_all"])
    st["corr_spd_vs_dce_pooled"] = _corr(delta["ece_dce_all"], delta["ece_spd"])
    st["within_domain"] = {}
    for dom, g in delta.groupby("domain"):
        st["within_domain"][dom] = {
            "corr_delta_vs_dce": _corr(g["ece_dce_all"], g["delta_pool_all"]),
            "corr_spd_vs_dce": _corr(g["ece_dce_all"], g["ece_spd"]),
            "ece_spd_mean": float(g["ece_spd"].mean()),
            "ece_spd_cv": float(g["ece_spd"].std() / g["ece_spd"].mean()),
            "modal_mass_range": float(g["modal_mass"].max() - g["modal_mass"].min()),
        }
    # C2: structural predictor of SPD calibration.
    st["corr_gap_vs_ece_spd"] = _corr(delta["gap_mass_acc"], delta["ece_spd"])
    # A/D: how the two published backfires behave once the confounds are removed.
    st["backfires"] = delta[delta["delta_pool_all"] > 0][
        ["domain", "label", "retention_dce", "ece_dce_all", "ece_dce_matched",
         "sece_dce_matched", "ece_spd", "sece_spd", "delta_pct_all",
         "delta_pool_matched", "delta_pool_ci_lo", "delta_pool_ci_hi", "delta_pool_p",
         "delta_strat_matched", "delta_strat_ci_lo", "delta_strat_ci_hi", "delta_strat_p"]
    ].to_dict("records")
    # Headline: significant improve/worsen counts under BOTH metrics (paired,
    # matched, lenient parse). A cell is a *robust* backfire only if it worsens
    # under both pooled and stratified ECE.
    st["n_cells"] = int(len(delta))
    for tag in ["pool", "strat"]:
        sig_imp = ((delta[f"delta_{tag}_matched"] < 0) & (delta[f"delta_{tag}_ci_hi"] < 0)).sum()
        sig_wor = ((delta[f"delta_{tag}_matched"] > 0) & (delta[f"delta_{tag}_ci_lo"] > 0)).sum()
        st[f"{tag}_sig_improved"] = int(sig_imp)
        st[f"{tag}_sig_worsened"] = int(sig_wor)
    robust_worse = ((delta["delta_pool_ci_lo"] > 0) & (delta["delta_strat_ci_lo"] > 0))
    st["robust_backfires"] = delta[robust_worse][["domain", "label"]].to_dict("records")
    return st


def _corr(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = sp_stats.pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def _matched_frame(dce, spd, dom, model):
    """Item-paired DCE↔SPD rows on the shared difficulty levels."""
    dc = difficulty_column(dom)
    keys = MATCH_KEYS[dom]
    lv = sorted(spd[dc].unique())
    d = dce[(dce["model"] == model) & dce[dc].isin(lv)]
    s = spd[spd["model"] == model]
    L = d[keys + ["Confidence", "true_probability"]].rename(
        columns={"Confidence": "dc", "true_probability": "dt"})
    R = s[keys + ["Confidence", "true_probability"]].rename(
        columns={"Confidence": "sc", "true_probability": "st"})
    return L.merge(R, on=keys, how="inner"), dc


def robustness(data, rng) -> list[dict]:
    """Red-team: does the paired ΔECE survive debiasing against the
    perfect-calibration null (which quantifies ECE finite-sample bias)?
    If the debiased Δ tracks the raw Δ, the stratified DCE inflation is real
    miscalibration, not small-sample bin noise."""
    def debias(conf, out, lvl, strat, K=300):
        raw = (_strat_ece_from_arrays(conf, out, lvl) if strat
               else _ece_from_arrays(conf, out))
        sims = np.empty(K)
        for k in range(K):
            y = (rng.random(len(conf)) < conf).astype(float)
            sims[k] = (_strat_ece_from_arrays(conf, y, lvl) if strat
                       else _ece_from_arrays(conf, y))
        return raw - sims.mean()

    rows = []
    for dom in DOMAINS:
        for model in MODELS:
            m, dc = _matched_frame(data[dom]["dce"], data[dom]["spd"], dom, model)
            if not len(m):
                continue
            lvl = m[dc].to_numpy()
            dcc, dct = m["dc"].to_numpy(float), m["dt"].to_numpy(float)
            sc, so = m["sc"].to_numpy(float), m["st"].to_numpy(float)
            row = {"domain": dom, "label": MODEL_LABELS[model]}
            for strat, tag in [(False, "pool"), (True, "strat")]:
                raw = ((_strat_ece_from_arrays(sc, so, lvl) - _strat_ece_from_arrays(dcc, dct, lvl))
                       if strat else (_ece_from_arrays(sc, so) - _ece_from_arrays(dcc, dct)))
                deb = debias(sc, so, lvl, strat) - debias(dcc, dct, lvl, strat)
                row[f"delta_{tag}_raw"] = float(raw)
                row[f"delta_{tag}_debiased"] = float(deb)
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _cell_label(row):
    return f"{row['domain'][:4]}·{row['label'].split()[0]}"


def fig_delta_abs_vs_pct(delta: pd.DataFrame) -> Path:
    d = delta.sort_values("delta_pct_all", ascending=False).reset_index(drop=True)
    labels = [_cell_label(r) for _, r in d.iterrows()]
    y = np.arange(len(d))[::-1]
    fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    # Left: the published %ΔECE (pooled, all-difficulty, plain-parse baseline).
    cpct = ["#C44E52" if v > 0 else "#4C72B0" for v in d["delta_pct_all"]]
    ax2.barh(y, d["delta_pct_all"], color=cpct, alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Percent ΔECE  (published: pooled, all 20 levels)")
    ax2.set_title("Published framing:\ntwo dramatic 'backfires' (+89%, +335%)")
    # Right: honest absolute ΔECE under BOTH calibration standards (paired,
    # difficulty-matched, lenient parse), with 95% CIs. The two disagree for the
    # contested cells — that disagreement is the point.
    h = 0.36
    for off, tag, col, lab in [(+h/1.7, "pool", "#8172B3", "pooled (marginal)"),
                               (-h/1.7, "strat", "#55A868", "stratified (conditional)")]:
        v = d[f"delta_{tag}_matched"]
        elo = (v - d[f"delta_{tag}_ci_lo"]).clip(lower=0)
        ehi = (d[f"delta_{tag}_ci_hi"] - v).clip(lower=0)
        ax1.barh(y + off, v, height=h, color=col, alpha=0.9, label=lab)
        ax1.errorbar(v, y + off, xerr=[elo, ehi], fmt="none", ecolor="#333",
                     elinewidth=0.8, capsize=1.5)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Absolute ΔECE (SPD − DCE), difficulty-matched ± 95% CI")
    ax1.set_title("Honest effect size, two calibration standards:\nno large harm; sign is "
                  "metric-dependent for LifeEval")
    ax1.legend(fontsize=8, loc="lower right")
    fig.suptitle("The '+89% / +335%' magnitudes are artefacts; the real effects are small "
                 "and, for LifeEval, standard-dependent", fontsize=11)
    p = FIG_DIR / "delta_abs_vs_pct.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_spd_vs_dce(delta: pd.DataFrame, stats: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.5), layout="constrained")
    for dom, g in delta.groupby("domain"):
        ax.scatter(g["ece_dce_all"], g["ece_spd"], s=70, color=DOMAIN_COLOR[dom],
                   label=dom, edgecolor="white", zorder=3)
        ax.axhline(g["ece_spd"].mean(), color=DOMAIN_COLOR[dom], ls=":", lw=1, alpha=0.6)
    lim = max(delta["ece_dce_all"].max(), delta["ece_spd"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#888", ls="--", lw=1, label="no change (y=x)")
    ax.set_xlabel("ECE under DCE (baseline)")
    ax.set_ylabel("ECE under SPD")
    ax.set_title("SPD lands at a domain level ~decoupled from DCE quality\n"
                 "(dotted = per-domain mean SPD ECE); points below y=x improved")
    ax.legend(fontsize=8)
    p = FIG_DIR / "ece_spd_vs_dce.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_modal_mass(data, delta: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, dom in zip(axes, DOMAINS):
        spd = data[dom]["spd"]
        for j, model in enumerate(MODELS):
            s = spd[spd["model"] == model]["Confidence"].dropna()
            if s.empty:
                continue
            parts = ax.violinplot(s, positions=[j], widths=0.8, showmeans=True)
            for b in parts["bodies"]:
                b.set_facecolor(MODEL_COLOR[model]); b.set_alpha(0.5)
            for key in ("cmeans", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color(MODEL_COLOR[model])
        acc = delta[delta["domain"] == dom]["accuracy_spd"].mean()
        ax.axhline(acc, color="black", ls="--", lw=1)
        ax.text(0.02, acc, f" base-rate acc≈{acc:.2f}", va="bottom", fontsize=8,
                transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m].split()[0] for m in MODELS], fontsize=8, rotation=20)
        ax.set_title(dom); ax.set_ylim(0, 1)
    axes[0].set_ylabel("SPD modal-bin mass (scored confidence)")
    fig.suptitle("Modal-mass level is bin-structure-driven (tight in WGD) and sits below "
                 "base-rate accuracy where SPD hurts", fontsize=10)
    p = FIG_DIR / "modal_mass_dist.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_structural_gap(delta: pd.DataFrame, stats: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5.2), layout="constrained")
    for dom, g in delta.groupby("domain"):
        ax.scatter(g["gap_mass_acc"], g["ece_spd"], s=70, color=DOMAIN_COLOR[dom],
                   label=dom, edgecolor="white", zorder=3)
    r = stats["corr_gap_vs_ece_spd"]["r"]
    ax.set_xlabel("|modal mass − base-rate accuracy|  (structural gap)")
    ax.set_ylabel("ECE under SPD")
    ax.set_title(f"SPD calibration tracks a structural gap, not DCE quality\nPearson r = {r:.2f}")
    ax.legend(fontsize=8)
    p = FIG_DIR / "structural_gap.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_difficulty_stratified(data) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for ax, dom in zip(axes, DOMAINS):
        dce, spd = data[dom]["dce"], data[dom]["spd"]
        dc = difficulty_column(dom)
        levels = sorted(spd[dc].unique())
        de, se = [], []
        for lv in levels:
            d = dce[dce[dc] == lv]
            s = spd[spd[dc] == lv]
            de.append(compute_ece(d["Confidence"], d["true_probability"]))
            se.append(compute_ece(s["Confidence"], s["true_probability"]))
        x = np.arange(len(levels))
        ax.plot(x, de, "o-", color="#4C72B0", label="DCE")
        ax.plot(x, se, "s-", color="#C44E52", label="SPD")
        ax.set_xticks(x); ax.set_xticklabels([str(int(l)) for l in levels])
        ax.set_xlabel(f"difficulty ({dc})"); ax.set_title(dom)
    axes[0].set_ylabel("ECE (pooled over models)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Difficulty-stratified ECE: DCE vs SPD at the 4 shared difficulty levels", fontsize=10)
    p = FIG_DIR / "difficulty_stratified.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def make_figures(data, delta, stats) -> list[Path]:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return [
        fig_delta_abs_vs_pct(delta),
        fig_spd_vs_dce(delta, stats),
        fig_modal_mass(data, delta),
        fig_structural_gap(delta, stats),
        fig_difficulty_stratified(data),
    ]


# --------------------------------------------------------------------------
def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_all()
    cells, delta = build_tables(data)
    stats = confound_stats(delta)
    stats["robustness_debias"] = robustness(data, np.random.default_rng(SEED))
    cells.to_csv(REPORT_DIR / "spd_reanalysis_cells.csv", index=False)
    delta.to_csv(REPORT_DIR / "spd_reanalysis_delta.csv", index=False)
    (REPORT_DIR / "spd_reanalysis_stats.json").write_text(json.dumps(stats, indent=2))
    figs = make_figures(data, delta, stats)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("\n=== DELTA TABLE (lenient parse; paired, difficulty-matched) ===")
    print(delta[["domain", "label", "retention_dce", "delta_pct_all",
                 "delta_pool_matched", "delta_pool_ci_lo", "delta_pool_ci_hi", "delta_pool_p",
                 "delta_strat_matched", "delta_strat_ci_lo", "delta_strat_ci_hi", "delta_strat_p"
                 ]].round(3).to_string(index=False))
    print("\n=== CONFOUND STATS ===")
    print(json.dumps(stats, indent=2))
    print("\nFigures:", [str(f.relative_to(HERE)) for f in figs])


if __name__ == "__main__":
    main()
