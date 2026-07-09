"""Validate naive Bayes posterior estimation against DDXPlus ground-truth differentials."""

import argparse
import ast
import csv
import io
import math
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_patients(data_dir: Path) -> list[dict]:
    zip_path = data_dir / "release_test_patients.zip"
    patients = []
    with zipfile.ZipFile(zip_path) as z:
        inner = z.namelist()[0]
        with z.open(inner) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            for row in reader:
                evidences = set(ast.literal_eval(row["EVIDENCES"]))
                differential = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
                patients.append({
                    "pathology": row["PATHOLOGY"],
                    "evidences": evidences,
                    "differential": differential,
                })
    return patients


def estimate_parameters(patients: list[dict], alpha: float):
    pathology_counts = Counter(p["pathology"] for p in patients)
    pathologies = sorted(pathology_counts.keys())
    path_idx = {p: i for i, p in enumerate(pathologies)}

    all_evidences = set()
    for p in patients:
        all_evidences.update(p["evidences"])
    evidences = sorted(all_evidences)
    ev_idx = {e: i for i, e in enumerate(evidences)}

    n_path = len(pathologies)
    n_ev = len(evidences)
    N = len(patients)

    counts = np.zeros((n_path, n_ev), dtype=np.float64)
    for p in patients:
        d_i = path_idx[p["pathology"]]
        for e in p["evidences"]:
            counts[d_i, ev_idx[e]] += 1

    path_totals = np.array([pathology_counts[d] for d in pathologies], dtype=np.float64)
    log_priors = np.log(path_totals / N)

    # P(e|d) with Laplace smoothing
    log_lik = np.log((counts + alpha) / (path_totals[:, None] + 2 * alpha))

    return {
        "log_priors": log_priors,
        "log_lik": log_lik,
        "pathologies": pathologies,
        "path_idx": path_idx,
        "evidences": evidences,
        "ev_idx": ev_idx,
    }


def compute_posteriors(evidence_set: set, params: dict, temperature: float,
                       candidate_set: list[str] | None = None) -> dict[str, float]:
    log_priors = params["log_priors"]
    log_lik = params["log_lik"]
    ev_idx = params["ev_idx"]
    pathologies = params["pathologies"]
    path_idx = params["path_idx"]

    # Sum log-likelihoods for present evidences only
    ev_indices = [ev_idx[e] for e in evidence_set if e in ev_idx]
    if ev_indices:
        log_post = log_priors + log_lik[:, ev_indices].sum(axis=1)
    else:
        log_post = log_priors.copy()

    # Temperature scaling
    log_post = log_post / temperature

    # Restrict to candidate set if provided
    if candidate_set:
        cand_indices = [path_idx[c] for c in candidate_set if c in path_idx]
        cand_log_post = log_post[cand_indices]
        cand_log_post -= cand_log_post.max()
        probs = np.exp(cand_log_post)
        probs /= probs.sum()
        cand_names = [pathologies[i] for i in cand_indices]
        return dict(zip(cand_names, probs))
    else:
        log_post -= log_post.max()
        probs = np.exp(log_post)
        probs /= probs.sum()
        return dict(zip(pathologies, probs))


def evaluate(sample: list[dict], params: dict, temperature: float):
    maes, pearsons, spearmans = [], [], []
    top1_correct, top3_correct = 0, 0
    kls, jsds = [], []
    n_valid = 0

    for patient in sample:
        gt_diff = patient["differential"]
        if len(gt_diff) < 2:
            continue
        n_valid += 1

        candidate_set = [name for name, _ in gt_diff]
        nb_post = compute_posteriors(patient["evidences"], params, temperature, candidate_set)

        gt_vec = np.array([prob for _, prob in gt_diff])
        nb_vec = np.array([nb_post.get(name, 0.0) for name, _ in gt_diff])

        # MAE
        maes.append(np.mean(np.abs(gt_vec - nb_vec)))

        # Correlation
        if np.std(gt_vec) > 1e-10 and np.std(nb_vec) > 1e-10:
            r, _ = pearsonr(gt_vec, nb_vec)
            pearsons.append(r)
            rho, _ = spearmanr(gt_vec, nb_vec)
            spearmans.append(rho)

        # Top-k accuracy
        gt_top1 = gt_diff[np.argmax(gt_vec)][0]
        nb_ranking = sorted(nb_post.items(), key=lambda x: -x[1])
        if nb_ranking[0][0] == gt_top1:
            top1_correct += 1
        if gt_top1 in [name for name, _ in nb_ranking[:3]]:
            top3_correct += 1

        # KL divergence: KL(GT || NB)
        eps = 1e-10
        nb_vec_safe = np.clip(nb_vec, eps, None)
        gt_vec_safe = np.clip(gt_vec, eps, None)
        kl = np.sum(gt_vec_safe * np.log(gt_vec_safe / nb_vec_safe))
        kls.append(kl)

        # Jensen-Shannon divergence
        m = 0.5 * (gt_vec_safe + nb_vec_safe)
        jsd = 0.5 * np.sum(gt_vec_safe * np.log2(gt_vec_safe / m)) + \
              0.5 * np.sum(nb_vec_safe * np.log2(nb_vec_safe / m))
        jsds.append(jsd)

    return {
        "n_valid": n_valid,
        "mae": np.mean(maes),
        "pearson": np.mean(pearsons),
        "spearman": np.mean(spearmans),
        "top1_acc": top1_correct / n_valid,
        "top3_acc": top3_correct / n_valid,
        "kl_div": np.mean(kls),
        "jsd": np.mean(jsds),
    }


def validate_removal(sample: list[dict], params: dict, temperature: float, seed: int):
    removal_pcts = [0, 10, 25, 50]
    rng = random.Random(seed)

    print("\n--- Evidence Removal Validation ---")
    print(f"{'Removal%':<10} {'Mean TV Distance':<20} {'Mean MAE vs Full':<20}")
    print("-" * 50)

    for pct in removal_pcts:
        tv_distances = []
        mae_vs_full = []

        for patient in sample:
            if len(patient["differential"]) < 2:
                continue
            candidate_set = [name for name, _ in patient["differential"]]
            evidences = list(patient["evidences"])

            # Full posterior
            full_post = compute_posteriors(patient["evidences"], params, temperature, candidate_set)

            # Reduced posterior
            if pct > 0:
                n_remove = math.ceil(len(evidences) * pct / 100)
                remove_idx = set(rng.sample(range(len(evidences)), min(n_remove, len(evidences) - 1)))
                reduced = {e for j, e in enumerate(evidences) if j not in remove_idx}
            else:
                reduced = patient["evidences"]

            red_post = compute_posteriors(reduced, params, temperature, candidate_set)

            # Total variation distance
            full_vec = np.array([full_post.get(c, 0.0) for c in candidate_set])
            red_vec = np.array([red_post.get(c, 0.0) for c in candidate_set])
            tv = 0.5 * np.sum(np.abs(full_vec - red_vec))
            tv_distances.append(tv)
            mae_vs_full.append(np.mean(np.abs(full_vec - red_vec)))

        print(f"{pct}%{'':<8} {np.mean(tv_distances):<20.4f} {np.mean(mae_vs_full):<20.4f}")


def main():
    parser = argparse.ArgumentParser(description="Validate naive Bayes vs DDXPlus GT")
    parser.add_argument("--n-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperatures", type=str, default="1,10,50,100,150,200")
    parser.add_argument("--data-dir", type=str, default="..")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    temperatures = [float(t) for t in args.temperatures.split(",")]

    print("=== MedEval Naive Bayes Validation ===\n")
    print("Loading patients...")
    patients = load_patients(data_dir)
    print(f"Loaded {len(patients)} patients")

    unique_paths = set(p["pathology"] for p in patients)
    all_ev = set()
    for p in patients:
        all_ev.update(p["evidences"])
    print(f"  {len(unique_paths)} pathologies, {len(all_ev)} evidence features\n")

    print("Estimating parameters...")
    params = estimate_parameters(patients, args.alpha)
    print("Done.\n")

    rng = random.Random(args.seed)
    sample = rng.sample(patients, min(args.n_sample, len(patients)))
    print(f"Evaluating on {len(sample)} sampled patients\n")

    print("--- Temperature Sweep (candidate-normalized) ---")
    header = f"{'T':<8} {'MAE':<8} {'Pearson':<9} {'Spearman':<9} {'Top-1':<7} {'Top-3':<7} {'KL':<8} {'JSD':<8}"
    print(header)
    print("-" * len(header))

    best_temp = temperatures[0]
    best_mae = float("inf")
    results = {}

    for temp in temperatures:
        metrics = evaluate(sample, params, temp)
        results[temp] = metrics
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_temp = temp
        print(f"{temp:<8.0f} {metrics['mae']:<8.4f} {metrics['pearson']:<9.4f} "
              f"{metrics['spearman']:<9.4f} {metrics['top1_acc']:<7.3f} "
              f"{metrics['top3_acc']:<7.3f} {metrics['kl_div']:<8.4f} {metrics['jsd']:<8.4f}")

    print(f"\nBest temperature by MAE: T={best_temp}")
    m = results[best_temp]
    print(f"  MAE={m['mae']:.4f}, Pearson={m['pearson']:.4f}, "
          f"Spearman={m['spearman']:.4f}, Top-1={m['top1_acc']:.3f}, Top-3={m['top3_acc']:.3f}")

    validate_removal(sample, params, best_temp, args.seed)


if __name__ == "__main__":
    main()
