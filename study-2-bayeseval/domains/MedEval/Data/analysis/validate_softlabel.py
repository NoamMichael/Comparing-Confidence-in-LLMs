"""Soft-label logistic regression (KL loss) for posterior estimation. Pure numpy/scipy."""

import ast
import csv
import io
import random
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
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


def build_feature_matrix(patients, ev_list, ev_idx):
    X = np.zeros((len(patients), len(ev_list)), dtype=np.float64)
    for i, p in enumerate(patients):
        for e in p["evidences"]:
            if e in ev_idx:
                X[i, ev_idx[e]] = 1.0
    return X


def build_soft_targets(patients, pathologies, path_idx):
    Y = np.full((len(patients), len(pathologies)), 1e-8, dtype=np.float64)
    for i, p in enumerate(patients):
        for name, prob in p["differential"]:
            if name in path_idx:
                Y[i, path_idx[name]] = prob
    Y /= Y.sum(axis=1, keepdims=True)
    return Y


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def kl_loss_and_grad(W_flat, X, Y, n_classes, reg=1e-4):
    """KL(Y || softmax(X @ W)) + L2 regularization."""
    n_features = X.shape[1]
    W = W_flat.reshape(n_features, n_classes)

    logits = X @ W
    P = softmax(logits)

    # KL divergence: sum of Y * log(Y/P) = sum of Y*log(Y) - Y*log(P)
    # We only need the part that depends on W: -sum(Y * log(P))
    log_P = np.log(P + 1e-10)
    loss = -np.mean(np.sum(Y * log_P, axis=1)) + 0.5 * reg * np.sum(W ** 2)

    # Gradient: d/dW = X.T @ (P - Y) / N + reg * W
    grad_W = X.T @ (P - Y) / X.shape[0] + reg * W
    return loss, grad_W.ravel()


def evaluate_model(X_test, W, test_patients, pathologies, path_idx):
    logits = X_test @ W
    P = softmax(logits)

    maes, pearsons, spearmans = [], [], []
    top1_correct, top3_correct = 0, 0
    n_valid = 0

    for i, patient in enumerate(test_patients):
        gt_diff = patient["differential"]
        if len(gt_diff) < 2:
            continue
        n_valid += 1

        gt_names = [name for name, _ in gt_diff]
        gt_vec = np.array([prob for _, prob in gt_diff])

        # Get predicted probs for GT candidates, renormalize
        pred_raw = np.array([P[i, path_idx[name]] if name in path_idx else 1e-10 for name in gt_names])
        pred_vec = pred_raw / pred_raw.sum()

        maes.append(np.mean(np.abs(gt_vec - pred_vec)))

        if np.std(gt_vec) > 1e-10 and np.std(pred_vec) > 1e-10:
            r, _ = pearsonr(gt_vec, pred_vec)
            pearsons.append(r)
            rho, _ = spearmanr(gt_vec, pred_vec)
            spearmans.append(rho)

        gt_top1 = gt_names[np.argmax(gt_vec)]
        pred_ranking = sorted(zip(gt_names, pred_vec), key=lambda x: -x[1])
        if pred_ranking[0][0] == gt_top1:
            top1_correct += 1
        if gt_top1 in [name for name, _ in pred_ranking[:3]]:
            top3_correct += 1

    return {
        "n_valid": n_valid,
        "mae": np.mean(maes),
        "pearson": np.mean(pearsons),
        "spearman": np.mean(spearmans),
        "top1_acc": top1_correct / n_valid,
        "top3_acc": top3_correct / n_valid,
    }


def train_sgd(X, Y, n_classes, n_epochs=30, batch_size=1024, lr=0.1, reg=1e-4):
    """Mini-batch SGD — much lighter than L-BFGS on full dataset."""
    n_samples, n_features = X.shape
    W = np.random.randn(n_features, n_classes).astype(np.float64) * 0.01
    indices = np.arange(n_samples)

    for epoch in range(n_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            batch = indices[start:start + batch_size]
            Xb, Yb = X[batch], Y[batch]

            logits = Xb @ W
            P = softmax(logits)
            log_P = np.log(P + 1e-10)
            loss = -np.mean(np.sum(Yb * log_P, axis=1)) + 0.5 * reg * np.sum(W ** 2)
            epoch_loss += loss

            grad = Xb.T @ (P - Yb) / len(batch) + reg * W
            W -= lr * grad
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

    return W


def main():
    data_dir = Path(".")
    print("=== Soft-Label Model Comparison ===\n")

    print("Loading patients...")
    patients = load_patients(data_dir)
    print(f"Loaded {len(patients)} patients")

    all_ev = set()
    for p in patients:
        all_ev.update(p["evidences"])
    ev_list = sorted(all_ev)
    ev_idx = {e: i for i, e in enumerate(ev_list)}

    pathologies = sorted(set(p["pathology"] for p in patients))
    path_idx = {p: i for i, p in enumerate(pathologies)}
    n_classes = len(pathologies)

    # Train/test split
    rng = random.Random(42)
    indices = list(range(len(patients)))
    rng.shuffle(indices)
    split = int(0.8 * len(patients))
    train_patients = [patients[i] for i in indices[:split]]
    test_patients = [patients[i] for i in indices[split:]]

    print(f"Train: {len(train_patients)}, Test: {len(test_patients)}")
    print(f"Features: {len(ev_list)}, Classes: {n_classes}\n")

    print("Building matrices...")
    X_train = build_feature_matrix(train_patients, ev_list, ev_idx)
    X_test = build_feature_matrix(test_patients, ev_list, ev_idx)
    Y_train = build_soft_targets(train_patients, pathologies, path_idx)
    print("Done.\n")

    # --- SGD softmax regression ---
    print("Training soft-label softmax regression (SGD, 30 epochs)...")
    W_sgd = train_sgd(X_train, Y_train, n_classes, n_epochs=30, batch_size=1024, lr=0.1, reg=1e-4)
    metrics_sgd = evaluate_model(X_test, W_sgd, test_patients, pathologies, path_idx)
    print()

    # --- Try lower regularization ---
    print("Training with lower reg (1e-5)...")
    W_sgd2 = train_sgd(X_train, Y_train, n_classes, n_epochs=50, batch_size=1024, lr=0.05, reg=1e-5)
    metrics_sgd2 = evaluate_model(X_test, W_sgd2, test_patients, pathologies, path_idx)
    print()

    # --- Results ---
    print("\n--- Results (candidate-normalized) ---")
    header = f"{'Model':<35} {'MAE':<8} {'Pearson':<9} {'Spearman':<9} {'Top-1':<7} {'Top-3':<7}"
    print(header)
    print("-" * len(header))

    print(f"{'Soft LogReg (reg=1e-4, 30ep)':<35} {metrics_sgd['mae']:<8.4f} {metrics_sgd['pearson']:<9.4f} "
          f"{metrics_sgd['spearman']:<9.4f} {metrics_sgd['top1_acc']:<7.3f} {metrics_sgd['top3_acc']:<7.3f}")
    print(f"{'Soft LogReg (reg=1e-5, 50ep)':<35} {metrics_sgd2['mae']:<8.4f} {metrics_sgd2['pearson']:<9.4f} "
          f"{metrics_sgd2['spearman']:<9.4f} {metrics_sgd2['top1_acc']:<7.3f} {metrics_sgd2['top3_acc']:<7.3f}")
    print()
    print("--- Baselines ---")
    print(f"{'Naive Bayes (T=150)':<35} {'0.0464':<8} {'0.5602':<9} {'0.4102':<9} {'0.729':<7} {'0.885':<7}")
    print(f"{'Hard-label LogReg (sklearn)':<35} {'0.2025':<8} {'0.5227':<9} {'0.4806':<9} {'0.716':<7} {'0.897':<7}")


if __name__ == "__main__":
    main()
