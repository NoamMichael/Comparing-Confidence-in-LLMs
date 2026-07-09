"""Compare naive Bayes, logistic regression, and MLP for posterior estimation."""

import ast
import csv
import io
import random
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


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
    X = np.zeros((len(patients), len(ev_list)), dtype=np.float32)
    for i, p in enumerate(patients):
        for e in p["evidences"]:
            if e in ev_idx:
                X[i, ev_idx[e]] = 1.0
    return X


def evaluate_model(predictions: list[dict], sample: list[dict]):
    """predictions[i] = dict mapping pathology -> predicted probability"""
    maes, pearsons, spearmans = [], [], []
    top1_correct, top3_correct = 0, 0
    n_valid = 0

    for pred, patient in zip(predictions, sample):
        gt_diff = patient["differential"]
        if len(gt_diff) < 2:
            continue
        n_valid += 1

        gt_names = [name for name, _ in gt_diff]
        gt_vec = np.array([prob for _, prob in gt_diff])

        # Get predicted probs for GT candidates, renormalize
        pred_raw = np.array([pred.get(name, 1e-10) for name in gt_names])
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


def main():
    data_dir = Path(".")
    print("=== Model Comparison: NB vs LogReg vs MLP ===\n")

    print("Loading patients...")
    patients = load_patients(data_dir)
    print(f"Loaded {len(patients)} patients\n")

    # Build feature space
    all_ev = set()
    for p in patients:
        all_ev.update(p["evidences"])
    ev_list = sorted(all_ev)
    ev_idx = {e: i for i, e in enumerate(ev_list)}

    pathologies = sorted(set(p["pathology"] for p in patients))
    path_idx = {p: i for i, p in enumerate(pathologies)}

    # Train/test split (use 80% train, 20% test)
    rng = random.Random(42)
    indices = list(range(len(patients)))
    rng.shuffle(indices)
    split = int(0.8 * len(patients))
    train_idx = indices[:split]
    test_idx = indices[split:]

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"Features: {len(ev_list)}, Classes: {len(pathologies)}\n")

    # Build matrices
    print("Building feature matrices...")
    train_patients = [patients[i] for i in train_idx]
    test_patients = [patients[i] for i in test_idx]

    X_train = build_feature_matrix(train_patients, ev_list, ev_idx)
    X_test = build_feature_matrix(test_patients, ev_list, ev_idx)
    y_train = np.array([path_idx[p["pathology"]] for p in train_patients])

    # --- Logistic Regression ---
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)
    lr_predictions = []
    for i in range(len(test_patients)):
        pred = {pathologies[j]: lr_proba[i, j] for j in range(len(pathologies))}
        lr_predictions.append(pred)
    print("  Done.")

    # --- MLP (small) ---
    print("Training MLP (256, 128)...")
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True,
                        validation_fraction=0.1, random_state=42, batch_size=512)
    mlp.fit(X_train, y_train)
    mlp_proba = mlp.predict_proba(X_test)
    mlp_predictions = []
    for i in range(len(test_patients)):
        pred = {pathologies[j]: mlp_proba[i, j] for j in range(len(pathologies))}
        mlp_predictions.append(pred)
    print("  Done.")

    # --- MLP (larger) ---
    print("Training MLP (512, 256, 128)...")
    mlp2 = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=300, early_stopping=True,
                         validation_fraction=0.1, random_state=42, batch_size=512)
    mlp2.fit(X_train, y_train)
    mlp2_proba = mlp2.predict_proba(X_test)
    mlp2_predictions = []
    for i in range(len(test_patients)):
        pred = {pathologies[j]: mlp2_proba[i, j] for j in range(len(pathologies))}
        mlp2_predictions.append(pred)
    print("  Done.")

    # --- Evaluate all ---
    print("\n--- Results (candidate-normalized) ---")
    header = f"{'Model':<25} {'MAE':<8} {'Pearson':<9} {'Spearman':<9} {'Top-1':<7} {'Top-3':<7}"
    print(header)
    print("-" * len(header))

    for name, preds in [("LogisticRegression", lr_predictions),
                        ("MLP (256,128)", mlp_predictions),
                        ("MLP (512,256,128)", mlp2_predictions)]:
        metrics = evaluate_model(preds, test_patients)
        print(f"{name:<25} {metrics['mae']:<8.4f} {metrics['pearson']:<9.4f} "
              f"{metrics['spearman']:<9.4f} {metrics['top1_acc']:<7.3f} {metrics['top3_acc']:<7.3f}")

    # --- Naive Bayes baseline for comparison ---
    print("\n--- Naive Bayes baseline (T=150, from previous script) ---")
    print(f"{'Naive Bayes (T=150)':<25} {'0.0464':<8} {'0.5602':<9} {'0.4102':<9} {'0.729':<7} {'0.885':<7}")

    # --- Also try: train on soft labels with KL loss (PyTorch) ---
    print("\n\n--- Soft-label MLP (trained on GT differential via KL divergence) ---")
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Build soft labels: for each training patient, the target is the GT differential distribution
        print("Building soft label targets...")
        y_soft_train = np.zeros((len(train_patients), len(pathologies)), dtype=np.float32)
        for i, p in enumerate(train_patients):
            for name, prob in p["differential"]:
                if name in path_idx:
                    y_soft_train[i, path_idx[name]] = prob

        y_soft_train /= y_soft_train.sum(axis=1, keepdims=True)  # ensure normalized

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_soft_train)
        X_test_t = torch.from_numpy(X_test)

        dataset = TensorDataset(X_train_t, y_train_t)
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

        class SoftMLP(nn.Module):
            def __init__(self, n_in, n_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, n_out),
                )

            def forward(self, x):
                return self.net(x)

        device = torch.device("cpu")
        model = SoftMLP(len(ev_list), len(pathologies)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("Training soft-label MLP (KL divergence loss)...")
        model.train()
        for epoch in range(50):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = torch.sum(yb * (torch.log(yb + 1e-10) - log_probs)) / xb.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        soft_predictions = []
        for i in range(len(test_patients)):
            pred = {pathologies[j]: float(probs[i, j]) for j in range(len(pathologies))}
            soft_predictions.append(pred)

        metrics = evaluate_model(soft_predictions, test_patients)
        print(f"\n{'Soft MLP (KL loss)':<25} {metrics['mae']:<8.4f} {metrics['pearson']:<9.4f} "
              f"{metrics['spearman']:<9.4f} {metrics['top1_acc']:<7.3f} {metrics['top3_acc']:<7.3f}")

    except ImportError:
        print("  PyTorch not available, skipping soft-label MLP.")


if __name__ == "__main__":
    main()
