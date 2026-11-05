"""
Tests 4 ways of putting data into a quantum model and compares them.
Also runs a few normal (classical) models to compare against.

Run:  python run_experiments.py
Fast test:  python run_experiments.py --quick
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from vqc_encodings import prepare, train_and_evaluate

METHODS = ["basis", "angle", "amplitude", "reupload"]


def classical_baselines(X_tr, y_tr, X_te, y_te, n_features, split_seed):
    """Run 3 normal models on all features and on the PCA features."""
    scaler = StandardScaler().fit(X_tr)
    Str, Ste = scaler.transform(X_tr), scaler.transform(X_te)

    pca = PCA(n_components=n_features).fit(Str)
    Ptr, Pte = pca.transform(Str), pca.transform(Ste)

    models = {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=split_seed),
        "svm_rbf": lambda: SVC(kernel="rbf"),
    }
    results = []
    for name, make in models.items():
        acc_all = make().fit(Str, y_tr).score(Ste, y_te)
        acc_pca = make().fit(Ptr, y_tr).score(Pte, y_te)
        results.append({"model": name, "features": "all_30", "test_accuracy": acc_all})
        results.append({"model": name, "features": f"pca_{n_features}", "test_accuracy": acc_pca})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-features", type=int, default=4, help="PCA components for every quantum model")
    ap.add_argument("--bits", type=int, default=2, help="bits per feature for basis encoding")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=5, help="weight-initialization seeds per method")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="2 epochs, 1 seed, 40/20 samples")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.seeds = 2, 1

    out_dir = Path(__file__).resolve().parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Split data into train and test
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=args.test_size,
        stratify=data.target, random_state=args.split_seed)
    if args.quick:
        X_tr, y_tr, X_te, y_te = X_tr[:40], y_tr[:40], X_te[:20], y_te[:20]

    majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    majority_acc = majority.score(X_te, y_te)
    print(f"Train size: {len(y_tr)}  Test size: {len(y_te)}  "
          f"Majority-class test accuracy: {majority_acc:.3f}")

    # Run normal models
    baseline_rows = classical_baselines(X_tr, y_tr, X_te, y_te, args.n_features, args.split_seed)
    for r in baseline_rows:
        print(f"baseline {r['model']:20s} {r['features']:8s} {r['test_accuracy']:.3f}")

    # Run quantum models
    per_seed_rows = []
    for method in METHODS:
        Xtr, Xte, n_wires = prepare(method, X_tr, X_te,
                                    n_features=args.n_features, bits=args.bits)
        for seed in range(args.seeds):
            t0 = time.time()
            acc = train_and_evaluate(method, Xtr, y_tr, Xte, y_te,
                                     n_wires=n_wires, n_layers=args.layers,
                                     epochs=args.epochs, lr=args.lr, seed=seed)
            secs = round(time.time() - t0, 1)
            per_seed_rows.append({"model": method, "seed": seed, "wires": n_wires,
                                  "test_accuracy": acc, "seconds": secs})
            print(f"{method:10s} seed={seed} wires={n_wires} acc={acc:.3f} ({secs}s)")

    # Average the results
    summary_rows = []
    for method in METHODS:
        accs = np.array([r["test_accuracy"] for r in per_seed_rows if r["model"] == method])
        std = accs.std(ddof=1) if len(accs) > 1 else 0.0
        summary_rows.append({"model": method, "n_seeds": len(accs),
                             "mean_accuracy": accs.mean(), "std_accuracy": std,
                             "min_accuracy": accs.min(), "max_accuracy": accs.max()})

    print("\nSummary (test accuracy over seeds)")
    for s in summary_rows:
        print(f"{s['model']:10s} mean={s['mean_accuracy']:.3f} "
              f"std={s['std_accuracy']:.3f} min={s['min_accuracy']:.3f} max={s['max_accuracy']:.3f}")

    # Save results
    def write_csv(name, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write_csv("per_seed.csv", per_seed_rows)
    write_csv("summary.csv", summary_rows)
    write_csv("classical_baselines.csv", baseline_rows)

    config = {
        "args": vars(args),
        "dataset": "sklearn load_breast_cancer (569 samples, 30 features)",
        "train_size": int(len(y_tr)),
        "test_size": int(len(y_te)),
        "majority_class_test_accuracy": float(majority_acc),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pennylane": qml.__version__,
        "simulator": "default.qubit (noiseless)",
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
