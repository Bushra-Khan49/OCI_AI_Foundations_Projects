"""
train_eval.py

Standalone, reproducible comparison of linear vs. non-linear classifiers on
the Golub et al. (1999) leukemia gene expression dataset (ALL vs. AML).

Run `python data/download_data.py` first to produce data/leukemia_clean.csv.

Usage:
    python train_eval.py
    python train_eval.py --n-features 100 --n-splits 10 --results-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier


def parse_args():
    p = argparse.ArgumentParser(description="Linear vs. non-linear classifiers on Golub leukemia data.")
    p.add_argument("--data-path", type=str, default="data/leukemia_clean.csv")
    p.add_argument("--n-features", type=int, default=50, help="Genes to keep via ANOVA F-test, per fold.")
    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run `python data/download_data.py` first."
        )

    df = pd.read_csv(data_path)
    X = df.drop(columns=["label"]).values
    y = (df["label"] == "AML").astype(int).values

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} genes.")
    print(f"Class balance: ALL={np.sum(y==0)}, AML={np.sum(y==1)}")

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    # --- Baseline ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy_scores = cross_val_score(dummy, X, y, cv=cv, scoring="accuracy")
    print(f"\nMajority-class baseline: {dummy_scores.mean():.3f} +/- {dummy_scores.std():.3f}")

    # --- PCA plot ---
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=args.random_state)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, name, color in [(0, "ALL", "tab:blue"), (1, "AML", "tab:orange")]:
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.7, color=color)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA projection: ALL vs. AML gene expression")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "pca_projection.png", dpi=150)
    plt.close(fig)

    # --- Model comparison via CV, feature selection inside the pipeline ---
    def make_pipeline(clf):
        return Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=args.n_features)),
            ("clf", clf),
        ])

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000),
        "linear_svm": LinearSVC(max_iter=5000),
        "rbf_svm": SVC(kernel="rbf"),
        "mlp_10_hidden": MLPClassifier(hidden_layer_sizes=(10,), max_iter=3000, random_state=args.random_state),
    }

    cv_results = {}
    print(f"\nComparing models with {args.n_splits}-fold CV, top {args.n_features} genes per fold:")
    for name, clf in models.items():
        pipe = make_pipeline(clf)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        cv_results[name] = {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}
        print(f"  {name:25s}  {scores.mean():.3f} +/- {scores.std():.3f}")

    # --- Comparison plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(cv_results.keys())
    means = [cv_results[n]["mean"] for n in names]
    stds = [cv_results[n]["std"] for n in names]
    ax.barh(names, means, xerr=stds, color="tab:blue", alpha=0.8)
    ax.axvline(dummy_scores.mean(), color="red", linestyle="--",
               label=f"Majority-class baseline ({dummy_scores.mean():.2f})")
    ax.set_xlabel(f"{args.n_splits}-fold CV accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Linear vs. non-linear models (top {args.n_features} genes)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=150)
    plt.close(fig)

    # --- Save metrics ---
    summary = {
        "dataset": {
            "name": "Golub et al. 1999 leukemia gene expression (ALL vs AML)",
            "n_samples": int(X.shape[0]),
            "n_genes_total": int(X.shape[1]),
            "n_genes_selected_per_fold": args.n_features,
            "class_counts": {"ALL": int(np.sum(y == 0)), "AML": int(np.sum(y == 1))},
        },
        "cv": {"n_splits": args.n_splits, "random_state": args.random_state},
        "majority_class_baseline_accuracy": {
            "mean": float(dummy_scores.mean()), "std": float(dummy_scores.std())
        },
        "model_comparison": cv_results,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved plots and metrics.json to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
