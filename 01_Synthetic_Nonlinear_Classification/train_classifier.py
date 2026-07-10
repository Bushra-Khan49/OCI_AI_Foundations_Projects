"""
train_classifier.py

Standalone, reproducible version of the make_circles classification experiment.
Unlike the original version, this script:
  - splits data into train/test and reports accuracy on held-out data
  - fits a linear baseline (LogisticRegression) alongside the MLP
  - sweeps hidden layer size and records test accuracy at each step
  - saves a trained model, plots, and a metrics.json file to --results-dir

In this project, I explore the limits of linear models when dealing with non-linear data, 
and demonstrate how introducing a simple Multi-Layer Perceptron (MLP) neural network solves it.

Usage:
    python train_classifier.py
    python train_classifier.py --hidden-sizes 1 5 10 20 50 --noise 0.05 --results-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed; we only save figures
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def parse_args():
    p = argparse.ArgumentParser(description="Train MLP vs. linear baseline on make_circles.")
    p.add_argument("--n-samples", type=int, default=300, help="Number of points in the dataset.")
    p.add_argument("--noise", type=float, default=0.05, help="Noise level for make_circles.")
    p.add_argument("--factor", type=float, default=0.5, help="Inner/outer circle scale factor.")
    p.add_argument("--test-size", type=float, default=0.3, help="Fraction of data held out for testing.")
    p.add_argument("--random-state", type=int, default=0, help="Random seed for data + splits.")
    p.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50],
        help="Hidden layer sizes to sweep over."
    )
    p.add_argument("--results-dir", type=str, default="results", help="Where to save outputs.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generating Data
    # make_circles generates data points in the shape of two interlocking, concentric circles.
    # By design, you cannot draw a single straight line to separate the inner circle from the outer one.
    print("Generating dataset...")
    X, y = make_circles(
        n_samples=args.n_samples, noise=args.noise, factor=args.factor,
        random_state=args.random_state,
    )
    
    # 2. Validation Cohort
    # I hold out a portion of this data as a validation cohort. 
    # Evaluating a model on the same data it learned from leads to a false sense of security due to overfitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # 3. Linear Baseline
    # I fit a standard Logistic Regression baseline. Logistic Regression attempts to find 
    # a linear decision boundary (a flat plane). 
    print("\nFitting linear baseline (LogisticRegression)...")
    baseline = LogisticRegression()
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"  Baseline test accuracy: {baseline_acc:.3f}")

    # 4. MLP Sweep
    # Next, I introduce an MLP neural network. I sweep through different hidden layer sizes 
    # to observe how adding capacity allows the network to combine multiple flat boundaries 
    # into a smooth, curved shape. Both models use the exact same validation splits for a fair comparison.
    print("\nSweeping MLP hidden layer size...")
    sweep_results = []
    best_clf, best_acc, best_h = None, -1.0, None
    for h in args.hidden_sizes:
        clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", max_iter=3000, random_state=1)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        sweep_results.append({"hidden_layer_size": h, "test_accuracy": acc})
        print(f"  hidden_layer_size={h:>3}  ->  test accuracy={acc:.3f}")
        if acc > best_acc:
            best_clf, best_acc, best_h = clf, acc, h

    print(f"\nBest hidden layer size: {best_h} (test accuracy={best_acc:.3f})")
    print("\nClassification report for best model:")
    print(classification_report(y_test, best_clf.predict(X_test)))

    # --- Plot 1: accuracy vs. hidden layer size ---
    sizes = [r["hidden_layer_size"] for r in sweep_results]
    accs = [r["test_accuracy"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, accs, marker="o", label="MLP")
    ax.axhline(baseline_acc, color="red", linestyle="--", label=f"Logistic Regression ({baseline_acc:.2f})")
    ax.set_xlabel("Hidden layer size (neurons)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs. hidden layer size (make_circles)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "accuracy_vs_hidden_size.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: decision boundary for the best model ---
    # By mapping a grid of points and plotting the best model's predictions, 
    # we can visually see the exact non-linear shape it learned. 
    x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
    y_vals = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200)
    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))
    Z = best_clf.predict(grid_points).reshape(X_plane.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(X_plane, Y_plane, Z, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="red", marker="s", label="test")
    ax.set_title(f"Decision boundary, hidden_layer_size={best_h} (test acc={best_acc:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "decision_boundary.png", dpi=150)
    plt.close(fig)

    # --- Save model + metrics ---
    joblib.dump(best_clf, results_dir / "best_mlp_model.joblib")

    metrics = {
        "dataset": {
            "generator": "sklearn.datasets.make_circles",
            "n_samples": args.n_samples, "noise": args.noise, "factor": args.factor,
        },
        "split": {"test_size": args.test_size, "random_state": args.random_state, "stratify": True},
        "baseline_logistic_regression_test_accuracy": baseline_acc,
        "hidden_layer_sweep": sweep_results,
        "best_hidden_layer_size": best_h,
        "best_test_accuracy": best_acc,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plots, model, and metrics.json to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
