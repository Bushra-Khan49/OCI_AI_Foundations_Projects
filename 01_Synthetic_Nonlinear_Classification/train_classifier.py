"""
train_classifier.py

In this project, I use a synthetic dataset called `make_circles` to illustrate a fundamental concept in machine learning: the limitation of linear models when dealing with non-linear data, and how a simple Multi-Layer Perceptron (MLP) neural network overcomes this. 

This script is the fully reproducible, automated version of that experiment. It generates the data, fits both a linear baseline and a series of neural networks, evaluates them, and saves the results so they can be reviewed later without needing to rerun everything.

Usage:
    python train_classifier.py
    python train_classifier.py --hidden-sizes 1 5 10 20 50 --noise 0.05 --results-dir results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # We use the 'Agg' backend because we only want to save figures to disk, not display them in a GUI.
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
    p.add_argument("--test-size", type=float, default=0.3, help="Fraction of data held out as our validation cohort.")
    p.add_argument("--random-state", type=int, default=0, help="Random seed for data generation and splitting.")
    p.add_argument(
        "--hidden-sizes", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 50],
        help="Hidden layer sizes to sweep over."
    )
    p.add_argument("--results-dir", type=str, default="results", help="Directory to save output files.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # First, I generate the synthetic dataset. 
    # I chose `make_circles` because it creates two concentric circles of data points. 
    # This is a classic non-linear problem: you cannot draw a single straight line to separate the inner circle from the outer circle.
    print("Generating dataset...")
    X, y = make_circles(
        n_samples=args.n_samples, noise=args.noise, factor=args.factor,
        random_state=args.random_state,
    )
    
    # Next, I split the data into a training set and an unseen validation cohort (test set).
    # This is critical. Evaluating a model on the same data it learned from leads to a false sense of security (overfitting).
    # We must measure performance on data the model has never seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    print(f"Train: {len(X_train)} samples | Validation: {len(X_test)} samples")

    # To prove that a linear model struggles here, I train a Logistic Regression baseline.
    # Logistic Regression attempts to find a linear decision boundary.
    print("\nFitting linear baseline (LogisticRegression)...")
    baseline = LogisticRegression()
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    print(f"  Baseline validation accuracy: {baseline_acc:.3f}")
    # I expect this accuracy to be around 0.5 (random guessing), because a straight line simply cannot separate a circle.

    # Now, I introduce non-linearity using a Multi-Layer Perceptron (MLP) neural network.
    # I iterate through different hidden layer sizes to observe how the network's capacity to learn complex shapes increases as we add neurons.
    print("\nSweeping MLP hidden layer size...")
    sweep_results = []
    best_clf, best_acc, best_h = None, -1.0, None
    for h in args.hidden_sizes:
        clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", max_iter=3000, random_state=1)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        sweep_results.append({"hidden_layer_size": h, "validation_accuracy": acc})
        print(f"  hidden_layer_size={h:>3}  ->  validation accuracy={acc:.3f}")
        
        # Keep track of the highest performing model to save later.
        if acc > best_acc:
            best_clf, best_acc, best_h = clf, acc, h

    print(f"\nBest hidden layer size: {best_h} (validation accuracy={best_acc:.3f})")
    print("\nClassification report for best model:")
    print(classification_report(y_test, best_clf.predict(X_test)))

    # Now I create visualizations to make the results easily interpretable.
    # The first plot shows how validation accuracy improves as we increase the number of neurons, comparing it against the flat linear baseline.
    sizes = [r["hidden_layer_size"] for r in sweep_results]
    accs = [r["validation_accuracy"] for r in sweep_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, accs, marker="o", label="MLP")
    ax.axhline(baseline_acc, color="red", linestyle="--", label=f"Logistic Regression ({baseline_acc:.2f})")
    ax.set_xlabel("Hidden layer size (neurons)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy vs. hidden layer size")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "accuracy_vs_hidden_size.png", dpi=150)
    plt.close(fig)

    # The second plot is a contour map showing the actual decision boundary the best neural network learned.
    # By mapping a grid of points and asking the model to predict their class, we can visually see the non-linear shape (often a polygon or circle) the MLP drew to separate the classes.
    x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
    y_vals = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200)
    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))
    Z = best_clf.predict(grid_points).reshape(X_plane.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(X_plane, Y_plane, Z, alpha=0.3)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", label="train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="red", marker="s", label="validation")
    ax.set_title(f"Decision boundary (Neurons={best_h}, Acc={best_acc:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "decision_boundary.png", dpi=150)
    plt.close(fig)

    # Finally, I save the best trained model and the metrics to disk.
    # Saving outputs is standard practice in data science pipelines. It ensures that the experiment is reproducible, and the trained model can be reloaded for future predictions or deployments without having to retrain it from scratch.
    joblib.dump(best_clf, results_dir / "best_mlp_model.joblib")

    metrics = {
        "dataset": {
            "generator": "sklearn.datasets.make_circles",
            "n_samples": args.n_samples, "noise": args.noise, "factor": args.factor,
        },
        "split": {"test_size": args.test_size, "random_state": args.random_state, "stratify": True},
        "baseline_logistic_regression_validation_accuracy": baseline_acc,
        "hidden_layer_sweep": sweep_results,
        "best_hidden_layer_size": best_h,
        "best_validation_accuracy": best_acc,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plots, model, and metrics.json to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
