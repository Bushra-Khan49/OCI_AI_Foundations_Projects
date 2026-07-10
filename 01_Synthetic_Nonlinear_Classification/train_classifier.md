# Train Classifier (train_classifier.py)

Standalone, reproducible version of the make_circles classification experiment.

**Note:** This file represents the standalone Python script `train_classifier.py` converted into a Markdown walkthrough. While the Jupyter Notebook (`01_mlp_make_circles.ipynb`) is designed for interactive data exploration and step-by-step visualization, this script is structured for automated, reproducible command-line execution (e.g., using `argparse` for parameter sweeps and saving artifacts to a results directory). Both share the same underlying logic.

Unlike the original version, this script:
  - splits data into train/test and reports accuracy on held-out data
  - fits a linear baseline (LogisticRegression) alongside the MLP
  - sweeps hidden layer size and records test accuracy at each step
  - saves a trained model, plots, and a metrics.json file to `--results-dir`

In this project, I explore the limits of linear models when dealing with non-linear data, 
and demonstrate how introducing a simple Multi-Layer Perceptron (MLP) neural network solves it.

**Usage:**
```bash
python train_classifier.py
python train_classifier.py --hidden-sizes 1 5 10 20 50 --noise 0.05 --results-dir results
```

## Step 1: Import libraries and setup CLI
First, I import the necessary libraries for data processing, machine learning models, and visualization. I also set up `argparse` to allow running this script from the terminal with custom hyperparameters, making it highly reproducible.

```python
# Import argparse for CLI usage
import argparse
# Import json for saving metrics
import json
# Import Path for cross-platform file paths
from pathlib import Path

# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting
import matplotlib
# We use the Agg backend so matplotlib doesn't try to open a GUI window during automated script execution
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Import joblib to save trained machine learning models
import joblib

# Import make_circles to generate synthetic non-linear data
from sklearn.datasets import make_circles
# Import train_test_split to safely divide data into training and validation sets
from sklearn.model_selection import train_test_split
# Import LogisticRegression for our linear baseline model
from sklearn.linear_model import LogisticRegression
# Import MLPClassifier for our non-linear neural network model
from sklearn.neural_network import MLPClassifier
# Import evaluation metrics
from sklearn.metrics import accuracy_score, classification_report


def parse_args():
    # Setup argparse for CLI usage so users can sweep parameters easily
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
    # Parse CLI arguments passed by the user
    args = parse_args()
    
    # Create the results directory if it doesn't already exist
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
```

## Step 2: Generating Data
`make_circles` generates data points in the shape of two interlocking, concentric circles. By design, you cannot draw a single straight line to separate the inner circle from the outer one. This synthetic dataset perfectly isolates the concept of linear vs. non-linear separability.

```python
    print("Generating dataset...")
    # Generate data points in the shape of two concentric circles with slight noise
    X, y = make_circles(
        n_samples=args.n_samples, noise=args.noise, factor=args.factor,
        random_state=args.random_state,
    )
```

## Step 3: Validation Cohort
I hold out a portion of this data as a validation cohort. Evaluating a model on the same data it learned from leads to a false sense of security due to overfitting. We must strictly measure performance on data the model has never seen before.

```python
    # Hold out a portion of the data for testing, stratifying by class to maintain balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    
    # Print the size of our training and testing cohorts to stdout
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
```

## Step 4: Linear Baseline
I fit a standard Logistic Regression baseline. Logistic Regression attempts to find a linear decision boundary (a flat plane). Because our data is circular, we expect this model to fail entirely, achieving roughly 50% (random guess) accuracy.

```python
    print("\nFitting linear baseline (LogisticRegression)...")
    
    # Initialize a standard Logistic Regression model (linear decision boundary)
    baseline = LogisticRegression()
    # Fit the linear model to our training data
    baseline.fit(X_train, y_train)
    
    # Calculate accuracy by evaluating predictions on the held-out test data
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    
    # Print the baseline accuracy (expected to be around 50% since data is circular)
    print(f"  Baseline test accuracy: {baseline_acc:.3f}")
```

## Step 5: MLP Sweep
Next, I introduce an MLP neural network. I sweep through different hidden layer sizes to observe how adding capacity allows the network to combine multiple flat boundaries into a smooth, curved shape. Both models use the exact same validation splits for a fair, apples-to-apples comparison.

```python
    print("\nSweeping MLP hidden layer size...")
    # Initialize an empty list to store the results of our sweep
    sweep_results = []
    
    # Initialize tracking variables for the best model configuration
    best_clf, best_acc, best_h = None, -1.0, None
    
    # Iterate over each hidden layer size defined in our CLI arguments
    for h in args.hidden_sizes:
        # Initialize a new MLP model with the current hidden layer size
        clf = MLPClassifier(hidden_layer_sizes=(h,), activation="relu", max_iter=3000, random_state=1)
        # Train the model
        clf.fit(X_train, y_train)
        # Evaluate accuracy on the test set
        acc = accuracy_score(y_test, clf.predict(X_test))
        
        # Append the configuration and result to our tracking list
        sweep_results.append({"hidden_layer_size": h, "test_accuracy": acc})
        # Print the progress to the console
        print(f"  hidden_layer_size={h:>3}  ->  test accuracy={acc:.3f}")
        
        # If this is the best accuracy we've seen so far, save the model state
        if acc > best_acc:
            best_clf, best_acc, best_h = clf, acc, h

    # Print the absolute best configuration found during the sweep
    print(f"\nBest hidden layer size: {best_h} (test accuracy={best_acc:.3f})")
    
    # Print a detailed classification report to see precision, recall, and f1-score
    print("\nClassification report for best model:")
    print(classification_report(y_test, best_clf.predict(X_test)))
```

## Step 6: Plot 1 - Accuracy vs. hidden layer size
We plot the performance curve to visually inspect how accuracy improves as we add neurons to the network's hidden layer. As expected, a single neuron performs as poorly as the linear baseline, but as capacity increases, the network quickly learns to wrap around the data perfectly.

```python
    # Extract the hidden layer sizes for the X-axis
    sizes = [r["hidden_layer_size"] for r in sweep_results]
    # Extract the corresponding test accuracies for the Y-axis
    accs = [r["test_accuracy"] for r in sweep_results]
    
    # Create a new matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot the MLP accuracy curve
    ax.plot(sizes, accs, marker="o", label="MLP")
    # Draw a horizontal red dashed line representing the linear baseline accuracy
    ax.axhline(baseline_acc, color="red", linestyle="--", label=f"Logistic Regression ({baseline_acc:.2f})")
    
    # Add labels and title
    ax.set_xlabel("Hidden layer size (neurons)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy vs. hidden layer size (make_circles)")
    # Set the y-axis limits from 0 to slightly over 1
    ax.set_ylim(0, 1.05)
    # Add a legend
    ax.legend()
    # Adjust layout to prevent clipping
    fig.tight_layout()
    # Save the figure to the results directory
    fig.savefig(results_dir / "accuracy_vs_hidden_size.png", dpi=150)
    plt.close(fig)
```

![Accuracy Curve](results/accuracy_vs_hidden_size.png)

## Step 7: Plot 2 - Decision boundary for the best model
By mapping a dense grid of points across the feature space and plotting the best model's predictions, we can visually inspect the exact non-linear shape it learned. The neural network successfully carves out a curved boundary to separate the inner ring from the outer ring!

```python
    # Create a dense grid of points across the feature space for visualization
    x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
    y_vals = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200)
    X_plane, Y_plane = np.meshgrid(x_vals, y_vals)
    
    # Flatten the grid into a list of (x,y) coordinates
    grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))
    
    # Predict the class for every single point on the grid to map the decision boundary
    Z = best_clf.predict(grid_points).reshape(X_plane.shape)

    # Create a new matplotlib figure
    fig, ax = plt.subplots(figsize=(6, 5))
    # Draw the filled contour representing the model's decision regions
    ax.contourf(X_plane, Y_plane, Z, alpha=0.3)
    # Scatter plot the training data points in black
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", label="train")
    # Scatter plot the test data points as red squares
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="red", marker="s", label="test")
    
    # Add a title indicating the optimal hidden size and its accuracy
    ax.set_title(f"Decision boundary, hidden_layer_size={best_h} (test acc={best_acc:.2f})")
    # Add a legend
    ax.legend()
    # Adjust layout
    fig.tight_layout()
    # Save the decision boundary plot to disk
    fig.savefig(results_dir / "decision_boundary.png", dpi=150)
    plt.close(fig)
```

![Decision Boundary](results/decision_boundary.png)

## Step 8: Save model & metrics
We save the best trained neural network object (`.joblib`) to disk so it can be loaded into an API or production service later without retraining. We also serialize the experimental parameters and test metrics to `metrics.json` to ensure this run is permanently tracked and perfectly reproducible.

```python
    # Serialize the best scikit-learn model object to disk using joblib
    joblib.dump(best_clf, results_dir / "best_mlp_model.joblib")

    # Compile all experimental parameters and results into a single dictionary
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
    
    # Write the dictionary to metrics.json on disk
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Inform the user that the pipeline has completed successfully
    print(f"\nSaved plots, model, and metrics.json to: {results_dir.resolve()}")


## Step 9: Final Conclusion
print("\n================ FINAL CONCLUSION ================")
print("This project successfully demonstrated the fundamental flaw of linear models.")
print("When presented with non-linear, circular data, Logistic Regression failed completely.")
print("However, by adding a hidden layer of neurons (a Multi-Layer Perceptron), the model")
print("was able to combine multiple linear transformations to create a highly flexible,")
print("non-linear decision boundary, perfectly separating the two classes!")
print("==================================================\n")

# Boilerplate execution guard to allow the script to be imported safely without executing main()
if __name__ == "__main__":
    main()
```
