import matplotlib
matplotlib.use('Agg')
# Import json for saving metrics
import json
# Import Path for cross-platform file paths
from pathlib import Path

# Import numpy for numerical operations
import numpy as np
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import make_circles to generate synthetic non-linear data
from sklearn.datasets import make_circles
# Import train_test_split to divide data into training and validation sets
from sklearn.model_selection import train_test_split
# Import LogisticRegression for our linear baseline model
from sklearn.linear_model import LogisticRegression
# Import MLPClassifier for our non-linear neural network model
from sklearn.neural_network import MLPClassifier
# Import evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the directory to save output artifacts
RESULTS_DIR = Path("results")
# Create the directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

# Define a fixed random seed used everywhere below for absolute reproducibility
RANDOM_STATE = 0

# Generate 300 data points in the shape of two concentric circles with slight noise
X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=RANDOM_STATE)

# Hold out 30% of the data for testing, stratifying by class to maintain balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# Print the size of our training and testing cohorts
print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Initialize a standard Logistic Regression model (linear decision boundary)
baseline = LogisticRegression()
# Fit the linear model to our training data
baseline.fit(X_train, y_train)
# Calculate accuracy by evaluating predictions on the held-out test data
baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

# Print the baseline accuracy (expected to be around 50% since data is circular)
print(f"Logistic Regression (linear baseline) test accuracy: {baseline_acc:.3f}")
# Print a detailed classification report
print(classification_report(y_test, baseline.predict(X_test)))

# Initialize a Multi-Layer Perceptron neural network with a single hidden layer of 10 neurons
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=3000, random_state=1)
# Fit the neural network to the training data
mlp.fit(X_train, y_train)
# Calculate accuracy on the held-out test data
mlp_acc = accuracy_score(y_test, mlp.predict(X_test))

# Print the MLP's accuracy (expected to be much higher than the linear baseline)
print(f"MLP (hidden_layer_size=10) test accuracy: {mlp_acc:.3f}")
# Print a detailed classification report
print(classification_report(y_test, mlp.predict(X_test)))
# Print the confusion matrix to see false positives/negatives
print("Confusion matrix:\n", confusion_matrix(y_test, mlp.predict(X_test)))

# Define a list of hidden layer sizes to test (from 1 neuron up to 50 neurons)
hidden_sizes = [1, 2, 3, 5, 10, 20, 50]
# Initialize an empty list to store the results of our sweep
sweep_results = []

# Iterate over each hidden layer size
for h in hidden_sizes:
    # Initialize a new MLP model with the current hidden layer size
    clf = MLPClassifier(hidden_layer_sizes=(h,), activation='relu', max_iter=3000, random_state=1)
    # Train the model
    clf.fit(X_train, y_train)
    # Evaluate accuracy on the test set
    acc = accuracy_score(y_test, clf.predict(X_test))
    # Append the configuration and result to our tracking list
    sweep_results.append({"hidden_layer_size": h, "test_accuracy": acc})
    # Print the progress to the console
    print(f"hidden_layer_size={h:>3}  ->  test accuracy={acc:.3f}")

# Create a new matplotlib figure
fig, ax = plt.subplots(figsize=(6, 4))
# Extract the hidden layer sizes for the X-axis
sizes = [r["hidden_layer_size"] for r in sweep_results]
# Extract the corresponding test accuracies for the Y-axis
accs = [r["test_accuracy"] for r in sweep_results]
# Plot the MLP accuracy curve
ax.plot(sizes, accs, marker='o', label='MLP')
# Draw a horizontal red dashed line representing the linear baseline accuracy
ax.axhline(baseline_acc, color='red', linestyle='--', label=f'Logistic Regression baseline ({baseline_acc:.2f})')
# Add labels and title
ax.set_xlabel('Hidden layer size (neurons)')
ax.set_ylabel('Test accuracy')
ax.set_title('Test accuracy vs. hidden layer size (make_circles)')
# Set the y-axis limits from 0 to slightly over 1
ax.set_ylim(0, 1.05)
# Add a legend
ax.legend()
# Adjust layout to prevent clipping
fig.tight_layout()
# Save the figure to the results directory
fig.savefig(RESULTS_DIR / 'accuracy_vs_hidden_size.png', dpi=150)
# Display the plot inline
plt.show()

# Find the best performing model configuration from our sweep
best = max(sweep_results, key=lambda r: r["test_accuracy"])
# Extract the optimal hidden layer size
best_h = best["hidden_layer_size"]

# Re-initialize and train the model using the optimal capacity
clf = MLPClassifier(hidden_layer_sizes=(best_h,), activation='relu', max_iter=3000, random_state=1)
clf.fit(X_train, y_train)

# Create a dense grid of points across the feature space for visualization
x_vals = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 200)
y_vals = np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 200)
X_plane, Y_plane = np.meshgrid(x_vals, y_vals)
# Flatten the grid into a list of (x,y) coordinates
grid_points = np.column_stack((X_plane.ravel(), Y_plane.ravel()))
# Predict the class for every single point on the grid to map the decision boundary
Z = clf.predict(grid_points).reshape(X_plane.shape)

# Create a new matplotlib figure
fig, ax = plt.subplots(figsize=(6, 5))
# Draw the filled contour representing the model's decision regions
ax.contourf(X_plane, Y_plane, Z, alpha=0.3)
# Scatter plot the training data points in black
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', label='train')
# Scatter plot the test data points as red squares
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='red', marker='s', label='test')
# Add a title indicating the optimal hidden size and its accuracy
ax.set_title(f'Decision boundary, hidden_layer_size={best_h} (test acc={best["test_accuracy"]:.2f})')
# Add a legend
ax.legend()
# Adjust layout
fig.tight_layout()
# Save the decision boundary plot to disk
fig.savefig(RESULTS_DIR / 'decision_boundary.png', dpi=150)
# Display the plot inline
plt.show()

# Import necessary ipywidgets for interactive sliders
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interactive

# Define the function that will be called whenever the slider moves
def update_plot(hidden_layer_size):
    # Initialize and train an MLP with the dynamically selected hidden layer size
    clf = MLPClassifier(hidden_layer_sizes=(hidden_layer_size,), activation='relu', max_iter=3000, random_state=1)
    clf.fit(X_train, y_train)
    # Evaluate accuracy
    acc = accuracy_score(y_test, clf.predict(X_test))

    # Predict over the grid to find the decision boundary
    Z = clf.predict(grid_points).reshape(X_plane.shape)
    # Clear the previous plot
    plt.clf()
    # Draw the new contour
    plt.contourf(X_plane, Y_plane, Z, alpha=0.3)
    # Scatter all data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    # Update the title with the live metrics
    plt.title(f'Hidden layer size: {hidden_layer_size}  |  test accuracy: {acc:.2f}')
    # Display the plot
    plt.show()

# Create the interactive slider widget (from 1 to 50 neurons)
interactive_plot = interactive(update_plot, hidden_layer_size=widgets.IntSlider(min=1, max=50, step=1, value=10))
# Render the interactive widget in the notebook
update_plot(10)

# Compile all experimental parameters and results into a single dictionary
metrics = {
    "dataset": "sklearn.datasets.make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)",
    "train_test_split": "70/30, stratified, random_state=0",
    "baseline_logistic_regression_test_accuracy": baseline_acc,
    "mlp_fixed_hidden10_test_accuracy": mlp_acc,
    "hidden_layer_sweep": sweep_results,
    "best_hidden_layer_size": best_h,
    "best_test_accuracy": best["test_accuracy"],
}

# Write the dictionary to metrics.json on disk
with open(RESULTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Print the metrics JSON nicely formatted to stdout
print(json.dumps(metrics, indent=2))
