# Project 1: Synthetic Non-Linear Classification

In this project, I explore a classic problem in machine learning: understanding the limits of linear models when dealing with non-linear data, and demonstrating how introducing a simple Multi-Layer Perceptron (MLP) neural network solves it.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Bushra-Khan49/OCI_AI_Foundations_Projects/main?filepath=01_Synthetic_Nonlinear_Classification/01_mlp_make_circles.ipynb)
*(Note: If you are viewing this via Binder, please run all the cells to set up the environment before experimenting with the interactive decision boundary at the bottom.)*

## The Data and the Problem

I used a synthetic dataset called `make_circles` from scikit-learn. This generator creates data points in the shape of two interlocking, concentric circles. By design, you cannot draw a single straight line to separate the inner circle from the outer one. It is a perfectly balanced binary classification problem (50% class 0, 50% class 1), making it ideal for testing a model's ability to warp its decision boundary.

Before modeling, I held out 30% of this data as a **validation cohort**. This is a crucial step; evaluating a model on the same data it learned from leads to a false sense of security due to overfitting. By using an unseen validation cohort, I ensure that the model actually generalizes to new data rather than just memorizing the training set.

## Methodology & Identical Preprocessing

To prove that a linear model struggles here, I first fit a standard Logistic Regression baseline. Logistic Regression attempts to find a linear decision boundary (a flat plane). 

Next, I introduced an MLP neural network. I swept through different hidden layer sizes (from 1 to 50 neurons) to observe how adding capacity allows the network to combine multiple flat boundaries into a smooth, curved shape. 

Both models were trained and evaluated on the exact same train/validation splits, ensuring an apples-to-apples comparison. We are strictly comparing the architectural capability of the models themselves.

## Results & Interpretation

The automated script (`train_classifier.py`) evaluates the models and saves the metrics and visual outputs.

### Model Comparison
As expected, the Logistic Regression baseline completely failed. It achieved an accuracy of roughly **45.6%** on the validation cohort—which is effectively no better than random guessing for a balanced dataset. It simply cannot cut a circle with a straight line.

However, as I added neurons to the hidden layer of the MLP, performance drastically improved:
- **1-2 neurons**: ~50% accuracy (still struggling to enclose the circle).
- **3 neurons**: ~67.8% accuracy (beginning to form a triangle-like boundary).
- **5+ neurons**: **100% accuracy**.

The MLP achieved perfect classification on the unseen validation cohort once it had 5 hidden neurons. 

### Visualizing the Decision Boundary
By mapping a grid of points and plotting the best model's predictions, we can visually see the exact non-linear shape it learned. 

![Decision Boundary](results/decision_boundary.png)

*Interpretation*: The image shows the contour map of the neural network's decision space. The inner circle is perfectly encapsulated by the boundary the MLP learned. The model didn't just memorize the points; the smooth, continuous boundary proves it learned the underlying geometric distribution of the data. 

### Why This Matters
In a real-world, day-to-day context, many relationships are not linear. Whether we are predicting customer churn based on complex behavioral interactions or diagnosing a disease from non-linear biomarkers, using a strictly linear model can lead to total failure (as seen by our 45% baseline). This project demonstrates the power of neural networks to map non-linear spaces, validating why they are the foundation of modern deep learning.

## Reproducing the Results

You can reproduce the exact results and generate the plots by running the automated script. I programmed the script to automatically save the metrics to disk for future reference, allowing us to log experiments rigorously.

```bash
pip install -r requirements.txt
python train_classifier.py
```

Alternatively, open `01_mlp_make_circles.ipynb` (or click the Binder link above) for an interactive walkthrough.
