# Binary Classification using Neural Networks on make_circles Dataset  
**Project from Oracle Cloud Infrastructure (OCI) AI Foundations Course**

![preview](https://raw.githubusercontent.com/yourusername/oci-make-circles-classifier/main/preview.png)

### About This Project

This is one of the hands-on projects I completed as part of the **Oracle Cloud Infrastructure (OCI) AI Foundations Course** (November 2025).

The dataset used is **`make_circles`** from scikit-learn — a classic synthetic dataset consisting of two interlocking half-circles (1,000 points total).  
These two classes are **not linearly separable**, meaning no straight line (or any linear model) can separate them correctly.

The goal of the exercise was to:
- Train a **Multi-Layer Perceptron (MLP)** neural network on this data
- Visualize how the decision boundary evolves during training
- Observe how even a small neural network can learn complex, curved boundaries

The course provided the complete implementation, including the beautiful real-time animation that shows the decision boundary gradually changing from a simple line into a perfect circle that wraps around the inner cluster.

### Why This Example is Important (Especially for Bioinformatics)

In real-world biological and medical data, patterns are almost never linear:

- Gene expression profiles in diseased vs healthy tissues  
- Clustering of cell types in single-cell RNA-seq  
- Protein-ligand binding prediction  
- Patient stratification in complex diseases  

These often form non-linear manifolds in high-dimensional space — very similar to the make_circles pattern.

This simple example clearly shows why we need neural networks in bioinformatics:  
**Biological data is inherently non-linear, and neural networks are one of the few models capable of capturing such complexity.**

### What I Learned from This Project

- How hidden layers with non-linear activations (ReLU) enable modeling of complex patterns  
- The power of backpropagation in iteratively refining decision boundaries  
- Practical use of `MLPClassifier` with `warm_start=True` and `max_iter=1` for step-by-step training  
- Creating decision boundary visualizations using meshgrid and contour plots  
- Generating smooth animations with `matplotlib.animation.FuncAnimation`  
- Importance of noise, factor, and dataset structure in classification difficulty  

### Key Insight

Even a very small neural network (one hidden layer with 10–20 neurons) can perfectly separate highly non-linear data that completely defeats linear models.

This single project gave me strong intuition for when to reach for neural networks instead of traditional ML algorithms — a crucial skill in modern bioinformatics and healthcare AI.

**Course:** Oracle Cloud Infrastructure (OCI) AI Foundations  
**Completed:** November 11, 2025  
**Location:** India

Simple, elegant, and extremely effective teaching example from Oracle.  
Highly recommended for anyone starting with neural networks.

Feel free to run the notebook — watching the boundary evolve is genuinely satisfying!
