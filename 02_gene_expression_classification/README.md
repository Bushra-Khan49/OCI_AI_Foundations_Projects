# Project 2: Gene Expression Classification (Golub Leukemia Dataset)

Follow-on from Project 1. Project 1 showed that a linear model fails and an MLP succeeds on `make_circles` — clean, synthetic, low-dimensional data. This project asks the same linear-vs-non-linear question on **real biological data**, where the answer is not guaranteed to come out the same way.

## Dataset & Historical Significance

- **Golub TR, Slonim DK, Tamayo P, et al.** "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring." *Science*. 1999;286(5439):531-7.
- 72 leukemia patients, labeled ALL (Acute Lymphoblastic Leukemia, n=47) or AML (Acute Myeloid Leukemia, n=25).
- 3,571 gene expression measurements per patient.
- Source file: http://hastie.su.domains/CASI_files/DATA/leukemia_small.csv (republished by Efron & Hastie, *Computer Age Statistical Inference*, page 213).

**Historical Significance:** When published in 1999, this paper was groundbreaking. It proved for the first time that cancer could be classified purely by analyzing gene expression profiles, without relying on prior biological knowledge or traditional clinical pathology. They achieved this using a weighted voting scheme that effectively acted as a linear classifier.

**Dataset Rationale & Modern Relevance:** It's the opposite regime from `make_circles`. There, 300 samples and 2 features made a neural network the obvious right tool. Here, 72 samples and 3,571 features (**p ≫ n**) is a setting where extra model flexibility can just as easily mean overfitting. 

Even today, we frequently encounter $p \gg n$ datasets in bioinformatics, where features vastly outnumber samples. For example:
1. **Rare Disease Biomarker Discovery:** Sequencing 20,000 genes for a cohort of only 15 patients.
2. **Personalized Medicine Clinical Trials:** Gathering high-resolution proteomic profiles for 50 patients.

In these scenarios, complex deep learning models memorize the tiny sample set instead of learning biological signals. Testing the "neural nets are needed for non-linear biological data" claim requires a case where it's allowed to fail.

## Method

1. **Baseline:** majority-class ("always predict ALL") accuracy via 5-fold CV — the floor every real model has to beat.
2. **PCA** to 2D, purely for visual inspection of class separation — not used for modeling. *Context:* Because of human limitations, we cannot visually perceive 3,571 dimensions. PCA is a deterministic, standard linear projection that captures maximum variance without requiring hyperparameter tuning.
   - **PCA Projection Interpretation:**

     ![PCA Projection](results/pca_projection.png)

     In the resulting plot, the ALL and AML patients form distinct clusters but overlap in the center. This indicates that while there is strong biological signal separating the leukemias, they aren't trivially separable in 2D space. The models must leverage higher-dimensional space.
3. **Feature selection:** top genes by ANOVA F-test, done **inside** the cross-validation loop (via an sklearn `Pipeline`), not before it. *Context:* As taught in standard ML curriculums (like Kaggle's Intro to ML), feature selection *must* happen inside the loop. Fitting feature selection on the full dataset before splitting gives the training phase an unfair "sneak peek" at the validation cohort, which is a classic information leak.
4. **Models compared** (all with identical preprocessing to ensure apples-to-apples architectural comparison):
   - Logistic Regression (linear)
   - Linear SVM (linear)
   - RBF-kernel SVM (non-linear)
   - MLP, one hidden layer of 10 units (non-linear)
5. **Evaluation:** stratified 5-fold cross-validation, not a single train/test split. *Context:* With only 72 samples, doing a single 70/30 split would leave only ~21 patients in the validation cohort. One split's result depends too much on which patients happened to land in the test fold. 5-fold CV ensures a robust evaluation.

## Results

```text
Loaded 72 samples, 3571 genes.
Class balance: ALL=47, AML=25

Majority-class baseline: 0.652 +/- 0.012

Comparing models with 5-fold CV, top 50 genes per fold:
  logistic_regression        0.971 +/- 0.057
  linear_svm                 0.957 +/- 0.057
  rbf_svm                    0.971 +/- 0.057
  mlp_10_hidden              0.944 +/- 0.053
```

![Model Comparison](results/model_comparison.png)

As anticipated, with 72 samples and 3,571 genes, linear models (Logistic Regression, Linear SVM) perform just as well as non-linear models (RBF SVM), while the MLP shows slightly higher variance and lower mean accuracy. This demonstrates the critical lesson that non-linearity is not a silver bullet in high-dimensional biological data. The extra capacity of the neural network made it prone to overfitting the 72 samples, whereas simpler linear models generalized better.

## What's in this folder

```
02_gene_expression_classification/
  README.md
  data/
    download_data.py       <- downloads + reshapes the raw Golub dataset
    leukemia_clean.csv     <- generated by download_data.py (not committed)
  02_gene_expression_pipeline.ipynb            <- full walkthrough with PCA plot + CV comparison
  train_eval.py              <- CLI script version, no notebook required
  requirements.txt
  results/                   <- generated by running either of the above
    metrics.json
    pca_projection.png
    model_comparison.png
```

## Reproducing this

```bash
pip install -r requirements.txt
cd data && python download_data.py && cd ..
python train_eval.py                                  # default: top 50 genes, 5-fold CV
python train_eval.py --n-features 100 --n-splits 10   # custom settings
```

or open `02_gene_expression_pipeline.ipynb` in Jupyter for the interactive walkthrough.

## Limitations

- 3,571 genes with 72 samples means feature selection and cross-validation are load-bearing — a single train/test split (as in Project 1) would give a misleadingly confident or misleadingly poor result depending on luck.
- ANOVA F-test feature selection is univariate; it can miss genes that only matter in combination with others. A follow-up could compare against L1-regularized logistic regression (which does selection and modeling jointly).
- This dataset is a historical benchmark, not current clinical practice.

## Course context

Extends concepts from the **Oracle Cloud Infrastructure (OCI) AI Foundations Associate** course (Modules 3-4: Machine Learning Foundations, Deep Learning Foundations) by testing them against real, high-dimensional biological data rather than only the synthetic exercise data provided in the course.
