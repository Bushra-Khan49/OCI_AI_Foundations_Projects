# Project 2: Gene Expression Classification (Golub Leukemia Dataset)

While Project 1 demonstrated that neural networks easily map non-linear spaces where linear models fail, biological data often presents the exact opposite challenge.

In this project, I use the seminal **Golub et al. (1999)** leukemia gene expression dataset to compare linear and non-linear classifiers. 

## Historical Significance & Modern Relevance

**The Dataset**
- **Citation:** Golub TR, Slonim DK, Tamayo P, et al. "Molecular classification of cancer: class discovery and class prediction by gene expression monitoring." *Science*. 1999 Oct 15;286(5439):531-7.
- **Source:** Republished by Bradley Efron and Trevor Hastie in *Computer Age Statistical Inference* (Cambridge University Press, 2016), page 213. [Link to CASI Data](http://hastie.su.domains/CASI_files/DATA/leukemia_small.csv).
- **Format:** 72 leukemia patients (47 ALL, 25 AML), with 3,571 gene expression measurements per patient.

**Historical Significance**
When published in 1999, this paper was groundbreaking. It proved for the first time that cancer could be classified purely by analyzing gene expression profiles, without relying on prior biological knowledge or traditional clinical pathology. They achieved this using a weighted voting scheme that effectively acted as a linear classifier.

**Why does this matter today?**
Even with advanced modern AI, we frequently encounter $p \gg n$ datasets in bioinformatics, where the number of features (genes, proteins) vastly outnumbers the samples (patients). For example:
1. **Rare Disease Biomarker Discovery:** Sequencing 20,000 genes for a rare disease cohort of only 15 patients.
2. **Personalized Medicine Clinical Trials:** Gathering high-resolution proteomic profiles for 50 patients in a trial.

In these scenarios, complex deep learning models are highly prone to overfitting—they memorize the tiny sample set instead of learning biological signals. As a bioinformatician, understanding why classical linear models often outperform neural networks in high-dimensional biological data is a critical lesson in model selection.

## Methodology & Rigor

**Identical Preprocessing & Information Leaks**
To ensure an apples-to-apples comparison, I used a scikit-learn `Pipeline`. Every model (linear and non-linear) received the exact same standard scaling and feature selection (keeping the top 50 genes via ANOVA F-test). 

Crucially, as taught in standard Machine Learning curriculums (such as Kaggle's Intro to ML course), feature selection **must** happen *inside* the cross-validation loop. If I had filtered the top genes using the entire 72-patient dataset *before* splitting, the training phase would have gotten an unfair "sneak peek" at the validation cohort. This classic "information leak" artificially inflates accuracy and ruins the validity of biological findings.

**5-Fold Cross-Validation**
Because there are only 72 samples, doing a single 70/30 train/test split would leave only ~21 patients in the validation cohort—far too few for a reliable metric. From my learnings on robust evaluation, I used 5-fold cross-validation instead. This splits the data into 5 chunks, training on 4 and testing on 1, rotating until every patient has been evaluated as an unseen validation sample exactly once.

## Results & Interpretation

The automated script (`train_eval.py`) evaluates the models and saves the metrics and visual outputs.

### PCA Projection
Because of human limitations, we cannot visually perceive 3,571 dimensions. I used Principal Component Analysis (PCA) to compress the genes down to 2 dimensions purely for a visual sanity check, not for modeling. PCA is a deterministic, standard linear projection that captures maximum variance without requiring hyperparameter tuning.

![PCA Projection](results/pca_projection.png)
*Interpretation:* The scatter plot shows that even compressed to 2 dimensions, the ALL patients (blue) and AML patients (orange) form somewhat distinct clusters. This confirms there is a genuine biological signal separating the two cancers.

### Model Comparison
Before testing the real models, I established a "majority-class baseline" (guessing ALL every time), which achieved ~65.2% accuracy. This is the absolute floor our models must beat.

![Model Comparison](results/model_comparison.png)
*Interpretation:* The bar chart displays the 5-fold cross-validation accuracy of each model. As anticipated in this high-dimensional $p \gg n$ setting, the linear models (Logistic Regression, Linear SVM) achieved highly stable accuracies of ~96-97%. The complex non-linear MLP Neural Network showed slightly lower mean accuracy and higher variance (wider error bars). 

**Conclusion**
The extra capacity of the neural network made it prone to overfitting the 72 samples, whereas the simpler linear models generalized better to the unseen validation cohorts. This mirrors Golub et al.'s original findings and reinforces a critical lesson for modern bioinformatics: non-linearity is not a silver bullet, and linear models remain extraordinarily powerful for $p \gg n$ biological data.

## Reproducing the Results

```bash
pip install -r requirements.txt
cd data && python download_data.py && cd ..
python train_eval.py
```
Alternatively, open `notebook.ipynb` for an interactive walkthrough.
