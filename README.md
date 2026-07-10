# OCI AI Foundations — Projects

A portfolio of hands-on machine learning and computational biology projects.
Each project is designed as an actual, evaluated, reproducible scientific 
pipeline, demonstrating rigorous model evaluation and statistical validation.

## Projects

| # | Project | Question | Key result |
|---|---|---|---|
| 1 | [Linear vs. MLP on `make_circles`](01_Synthetic_Nonlinear_Classification/) | Does a linear model actually fail on non-linear synthetic data, and does a small neural net actually fix it? | Logistic Regression: 0.456 test accuracy. MLP (≥5 hidden neurons): 1.000. <br> [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Bushra-Khan49/OCI_AI_Foundations_Projects/main?filepath=01_Synthetic_Nonlinear_Classification/01_mlp_make_circles.ipynb) |
| 2 | [Gene expression classification (Golub leukemia)](02_gene_expression_classification/) | Does the same linear-vs-non-linear finding hold on real, high-dimensional (p≫n) biological data? | See project README — run `train_eval.py` to reproduce; results are not assumed to favor either model. |

More projects will be added here as they're completed.


## Repository structure

Each project is self-contained:
```
<NN>_project_name/
  README.md            <- question, method, results, how to reproduce
  *.ipynb               <- interactive notebook version
  *.py                  <- CLI script version (no notebook required)
  requirements.txt       <- project-specific dependencies
  data/                  <- present only if the project needs external data
  results/               <- metrics.json + plots, committed as proof of output
```

## Setup

Using conda (recommended, covers all projects):
```bash
conda env create -f environment.yml
conda activate oci-ai-foundations
```

Or, per-project, with pip:
```bash
cd 01_Synthetic_Nonlinear_Classification
pip install -r requirements.txt
```

Then either open the `.ipynb` in Jupyter, or run the `.py` script directly —
see each project's README for exact commands.



## License

MIT — see [LICENSE](LICENSE).
