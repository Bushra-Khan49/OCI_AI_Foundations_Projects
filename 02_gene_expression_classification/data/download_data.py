"""
download_data.py

Downloads the Golub et al. (1999) leukemia gene expression dataset
(AML vs. ALL) from Stanford's CASI book data page and saves a clean,
sklearn-ready CSV locally.

Source: http://hastie.su.domains/CASI_files/DATA/leukemia_small.csv
Citation: Golub TR, Slonim DK, Tamayo P, et al. "Molecular classification
of cancer: class discovery and class prediction by gene expression
monitoring." Science. 1999 Oct 15;286(5439):531-7.
Republished by Efron & Hastie in "Computer Age Statistical Inference" (CASI).

Raw file format (confirmed by inspection):
  - Row 1: 72 comma-separated class labels ("ALL" or "AML"), one per sample
  - Rows 2-3572: 3571 genes, each row has 72 expression values (one per sample)
  i.e. the raw file is (genes x samples). We transpose it to the standard
  (samples x genes) orientation expected by scikit-learn before saving.

Usage:
    python download_data.py                       # saves to ./leukemia_clean.csv
    python download_data.py --out-path custom.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import urllib.request

SOURCE_URL = "http://hastie.su.domains/CASI_files/DATA/leukemia_small.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Download and clean the Golub leukemia dataset.")
    p.add_argument("--out-path", type=str, default="leukemia_clean.csv",
                    help="Where to save the cleaned (samples x genes + label) CSV.")
    p.add_argument("--url", type=str, default=SOURCE_URL,
                    help="Source URL for the raw leukemia_small.csv file.")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_path)

    print(f"Downloading raw data from {args.url} ...")
    try:
        with urllib.request.urlopen(args.url, timeout=30) as resp:
            raw_text = resp.read().decode("utf-8")
    except Exception as e:
        print(f"ERROR: could not download the dataset ({e}).", file=sys.stderr)
        print("If this environment has no internet access, download the file manually", file=sys.stderr)
        print(f"from {args.url} and place it as 'leukemia_small_raw.csv' in this folder,", file=sys.stderr)
        print("then re-run with: python download_data.py --url file://<path>", file=sys.stderr)
        sys.exit(1)

    lines = raw_text.strip().splitlines()
    labels = [x.strip() for x in lines[0].split(",")]
    n_samples = len(labels)
    print(f"Found {n_samples} samples. Label counts: "
          f"ALL={labels.count('ALL')}, AML={labels.count('AML')}")

    # Remaining rows: one per gene, n_samples expression values each
    gene_rows = [line.split(",") for line in lines[1:]]
    n_genes = len(gene_rows)
    print(f"Found {n_genes} gene expression rows.")

    # Build a (genes x samples) DataFrame, then transpose to (samples x genes)
    expr = pd.DataFrame(gene_rows, dtype=float)
    expr.index = [f"gene_{i:04d}" for i in range(n_genes)]
    expr_samples_by_genes = expr.T  # now rows=samples, columns=genes
    expr_samples_by_genes.insert(0, "label", labels)
    expr_samples_by_genes.reset_index(drop=True, inplace=True)

    expr_samples_by_genes.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset ({expr_samples_by_genes.shape[0]} samples x "
          f"{expr_samples_by_genes.shape[1]-1} genes) to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
