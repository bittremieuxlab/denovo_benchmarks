"""Shared data loading for ranking visualizations.

Reads all peptide_precision_plot_data.csv files, extracts AUC per
algorithm (best version) per dataset, and computes ranks.
"""

import csv
import glob
import os

import numpy as np
import pandas as pd


def load_auc_data(results_dir="results"):
    """Load AUC values: returns DataFrame with columns [dataset, algorithm, auc]."""
    rows = []
    csv_files = sorted(glob.glob(os.path.join(results_dir, "*/peptide_precision_plot_data.csv")))

    for path in csv_files:
        dataset = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                algorithm = row.get("algorithm")
                version = row.get("version")
                auc_raw = row.get("auc")
                if not algorithm or not version or auc_raw in (None, ""):
                    print(f"Skipping malformed row in {path}: {row}")
                    continue
                try:
                    auc = float(auc_raw)
                except (TypeError, ValueError):
                    print(f"Skipping row with non-numeric auc in {path}: {row}")
                    continue
                rows.append({
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "version": version,
                    "auc": auc,
                })

    df = pd.DataFrame(rows)
    if df.empty or not {"dataset", "algorithm", "auc"}.issubset(df.columns):
        return pd.DataFrame(columns=["dataset", "algorithm", "auc"]).reset_index(drop=True)
    # Keep best version per algorithm per dataset
    df = df.loc[df.groupby(["dataset", "algorithm"])["auc"].idxmax()]
    return df[["dataset", "algorithm", "auc"]].reset_index(drop=True)


def compute_ranks(df):
    """Add a 'rank' column (1 = best) per dataset. Returns the augmented DataFrame."""
    df = df.copy()
    df["rank"] = df.groupby("dataset")["auc"].rank(ascending=False, method="min").astype(int)
    return df


def algo_order_by_median_rank(df):
    """Return list of algorithms sorted by median rank (best first)."""
    median_ranks = df.groupby("algorithm")["rank"].median().sort_values()
    return median_ranks.index.tolist()
