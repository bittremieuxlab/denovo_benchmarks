"""Category-wise algorithm performance.

Groups datasets by experimental type and shows per-category
algorithm rankings as faceted boxplots.
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def categorize_dataset(name):
    """Assign a category to a dataset based on its name."""
    if name.startswith("9_species"):
        return "9 Species"
    if name.startswith("21PTMs"):
        return "21 PTMs"
    if name.startswith("PT_"):
        return "ProteomeTools"
    if name.startswith("LFQ_"):
        return "LFQ"
    if name.startswith("CAMPI"):
        return "CAMPI"
    if "herceptin" in name:
        return "Herceptin"
    if "multiprotease_ptm" in name:
        return "Multiprotease PTM"
    if "multiprotease" in name:
        return "Multiprotease"
    if "mAb" in name:
        return "mAb"
    if "phospho" in name:
        return "Phospho"
    if "HLA" in name or "MHC" in name:
        return "Immunopeptidomics"
    if "single_cell" in name:
        return "Single cell"
    if name in ("animal_invertebrate", "animal_mammal_1", "animal_mammal_2"):
        return "Animal"
    if name in ("plant", "fungus"):
        return "Non-model organism"
    if name.startswith("multitool"):
        return "Multitool"
    if name.startswith("iPRG"):
        return "iPRG"
    return "Other"


def main():
    df = load_auc_data()
    df = compute_ranks(df)
    df["category"] = df["dataset"].apply(categorize_dataset)

    algo_order = algo_order_by_median_rank(df)

    # Count datasets per category
    cat_counts = df.groupby("category")["dataset"].nunique().sort_values(ascending=False)
    # Keep categories with at least 2 datasets
    keep_cats = cat_counts[cat_counts >= 2].index.tolist()
    df_filtered = df[df["category"].isin(keep_cats)]

    if not keep_cats:
        print("No category has >= 2 datasets; nothing to plot.")
        return

    n_cats = len(keep_cats)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for i, cat in enumerate(sorted(keep_cats)):
        ax = axes[i]
        cat_data = df_filtered[df_filtered["category"] == cat]
        n_ds = cat_data["dataset"].nunique()

        sns.boxplot(
            data=cat_data, x="algorithm", y="auc", order=algo_order,
            color="lightblue", fliersize=0, width=0.6, ax=ax,
        )
        sns.stripplot(
            data=cat_data, x="algorithm", y="auc", order=algo_order,
            color="steelblue", alpha=0.4, size=3, jitter=0.2, ax=ax,
        )
        ax.tick_params(axis="x", rotation=90)
        for label in ax.get_xticklabels():
            label.set_fontsize(7)
        ax.set_xlabel("")
        ax.set_ylabel("Peptide precision AUC" if i % n_cols == 0 else "")
        ax.set_title(f"{cat} (n={n_ds})", fontsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Algorithm performance by dataset category", fontsize=14, y=1.01)
    plt.tight_layout()

    out = "plots/category_performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
