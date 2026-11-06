"""Precision at fixed coverage thresholds.

Compares algorithms at 25%, 50%, and 75% coverage to see which
excel at high-confidence predictions vs comprehensive coverage.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import ast
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_ranking_data import algo_order_by_median_rank, load_auc_data, compute_ranks


def extract_precision_at_coverage(target_cov):
    """Extract precision at a specific coverage threshold across all datasets."""
    rows = []
    for path in sorted(glob.glob("results/*/peptide_precision_plot_data.csv")):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                cov = ast.literal_eval(row["coverage"])
                prec = ast.literal_eval(row["metric"])
                # Find precision at target coverage
                best_prec = None
                for c, p in zip(cov, prec):
                    if c >= target_cov:
                        best_prec = p
                        break
                if best_prec is not None:
                    rows.append({
                        "dataset": ds,
                        "algorithm": row["algorithm"],
                        "version": row["version"],
                        "precision": best_prec,
                    })
    df = pd.DataFrame(rows)
    # Keep best version
    df = df.loc[df.groupby(["dataset", "algorithm"])["precision"].idxmax()]
    return df[["dataset", "algorithm", "precision"]].reset_index(drop=True)


def main():
    # Get algorithm order from overall ranking
    full_df = compute_ranks(load_auc_data())
    algo_order = algo_order_by_median_rank(full_df)

    thresholds = [0.25, 0.50, 0.75]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, thr in enumerate(thresholds):
        ax = axes[idx]
        df = extract_precision_at_coverage(thr)

        sns.boxplot(
            data=df, x="algorithm", y="precision", order=algo_order,
            color="lightblue", fliersize=0, width=0.5, ax=ax,
        )
        sns.stripplot(
            data=df, x="algorithm", y="precision", order=algo_order,
            color="steelblue", alpha=0.35, size=3, jitter=0.2, ax=ax,
        )
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_fontsize(8)
        ax.set_xlabel("")
        ax.set_ylabel("Peptide precision" if idx == 0 else "")
        ax.set_title(f"Precision at {int(thr * 100)}% coverage")
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Peptide precision at fixed coverage thresholds\n"
        "(how fast does precision degrade as more predictions are included?)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = "plots/precision_at_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
