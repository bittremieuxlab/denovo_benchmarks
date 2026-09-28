"""Option 3: Box + strip plot of algorithm ranks across datasets.

Shows the distribution of ranks for each algorithm, ordered by median rank.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def main():
    df = load_auc_data()
    df = compute_ranks(df)
    order = algo_order_by_median_rank(df)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df, x="algorithm", y="rank", order=order,
        color="lightblue", fliersize=0, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=df, x="algorithm", y="rank", order=order,
        color="steelblue", alpha=0.4, size=4, jitter=0.2, ax=ax,
    )

    ax.invert_yaxis()
    ax.set_xlabel("")
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Algorithm ranking across all datasets (peptide precision AUC)")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    plt.tight_layout()

    out = "plots/algorithm_ranking_boxplot.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
