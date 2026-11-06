"""Dataset clustering by algorithm ranking profile.

Clusters datasets based on how algorithms rank on them, revealing
groups of datasets that produce similar performance patterns.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from load_ranking_data import load_auc_data, compute_ranks


def main():
    df = load_auc_data()
    df = compute_ranks(df)

    # Pivot: datasets x algorithms, values = rank
    rank_matrix = df.pivot(index="dataset", columns="algorithm", values="rank")
    # Fill missing with worst rank
    rank_matrix = rank_matrix.fillna(rank_matrix.max().max() + 1)

    g = sns.clustermap(
        rank_matrix,
        cmap="YlGnBu_r",
        figsize=(14, 20),
        row_cluster=True,
        col_cluster=True,
        method="ward",
        metric="euclidean",
        linewidths=0.2,
        cbar_kws={"label": "Rank (1 = best)"},
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=45, ha="right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=6)
    g.fig.suptitle("Dataset clustering by algorithm ranking profile", y=1.01, fontsize=14)

    out = "plots/dataset_clustering.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    g.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
