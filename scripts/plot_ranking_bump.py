"""Option 1: Bump chart showing algorithm rank across datasets.

Each algorithm is a colored line; x-axis = dataset, y-axis = rank.
Datasets are ordered to minimize line crossings (by first principal component
of the rank matrix).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def main():
    df = load_auc_data()
    df = compute_ranks(df)

    # Pivot to rank matrix: datasets x algorithms
    rank_matrix = df.pivot(index="dataset", columns="algorithm", values="rank")

    # Order datasets by mean rank profile (PCA-like: sort by first SVD component)
    filled = rank_matrix.fillna(rank_matrix.max().max() + 1)
    u, _, _ = np.linalg.svd(filled.values - filled.values.mean(axis=0), full_matrices=False)
    dataset_order = rank_matrix.index[np.argsort(u[:, 0])].tolist()
    rank_matrix = rank_matrix.loc[dataset_order]

    algo_order = algo_order_by_median_rank(df)

    # Use a qualitative colormap
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(algo_order))
    colors = {algo: cmap(i) for i, algo in enumerate(algo_order)}

    fig, ax = plt.subplots(figsize=(20, 8))

    x = np.arange(len(dataset_order))
    for algo in algo_order:
        y = rank_matrix[algo].values
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], marker=".", markersize=4, linewidth=1.2,
                label=algo, color=colors[algo], alpha=0.8)

    ax.invert_yaxis()
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_order, rotation=90, fontsize=6)
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Algorithm ranking bump chart across datasets")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out = "plots/algorithm_ranking_bump.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
