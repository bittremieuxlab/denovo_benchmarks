"""Option 2: Heatmap of algorithm ranks across datasets.

Rows = algorithms (sorted by median rank), columns = datasets.
Color encodes rank (dark = best).
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
    algo_order = algo_order_by_median_rank(df)

    rank_matrix = df.pivot(index="algorithm", columns="dataset", values="rank")
    rank_matrix = rank_matrix.loc[algo_order]

    # Sort datasets by average rank of top algorithm for visual coherence
    col_order = rank_matrix.mean(axis=0).sort_values().index
    rank_matrix = rank_matrix[col_order]

    fig, ax = plt.subplots(figsize=(22, 8))
    sns.heatmap(
        rank_matrix, cmap="YlGnBu_r", annot=False, linewidths=0.3,
        cbar_kws={"label": "Rank (1 = best)"}, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Algorithm ranking heatmap across datasets (peptide precision AUC)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()

    out = "plots/algorithm_ranking_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
