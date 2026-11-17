"""Algorithm agreement matrix.

Computes correlation between all algorithm pairs across datasets.
Two panels: Spearman rank correlation (on ranks) and Pearson correlation (on AUC).
High correlation = similar behavior across datasets.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def compute_pairwise_corr(matrix, method="spearman"):
    """Compute pairwise correlation between all columns."""
    algos = matrix.columns.tolist()
    n = len(algos)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mask = matrix[[algos[i], algos[j]]].notnull().all(axis=1)
            if mask.sum() >= 3:
                if method == "spearman":
                    corr, _ = spearmanr(matrix.loc[mask, algos[i]], matrix.loc[mask, algos[j]])
                else:
                    corr, _ = pearsonr(matrix.loc[mask, algos[i]], matrix.loc[mask, algos[j]])
                corr_matrix[i, j] = corr
            else:
                corr_matrix[i, j] = np.nan

    # Symmetrize, replace non-finite with neutral 0, and force diagonal to 1
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    corr_matrix = np.where(np.isfinite(corr_matrix), corr_matrix, 0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    return pd.DataFrame(corr_matrix, index=algos, columns=algos)


def main():
    df = load_auc_data()
    df = compute_ranks(df)

    rank_matrix = df.pivot(index="dataset", columns="algorithm", values="rank")
    auc_matrix = df.pivot(index="dataset", columns="algorithm", values="auc")

    # Rank-based (Spearman)
    rank_corr = compute_pairwise_corr(rank_matrix, method="spearman")

    g1 = sns.clustermap(
        rank_corr,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        figsize=(10, 10),
        method="ward",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Spearman rank correlation"},
    )
    g1.fig.suptitle(
        "Algorithm agreement: rank correlation across datasets\n"
        "(high = similar ranking behavior)",
        y=1.02, fontsize=13,
    )
    out1 = "plots/algorithm_agreement.png"
    g1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")

    # AUC-based (Pearson)
    auc_corr = compute_pairwise_corr(auc_matrix, method="pearson")

    g2 = sns.clustermap(
        auc_corr,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        figsize=(10, 10),
        method="ward",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        cbar_kws={"label": "Pearson correlation (AUC)"},
    )
    g2.fig.suptitle(
        "Algorithm agreement: AUC correlation across datasets\n"
        "(high = similar AUC patterns across datasets)",
        y=1.02, fontsize=13,
    )
    out2 = "plots/algorithm_agreement_auc.png"
    g2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
