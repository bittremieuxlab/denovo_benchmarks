"""Dataset difficulty ranking.

Ranks datasets by mean peptide precision AUC across all algorithms.
Reveals which datasets are universally hard vs easy.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
from load_ranking_data import load_auc_data


def main():
    df = load_auc_data()

    # Mean AUC across algorithms per dataset
    ds_stats = df.groupby("dataset")["auc"].agg(["mean", "std", "count"]).reset_index()
    ds_stats = ds_stats.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 18))
    y = np.arange(len(ds_stats))
    colors = plt.colormaps.get_cmap("RdYlGn")(ds_stats["mean"].values)

    ax.barh(y, ds_stats["mean"].values, xerr=ds_stats["std"].values,
            color=colors, height=0.7, capsize=2, error_kw={"linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(ds_stats["dataset"].values, fontsize=7)
    ax.set_xlabel("Mean peptide precision AUC across algorithms")
    ax.set_title("Dataset difficulty ranking\n(left = harder, right = easier)")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    out = "plots/dataset_difficulty.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
