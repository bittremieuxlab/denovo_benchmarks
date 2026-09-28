"""Rank stability / consistency analysis.

Shows which algorithms have stable rankings across datasets vs. those whose
performance is dataset-dependent. Combines:
- Left: IQR (interquartile range) of ranks as a consistency measure
- Right: Median rank vs IQR scatter (best = bottom-left: low median rank, low IQR)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from adjustText import adjust_text
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def main():
    df = load_auc_data()
    df = compute_ranks(df)

    stats = df.groupby("algorithm")["rank"].agg(
        median="median", mean="mean",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        std="std",
        min="min", max="max",
        count="count",
    ).reset_index()
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats = stats.sort_values("median")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.2, 1]})

    # --- Panel 1: Rank range plot (median + IQR + full range) ---
    ax = axes[0]
    y = np.arange(len(stats))
    algos = stats["algorithm"].values

    # Full range (min to max)
    ax.hlines(y, stats["min"].values, stats["max"].values,
              color="lightgray", linewidth=1.5, zorder=1)
    # IQR
    ax.hlines(y, stats["q25"].values, stats["q75"].values,
              color="steelblue", linewidth=4, alpha=0.7, zorder=2)
    # Median
    ax.scatter(stats["median"].values, y, color="darkred", s=50, zorder=3, label="Median")

    ax.set_yticks(y)
    ax.set_yticklabels(algos, fontsize=9)
    ax.set_xlabel("Rank (1 = best)")
    ax.set_title("Rank distribution per algorithm\n(bar = IQR, line = full range, dot = median)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)

    # Add IQR and count annotations
    for i, row in stats.iterrows():
        idx = np.where(algos == row["algorithm"])[0][0]
        ax.text(
            row["max"] + 0.3, idx,
            f"IQR={row['iqr']:.1f}  n={int(row['count'])}",
            va="center", fontsize=7, color="gray",
        )

    # --- Panel 2: Median rank vs IQR scatter ---
    ax = axes[1]
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(stats))

    texts = []
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.scatter(
            row["median"], row["iqr"],
            s=80, color=cmap(i), zorder=3, edgecolors="black", linewidth=0.5,
        )
        texts.append(ax.text(row["median"], row["iqr"], row["algorithm"], fontsize=7))

    ax.set_xlabel("Median rank (lower = better)")
    ax.set_ylabel("IQR of rank (lower = more consistent)")
    ax.set_title("Performance vs Consistency")

    # Quadrant shading
    med_x = stats["median"].median()
    med_y = stats["iqr"].median()
    ax.axvline(med_x, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(med_y, color="gray", linestyle=":", alpha=0.4)
    ax.text(
        0.02, 0.02, "Best:\nGood & consistent",
        transform=ax.transAxes, fontsize=7, color="green", alpha=0.6,
    )
    ax.text(
        0.98, 0.98, "Worst:\nPoor & inconsistent",
        transform=ax.transAxes, fontsize=7, color="red", alpha=0.6,
        ha="right", va="top",
    )
    ax.text(
        0.02, 0.98, "Good but\nunpredictable",
        transform=ax.transAxes, fontsize=7, color="orange", alpha=0.6,
        va="top",
    )
    ax.text(
        0.72, 0.02, "Poor but\nconsistent",
        transform=ax.transAxes, fontsize=7, color="slategray", alpha=0.6,
    )

    adjust_text(
        texts, ax=ax,
        force_points=(2, 2),
        force_text=(1.5, 1.5),
        expand=(2, 2),
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
    )

    plt.tight_layout()

    out = "plots/rank_stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
