"""Option 4: Critical Difference diagram (Demsar 2006).

Shows average rank per algorithm with Nemenyi post-hoc test significance
groups connected by horizontal bars.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from load_ranking_data import load_auc_data, compute_ranks


def _nemenyi_cd(k, n, alpha=0.05):
    """Critical difference for Nemenyi test.

    k = number of algorithms, n = number of datasets.
    Uses q_alpha values from Demsar (2006) Table 5 / Studentized range.
    """
    # q_alpha for alpha=0.05 from the Studentized range distribution
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n))


def main():
    df = load_auc_data()
    df = compute_ranks(df)

    # Only include algorithms present in all (or most) datasets
    n_datasets = df["dataset"].nunique()
    algo_counts = df.groupby("algorithm")["dataset"].nunique()
    # Keep algorithms in at least 50% of datasets
    keep = algo_counts[algo_counts >= n_datasets * 0.5].index
    df = df[df["algorithm"].isin(keep)]

    # Recompute ranks after filtering (within each dataset)
    df["rank"] = df.groupby("dataset")["auc"].rank(ascending=False, method="average")

    avg_ranks = df.groupby("algorithm")["rank"].mean().sort_values()
    algorithms = avg_ranks.index.tolist()
    ranks = avg_ranks.values
    k = len(algorithms)
    n = df["dataset"].nunique()

    cd = _nemenyi_cd(k, n)

    # --- Draw the CD diagram ---
    fig, ax = plt.subplots(figsize=(12, max(4, k * 0.35)))

    # Axis: average rank
    low, high = 1, max(ranks) + 0.5
    ax.set_xlim(low - 0.3, high + 0.3)
    ax.set_ylim(-0.5, k + 1)
    ax.invert_xaxis()  # rank 1 on the right

    # Top ruler
    ax.hlines(0, low, high, color="black", linewidth=0.8)
    for tick in range(1, int(high) + 1):
        ax.vlines(tick, -0.15, 0.15, color="black", linewidth=0.8)
        ax.text(tick, -0.35, str(tick), ha="center", va="top", fontsize=8)

    # CD bar
    ax.hlines(-0.7, low, low + cd, color="red", linewidth=2)
    ax.text(low + cd / 2, -1.0, f"CD = {cd:.2f}", ha="center", va="top",
            fontsize=8, color="red")

    # Algorithm positions
    y_positions = np.linspace(1, k, k)
    for i, (algo, r) in enumerate(zip(algorithms, ranks)):
        y = y_positions[i]
        ax.plot(r, y, "ko", markersize=5)
        side = "left" if r > np.median(ranks) else "right"
        offset = 0.15 if side == "right" else -0.15
        ax.text(r + offset, y, f" {algo} ({r:.1f})", va="center",
                ha=side, fontsize=8)

    # Draw cliques (groups not significantly different)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(ranks[i] - ranks[j]) < cd:
                # Check if this pair extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique or j in clique:
                        clique.update({i, j})
                        merged = True
                        break
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    merged = True
    while merged:
        merged = False
        new_cliques = []
        used = set()
        for i, c1 in enumerate(cliques):
            if i in used:
                continue
            for j, c2 in enumerate(cliques):
                if j <= i or j in used:
                    continue
                if c1 & c2:
                    # Only merge if all pairwise differences < cd
                    combined = c1 | c2
                    all_close = all(
                        abs(ranks[a] - ranks[b]) < cd
                        for a in combined for b in combined
                    )
                    if all_close:
                        c1 = c1 | c2
                        used.add(j)
                        merged = True
            new_cliques.append(c1)
            used.add(i)
        cliques = new_cliques

    # Draw clique bars
    bar_y = -1.5
    for clique in cliques:
        members = sorted(clique)
        r_min = ranks[members[0]]
        r_max = ranks[members[-1]]
        ax.hlines(bar_y, r_min - 0.05, r_max + 0.05, color="gray",
                  linewidth=3, alpha=0.6)
        bar_y -= 0.4

    ax.set_axis_off()
    ax.set_title("Critical Difference Diagram — Algorithm Average Ranks", pad=20)
    plt.tight_layout()

    out = "plots/algorithm_ranking_cd.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
