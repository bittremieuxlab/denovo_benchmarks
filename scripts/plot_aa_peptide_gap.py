"""AA precision vs peptide precision gap analysis.

Shows how much each algorithm gets individual amino acids right vs full peptide
sequences. A large gap indicates frequent small errors (residue swaps, I/L
confusion, near-isobaric substitutions) that break exact sequence matching.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from load_ranking_data import load_auc_data


def load_aa_auc():
    """Load AA precision AUC (best version per algo per dataset)."""
    data = {}
    for path in sorted(glob.glob("results/*/AA_precision_plot_data.csv")):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (ds, row["algorithm"])
                auc = float(row["auc"])
                if key not in data or auc > data[key]:
                    data[key] = auc
    return data


def main():
    pep_df = load_auc_data()
    # Best version per algo per dataset
    pep_best = pep_df.loc[pep_df.groupby(["dataset", "algorithm"])["auc"].idxmax()]
    pep_dict = {(r["dataset"], r["algorithm"]): r["auc"] for _, r in pep_best.iterrows()}
    aa_dict = load_aa_auc()

    # Build paired data
    records = []
    for (ds, algo), pep_auc in pep_dict.items():
        if (ds, algo) in aa_dict:
            aa_auc = aa_dict[(ds, algo)]
            records.append({
                "dataset": ds, "algorithm": algo,
                "pep_auc": pep_auc, "aa_auc": aa_auc,
                "gap": aa_auc - pep_auc,
            })

    # Aggregate per algorithm
    gap_by = defaultdict(list)
    pep_by = defaultdict(list)
    aa_by = defaultdict(list)
    for r in records:
        gap_by[r["algorithm"]].append(r["gap"])
        pep_by[r["algorithm"]].append(r["pep_auc"])
        aa_by[r["algorithm"]].append(r["aa_auc"])

    algos_sorted = sorted(gap_by.keys(), key=lambda a: np.median(gap_by[a]))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Panel 1: Gap boxplot ---
    ax = axes[0]
    import pandas as pd
    gap_records = [{"algorithm": algo, "gap": g} for algo in algos_sorted for g in gap_by[algo]]
    gap_df = pd.DataFrame(gap_records)
    sns.boxplot(data=gap_df, x="algorithm", y="gap", order=algos_sorted,
                color="lightyellow", fliersize=0, width=0.5, ax=ax)
    sns.stripplot(data=gap_df, x="algorithm", y="gap", order=algos_sorted,
                  color="darkorange", alpha=0.35, size=3, jitter=0.2, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("AA AUC − Peptide AUC")
    ax.set_title("Gap between AA and peptide precision\n(higher = more small errors per peptide)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")

    # --- Panel 2: Paired bar chart (median AA vs peptide AUC) ---
    ax = axes[1]
    # Sort by peptide AUC descending
    algos_by_pep = sorted(pep_by.keys(), key=lambda a: -np.median(pep_by[a]))
    x = np.arange(len(algos_by_pep))
    width = 0.35
    pep_medians = [np.median(pep_by[a]) for a in algos_by_pep]
    aa_medians = [np.median(aa_by[a]) for a in algos_by_pep]

    bars1 = ax.bar(x - width / 2, pep_medians, width, label="Peptide precision AUC",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, aa_medians, width, label="AA precision AUC",
                   color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos_by_pep, rotation=45, ha="right")
    ax.set_ylabel("Median AUC")
    ax.set_title("Peptide vs AA precision (median across datasets)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # --- Panel 3: Scatter AA AUC vs Peptide AUC (per algorithm median) ---
    ax = axes[2]
    texts = []
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(algos_sorted))
    for i, algo in enumerate(algos_by_pep):
        px = np.median(pep_by[algo])
        ay = np.median(aa_by[algo])
        ax.scatter(px, ay, s=80, color=cmap(i), edgecolors="black", linewidth=0.5, zorder=3)
        texts.append(ax.text(px, ay, algo, fontsize=7))

    # Diagonal and gap reference lines
    lims = [0.1, 1.0]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.3, label="AA = Peptide (no gap)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Median peptide precision AUC")
    ax.set_ylabel("Median AA precision AUC")
    ax.set_title("AA vs peptide precision\n(distance from diagonal = gap)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")

    # Annotate smsnet specifically
    adjust_text(
        texts, ax=ax,
        force_points=(2, 2), force_text=(1.5, 1.5), expand=(2, 2),
        arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
    )

    fig.suptitle(
        "AA vs Peptide Precision Gap — detecting algorithms with frequent small sequence errors",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = "plots/aa_peptide_gap.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
