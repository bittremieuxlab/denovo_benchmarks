"""Version improvement plot: InstaNovo v1.1.2 vs v1.2.2.

Shows paired AUC comparison across all datasets for both peptide and AA precision.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import glob

import matplotlib.pyplot as plt
import numpy as np


def load_instanovo_paired(metric_file="peptide_precision_plot_data.csv"):
    """Load paired AUC values for instanovo v1.1.2 and v1.2.2 across datasets."""
    data = {}  # dataset -> {version: auc}
    for path in sorted(glob.glob(f"results/*/{metric_file}")):
        dataset = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["algorithm"] == "instanovo" and row["version"] in ("1.1.2", "1.2.2"):
                    data.setdefault(dataset, {})[row["version"]] = float(row["auc"])

    # Keep only datasets with both versions
    paired = {ds: v for ds, v in data.items() if len(v) == 2}
    datasets = sorted(paired.keys())
    v1 = np.array([paired[ds]["1.1.2"] for ds in datasets])
    v2 = np.array([paired[ds]["1.2.2"] for ds in datasets])
    return datasets, v1, v2


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Paired dot plot (peptide AUC) ---
    datasets, v1, v2 = load_instanovo_paired("peptide_precision_plot_data.csv")
    improvement = v2 - v1

    ax = axes[0]
    order = np.argsort(improvement)
    colors = ["#d62728" if imp < 0 else "#2ca02c" for imp in improvement[order]]
    ax.barh(range(len(datasets)), improvement[order], color=colors, height=0.7, alpha=0.8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([datasets[i] for i in order], fontsize=5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("AUC improvement (v1.2.2 - v1.1.2)")
    ax.set_title("Peptide precision AUC change")
    median_imp = np.median(improvement)
    mean_imp = np.mean(improvement)
    ax.text(
        0.95, 0.05,
        f"Median: {median_imp:+.4f}\nMean: {mean_imp:+.4f}\nImproved: {(improvement > 0).sum()}/{len(improvement)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # --- Panel 2: Paired dot plot (AA AUC) ---
    datasets_aa, v1_aa, v2_aa = load_instanovo_paired("AA_precision_plot_data.csv")
    improvement_aa = v2_aa - v1_aa

    ax = axes[1]
    order_aa = np.argsort(improvement_aa)
    colors_aa = ["#d62728" if imp < 0 else "#2ca02c" for imp in improvement_aa[order_aa]]
    ax.barh(range(len(datasets_aa)), improvement_aa[order_aa], color=colors_aa, height=0.7, alpha=0.8)
    ax.set_yticks(range(len(datasets_aa)))
    ax.set_yticklabels([datasets_aa[i] for i in order_aa], fontsize=5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("AUC improvement (v1.2.2 - v1.1.2)")
    ax.set_title("AA precision AUC change")
    median_aa = np.median(improvement_aa)
    mean_aa = np.mean(improvement_aa)
    ax.text(
        0.95, 0.05,
        f"Median: {median_aa:+.4f}\nMean: {mean_aa:+.4f}\nImproved: {(improvement_aa > 0).sum()}/{len(improvement_aa)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # --- Panel 3: Scatter v1.1.2 vs v1.2.2 ---
    ax = axes[2]
    ax.scatter(v1, v2, alpha=0.5, s=20, color="steelblue", label="Peptide AUC")
    ax.scatter(v1_aa, v2_aa, alpha=0.5, s=20, color="darkorange", marker="^", label="AA AUC")
    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("InstaNovo v1.1.2 AUC")
    ax.set_ylabel("InstaNovo v1.2.2 AUC")
    ax.set_title("Version comparison scatter")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.suptitle("InstaNovo version improvement: v1.1.2 → v1.2.2", fontsize=14, y=1.02)
    plt.tight_layout()

    out = "plots/instanovo_version_improvement.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
