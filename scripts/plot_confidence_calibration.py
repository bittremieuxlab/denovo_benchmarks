"""Confidence calibration analysis.

Measures how well each algorithm's confidence ordering predicts
actual correctness by computing the precision drop-off rate
(slope of precision as coverage increases).

Also shows the "calibration gap": difference between precision
at low coverage (top predictions) vs high coverage (all predictions).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import ast
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_ranking_data import algo_order_by_median_rank, load_auc_data, compute_ranks


def extract_calibration_data():
    """Extract precision at low and high coverage for each algorithm/dataset."""
    rows = []
    for path in sorted(glob.glob("results/*/peptide_precision_plot_data.csv")):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                cov = ast.literal_eval(row["coverage"])
                prec = ast.literal_eval(row["metric"])
                if len(cov) < 10:
                    continue

                # Precision at ~10% coverage (top predictions)
                prec_low = None
                for c, p in zip(cov, prec):
                    if c >= 0.10:
                        prec_low = p
                        break

                # Precision at ~90% coverage (nearly all predictions)
                prec_high = None
                for c, p in zip(cov, prec):
                    if c >= 0.90:
                        prec_high = p
                        break

                # Precision at 100% coverage
                prec_full = prec[-1]

                if prec_low is not None and prec_high is not None:
                    rows.append({
                        "dataset": ds,
                        "algorithm": row["algorithm"],
                        "version": row["version"],
                        "prec_at_10": prec_low,
                        "prec_at_90": prec_high,
                        "prec_at_100": prec_full,
                        "calibration_gap": prec_low - prec_full,
                        "auc": float(row["auc"]),
                    })

    df = pd.DataFrame(rows)
    # Keep best version
    df = df.loc[df.groupby(["dataset", "algorithm"])["auc"].idxmax()]
    return df.reset_index(drop=True)


def main():
    calib = extract_calibration_data()

    full_df = compute_ranks(load_auc_data())
    algo_order = algo_order_by_median_rank(full_df)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Panel 1: Calibration gap boxplot ---
    ax = axes[0]
    sns.boxplot(
        data=calib, x="algorithm", y="calibration_gap", order=algo_order,
        color="lightyellow", fliersize=0, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=calib, x="algorithm", y="calibration_gap", order=algo_order,
        color="darkorange", alpha=0.35, size=3, jitter=0.2, ax=ax,
    )
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(8)
    ax.set_xlabel("")
    ax.set_ylabel("Precision drop (top 10% − full)")
    ax.set_title("Confidence calibration gap\n(higher = better calibrated confidence scores)")

    # --- Panel 2: Precision at 10% vs 100% coverage scatter ---
    ax = axes[1]
    medians = calib.groupby("algorithm").agg(
        p10=("prec_at_10", "median"),
        p100=("prec_at_100", "median"),
    ).reindex(algo_order)

    cmap = plt.colormaps.get_cmap("tab20").resampled(len(algo_order))
    from adjustText import adjust_text
    texts = []
    for i, algo in enumerate(algo_order):
        if algo in medians.index:
            ax.scatter(medians.loc[algo, "p100"], medians.loc[algo, "p10"],
                       s=80, color=cmap(i), edgecolors="black", linewidth=0.5, zorder=3)
            texts.append(ax.text(medians.loc[algo, "p100"], medians.loc[algo, "p10"],
                                 algo, fontsize=7))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Precision at 100% coverage (overall)")
    ax.set_ylabel("Precision at 10% coverage (top predictions)")
    ax.set_title("Top-prediction quality vs overall precision\n(above diagonal = well-calibrated confidence)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_aspect("equal")
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5))

    # --- Panel 3: Precision at 10%, 50%, 90%, 100% as grouped bars ---
    ax = axes[2]
    thresholds = {
        "10%": "prec_at_10",
        "90%": "prec_at_90",
        "100%": "prec_at_100",
    }
    width = 0.25
    x = np.arange(len(algo_order))

    for j, (label, col) in enumerate(thresholds.items()):
        vals = [calib[calib["algorithm"] == a][col].median() if a in calib["algorithm"].values else 0
                for a in algo_order]
        ax.bar(x + j * width, vals, width, label=f"Coverage {label}", alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(algo_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Median peptide precision")
    ax.set_title("Precision degradation across coverage levels")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Confidence calibration analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    out = "plots/confidence_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
