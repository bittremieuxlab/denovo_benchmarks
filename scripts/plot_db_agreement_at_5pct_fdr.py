"""DB search agreement rate at 5% FDR.

For each algorithm, computes the fraction of DB-identified peptides
recovered at >= 95% precision (i.e., at most 5% false discovery rate).

Note on terminology: since the ground truth labels come from database
search (not the true set of all peptides in the sample), this metric
measures agreement with DB search, not true recall. De novo algorithms
may find real peptides that DB search missed, but those are not counted
as correct here.

The metric is computed as:
    agreement_rate = precision_at_threshold * coverage_at_threshold
where the threshold is the point on the precision-coverage curve where
precision first drops below 95%, and coverage is relative to the total
number of DB-identified spectra (the labeled set).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import ast
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def compute_agreement_at_fdr(fdr=0.05):
    """Compute fraction of DB-identified peptides recovered at a given FDR threshold."""
    precision_threshold = 1.0 - fdr
    results = defaultdict(list)

    for path in sorted(glob.glob("results/*/peptide_precision_plot_data.csv")):
        ds = os.path.basename(os.path.dirname(path))
        best = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                cov = ast.literal_eval(row["coverage"])
                prec = ast.literal_eval(row["metric"])
                auc = float(row["auc"])
                algo = row["algorithm"]

                # Find the last coverage point where precision >= threshold
                cov_at_threshold = 0.0
                for c, p in zip(cov, prec):
                    if p >= precision_threshold:
                        cov_at_threshold = c
                    else:
                        break

                agreement = precision_threshold * cov_at_threshold

                if algo not in best or auc > best[algo][0]:
                    best[algo] = (auc, agreement)

        for algo, (_, agreement) in best.items():
            results[algo].append(agreement)

    return results


def main():
    df = compute_ranks(load_auc_data())
    algo_order = algo_order_by_median_rank(df)

    agreement = compute_agreement_at_fdr(fdr=0.05)

    # Build DataFrame for plotting
    records = [
        {"algorithm": algo, "agreement": val}
        for algo in algo_order
        for val in agreement.get(algo, [])
    ]
    plot_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=plot_df, x="algorithm", y="agreement", order=algo_order,
        color="lightblue", fliersize=0, width=0.5, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x="algorithm", y="agreement", order=algo_order,
        color="steelblue", alpha=0.4, size=4, jitter=0.2, ax=ax,
    )

    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of DB-identified peptides recovered")
    ax.set_title(
        "DB search agreement rate at 5% FDR\n"
        "(fraction of DB-identified peptides recovered at \u226595% precision)"
    )
    ax.set_ylim(-0.02, 1.02)

    # Add median annotations
    for i, algo in enumerate(algo_order):
        vals = agreement.get(algo, [])
        if vals:
            med = np.median(vals)
            ax.text(i, med + 0.03, f"{med:.2f}", ha="center", va="bottom",
                    fontsize=7, color="darkred")

    plt.tight_layout()

    out = "plots/db_agreement_at_5pct_fdr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Also print table
    print(f"\n{'Algorithm':20s} {'Median':>10s} {'Mean':>10s}")
    print("-" * 42)
    for algo in algo_order:
        vals = agreement.get(algo, [])
        if vals:
            print(f"{algo:20s} {np.median(vals):10.4f} {np.mean(vals):10.4f}")


if __name__ == "__main__":
    main()
