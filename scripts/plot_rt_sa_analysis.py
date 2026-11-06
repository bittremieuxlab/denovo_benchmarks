"""RT difference and Spectral Angle analysis across algorithms and datasets.

Since RT and SA CSVs don't have a pre-computed AUC, we compute the area under
each curve (coverage vs metric) as a summary statistic, then produce:
- Panel 1: RT median metric boxplot per algorithm
- Panel 2: SA median metric boxplot per algorithm
- Panel 3: Heatmap of RT AUC across algorithms x datasets
- Panel 4: Heatmap of SA AUC across algorithms x datasets
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import glob
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def load_curve_auc(metric_file):
    """Load coverage-vs-metric curves and compute AUC for each algorithm/dataset."""
    rows = []
    for path in sorted(glob.glob(f"results/*/{metric_file}")):
        dataset = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                # Handle nan/inf in data by replacing with JSON-compatible values
                cov_str = re.sub(r'\bnan\b', 'NaN', row["coverage"])
                met_str = re.sub(r'\bnan\b', 'NaN', row["metric"])
                coverage = json.loads(cov_str)
                metric = json.loads(met_str)
                # Compute AUC via trapezoidal rule
                coverage = np.array(coverage, dtype=float)
                metric = np.array(metric, dtype=float)
                # Skip if too many NaNs
                valid = ~(np.isnan(coverage) | np.isnan(metric))
                if valid.sum() < 2:
                    continue
                auc = float(np.trapezoid(metric[valid], coverage[valid]))
                rows.append({
                    "dataset": dataset,
                    "algorithm": row["algorithm"],
                    "version": row["version"],
                    "auc": auc,
                })

    df = pd.DataFrame(rows)
    # Keep best version per algorithm per dataset
    # For RT: lower is better (difference), so pick min. For SA: higher is better, pick max.
    return df


def best_version(df, higher_is_better=True):
    """Keep best version per algorithm per dataset."""
    if higher_is_better:
        idx = df.groupby(["dataset", "algorithm"])["auc"].idxmax()
    else:
        idx = df.groupby(["dataset", "algorithm"])["auc"].idxmin()
    return df.loc[idx][["dataset", "algorithm", "auc"]].reset_index(drop=True)


def main():
    print("Loading RT data...")
    rt_raw = load_curve_auc("RT_difference_plot_data.csv")
    # RT difference: lower AUC = better (less deviation). Pick min version.
    rt = best_version(rt_raw, higher_is_better=False)

    print("Loading SA data...")
    sa_raw = load_curve_auc("SA_plot_data.csv")
    # Spectral angle: higher AUC = better. Pick max version.
    sa = best_version(sa_raw, higher_is_better=True)

    # Order algorithms by median RT AUC (ascending = best first)
    rt_order = rt.groupby("algorithm")["auc"].median().sort_values().index.tolist()
    # Order algorithms by median SA AUC (descending = best first)
    sa_order = sa.groupby("algorithm")["auc"].median().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(22, 14))

    # --- Panel 1: RT boxplot ---
    ax = axes[0, 0]
    sns.boxplot(data=rt, x="algorithm", y="auc", order=rt_order,
                color="lightsalmon", fliersize=0, width=0.5, ax=ax)
    sns.stripplot(data=rt, x="algorithm", y="auc", order=rt_order,
                  color="firebrick", alpha=0.35, size=3, jitter=0.2, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("RT difference AUC (lower = better)")
    ax.set_title("Retention Time prediction error across datasets")

    # --- Panel 2: SA boxplot ---
    ax = axes[0, 1]
    sns.boxplot(data=sa, x="algorithm", y="auc", order=sa_order,
                color="lightgreen", fliersize=0, width=0.5, ax=ax)
    sns.stripplot(data=sa, x="algorithm", y="auc", order=sa_order,
                  color="darkgreen", alpha=0.35, size=3, jitter=0.2, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_xlabel("")
    ax.set_ylabel("Spectral angle AUC (higher = better)")
    ax.set_title("Spectral angle (predicted vs observed) across datasets")

    # --- Panel 3: RT heatmap ---
    ax = axes[1, 0]
    rt_pivot = rt.pivot(index="algorithm", columns="dataset", values="auc")
    rt_pivot = rt_pivot.loc[[a for a in rt_order if a in rt_pivot.index]]
    col_order = rt_pivot.mean(axis=0).sort_values().index
    rt_pivot = rt_pivot[col_order]
    sns.heatmap(rt_pivot, cmap="YlOrRd", ax=ax, linewidths=0.2,
                cbar_kws={"label": "RT diff AUC (lower=better)"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("RT difference AUC heatmap")

    # --- Panel 4: SA heatmap ---
    ax = axes[1, 1]
    sa_pivot = sa.pivot(index="algorithm", columns="dataset", values="auc")
    sa_pivot = sa_pivot.loc[[a for a in sa_order if a in sa_pivot.index]]
    col_order_sa = sa_pivot.mean(axis=0).sort_values(ascending=False).index
    sa_pivot = sa_pivot[col_order_sa]
    sns.heatmap(sa_pivot, cmap="YlGn", ax=ax, linewidths=0.2,
                cbar_kws={"label": "SA AUC (higher=better)"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=4)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Spectral angle AUC heatmap")

    fig.suptitle("Retention Time & Spectral Angle Analysis", fontsize=14, y=1.01)
    plt.tight_layout()

    out = "plots/rt_sa_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
