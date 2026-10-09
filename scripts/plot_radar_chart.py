"""Multi-metric radar chart per algorithm.

Shows peptide AUC, AA AUC, RT error (inverted), and SA in one
spider plot per algorithm for an at-a-glance profile.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import csv
import glob
import json
import re
import math

import matplotlib.pyplot as plt
import numpy as np
from load_ranking_data import load_auc_data, algo_order_by_median_rank, compute_ranks


def load_curve_auc(metric_file):
    """Load coverage-vs-metric curves and compute AUC."""
    data = {}
    for path in sorted(glob.glob(f"results/*/{metric_file}")):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                cov_str = re.sub(r'\bnan\b', 'NaN', row["coverage"])
                met_str = re.sub(r'\bnan\b', 'NaN', row["metric"])
                coverage = np.array(json.loads(cov_str), dtype=float)
                metric = np.array(json.loads(met_str), dtype=float)
                valid = ~(np.isnan(coverage) | np.isnan(metric))
                if valid.sum() < 2:
                    continue
                auc = float(np.trapezoid(metric[valid], coverage[valid]))
                key = (ds, row["algorithm"])
                if key not in data or auc > data[key]:
                    data[key] = auc
    return data


def main():
    # Load all metrics
    pep_df = load_auc_data()
    pep_best = pep_df.loc[pep_df.groupby(["dataset", "algorithm"])["auc"].idxmax()]
    pep = {a: g["auc"].median() for a, g in pep_best.groupby("algorithm")}

    aa_dict = {}
    for path in sorted(glob.glob("results/*/AA_precision_plot_data.csv")):
        ds = os.path.basename(os.path.dirname(path))
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (ds, row["algorithm"])
                auc = float(row["auc"])
                if key not in aa_dict or auc > aa_dict[key]:
                    aa_dict[key] = auc
    aa = {}
    from collections import defaultdict
    aa_by = defaultdict(list)
    for (ds, algo), v in aa_dict.items():
        aa_by[algo].append(v)
    aa = {a: np.median(v) for a, v in aa_by.items()}

    rt_raw = load_curve_auc("RT_difference_plot_data.csv")
    rt_by = defaultdict(list)
    for (ds, algo), v in rt_raw.items():
        rt_by[algo].append(v)
    rt = {a: np.median(v) for a, v in rt_by.items()}

    sa_raw = load_curve_auc("SA_plot_data.csv")
    sa_by = defaultdict(list)
    for (ds, algo), v in sa_raw.items():
        sa_by[algo].append(v)
    sa = {a: np.median(v) for a, v in sa_by.items()}

    # Get algorithm order
    df = compute_ranks(load_auc_data())
    algo_order = algo_order_by_median_rank(df)

    # Metrics: peptide AUC, AA AUC, 1-RT (inverted so higher=better), SA
    metrics = ["Peptide\nAUC", "AA\nAUC", "RT quality\n(1-error)", "Spectral\nAngle"]
    n_metrics = len(metrics)

    # Normalize all to 0-1 range
    all_algos = [a for a in algo_order if a in pep and a in aa and a in rt and a in sa]

    n_algos = len(all_algos)
    n_cols = 4
    n_rows = math.ceil(n_algos / n_cols)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows),
                              subplot_kw=dict(polar=True))
    axes = axes.flatten()

    # Compute global min/max for normalization
    all_pep = [pep[a] for a in all_algos]
    all_aa = [aa[a] for a in all_algos]
    all_rt_inv = [1 - rt[a] for a in all_algos]
    all_sa = [sa[a] for a in all_algos]

    def _norm(x, xs):
        lo, hi = min(xs), max(xs)
        if hi == lo:
            return 0.5
        return (x - lo) / (hi - lo)

    for i, algo in enumerate(all_algos):
        ax = axes[i]
        values = [
            _norm(pep[algo], all_pep),
            _norm(aa[algo], all_aa),
            _norm(1 - rt[algo], all_rt_inv),
            _norm(sa[algo], all_sa),
        ]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=1.5, markersize=4)
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(algo, fontsize=10, fontweight="bold", pad=15)
        ax.tick_params(axis="y", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Multi-metric algorithm profiles\n(all axes: higher = better)", fontsize=14, y=1.02)
    plt.tight_layout()

    out = "plots/radar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
