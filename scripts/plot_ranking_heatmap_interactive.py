"""Interactive heatmap of algorithm ranks across datasets using Plotly.

Rows = algorithms (sorted by median rank), columns = datasets.
Color encodes rank (dark = best). Hover shows algorithm, dataset, rank, and AUC.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import plotly.graph_objects as go
from load_ranking_data import load_auc_data, compute_ranks, algo_order_by_median_rank


def main():
    df = load_auc_data()
    df = compute_ranks(df)
    algo_order = algo_order_by_median_rank(df)

    rank_matrix = df.pivot(index="algorithm", columns="dataset", values="rank")
    auc_matrix = df.pivot(index="algorithm", columns="dataset", values="auc")

    rank_matrix = rank_matrix.loc[algo_order]
    auc_matrix = auc_matrix.loc[algo_order]

    # Sort datasets by average rank for visual coherence
    col_order = rank_matrix.mean(axis=0).sort_values().index
    rank_matrix = rank_matrix[col_order]
    auc_matrix = auc_matrix[col_order]

    # Build custom hover text
    hover_text = []
    for algo in rank_matrix.index:
        row = []
        for dataset in rank_matrix.columns:
            r = rank_matrix.loc[algo, dataset]
            a = auc_matrix.loc[algo, dataset]
            if str(r) == "nan":
                row.append(f"Algorithm: {algo}<br>Dataset: {dataset}<br>N/A")
            else:
                row.append(
                    f"Algorithm: {algo}<br>Dataset: {dataset}"
                    f"<br>Rank: {int(r)}<br>AUC: {a:.4f}"
                )
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=rank_matrix.values,
        x=rank_matrix.columns.tolist(),
        y=rank_matrix.index.tolist(),
        hovertext=hover_text,
        hoverinfo="text",
        colorscale="YlGnBu_r",
        colorbar=dict(title="Rank<br>(1 = best)"),
        zmin=1,
        zmax=rank_matrix.max().max(),
    ))

    fig.update_layout(
        title="Algorithm ranking heatmap across datasets (peptide precision AUC)",
        xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        width=1400,
        height=600,
        margin=dict(b=200),
    )

    out = "plots/algorithm_ranking_heatmap.html"
    fig.write_html(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
