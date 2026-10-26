import os
import ast
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


RESULTS_DIR = "results"
PLOT_HEIGHT = 500
# PLOT_WIDTH = 650

st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(["Main", "Adding an algorithm"])

@st.cache_data
def load_all_datasets():
    """Load metadata for all available datasets."""
    datasets = []
    if not os.path.exists(RESULTS_DIR):
        return []
    for dataset_name in os.listdir(RESULTS_DIR):
        dataset_path = os.path.join(RESULTS_DIR, dataset_name)
        if os.path.isdir(dataset_path):
            datasets.append(dataset_name)
    return sorted(datasets)


@st.cache_data
def load_plot_data(dataset_name, metric_name):
    """Load plot data CSV for a specific dataset and metric."""
    csv_path = os.path.join(RESULTS_DIR, dataset_name, f"{metric_name}_plot_data.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    # Parse list strings back to actual lists
    df["coverage"] = df["coverage"].apply(ast.literal_eval)
    df["metric"] = df["metric"].apply(ast.literal_eval)
    return df


def get_latest_versions(df):
    """Get the latest version for each algorithm."""
    latest = {}
    for algo in df["algorithm"].unique():
        versions = df[df["algorithm"] == algo]["version"].unique()
        # Prefer 'latest' if present, otherwise take the last one alphabetically
        if "latest" in versions:
            latest[algo] = "latest"
        else:
            latest[algo] = sorted(versions)[-1]
    return latest


def create_plot(df, selected_versions, title, xaxis_title="Coverage", yaxis_title="Precision", show_auc=False):
    """Create a Plotly figure from plot data."""
    fig = go.Figure()
    
    for _, row in df.iterrows():
        algo_name = row["algorithm"]
        algo_version = row["version"]
        
        # Skip if this version is not selected for this algorithm
        if algo_name not in selected_versions or algo_version not in selected_versions[algo_name]:
            continue
        
        # Create trace label
        label = f"{algo_name} {algo_version}" if algo_version != "latest" else algo_name
        if show_auc and "auc" in row and pd.notna(row["auc"]):
            label += f" AUC = {row['auc']:.3f}"
        
        fig.add_trace(
            go.Scatter(
                x=row["coverage"],
                y=row["metric"],
                mode="lines",
                name=label,
            )
        )
    
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=PLOT_HEIGHT,
        # width=PLOT_WIDTH,
        legend=dict(
            y=0.01,
            x=0.01,
            bgcolor="rgba(255,255,255,0.6)",
            font=dict(size=10),
        ),
        margin=dict(t=50, b=50, l=50, r=20),
    )
    
    # Set axis ranges for precision/coverage plots
    if "Precision" in yaxis_title or "Coverage" in xaxis_title:
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
    
    return fig


with tab1:
    st.title("*De novo* Benchmarks")

    st.header("Info")
    st.markdown(
        """
        *De novo* peptide sequencing identifies peptides from mass spectrometry data
        without a reference proteome database. 
        It is essential for discovering novel peptides and in applications 
        where database construction is complicated (immunopeptidomics, metabolomics).
        However, evaluation and comparison of multiple existing methods is challenging
        due to the lack of standardized metrics and universal test datasets.
       
        This project aims to provide a unified framework for the comprehensive benchmarking 
        of *de novo* peptide sequencing algorithms.
        It performs evaluation on both publicly available and internal datasets, 
        encompassing different species, variations in cleavage enzymes (tryptic and non-tryptic data), 
        and various post-translational modifications (PTMs), 
        as well as specific proteomics subfields such as immunopeptidomics. 
        Each algorithm is run in an isolated environment using Apptainer containers 
        to ensure reproducibility and consistency.

        Current benchmarking results are represented below. 
        """
    )

    st.header("Benchmarking results")
    st.markdown(
        """
        Results are split by the evaluation dataset. 

        Ground truth labels were obtained by running database search 
        with [MSFragger](https://msfragger.nesvilab.org/) and subsequence rescoring 
        with [Percolator](http://percolator.ms/) using features from Prosit-predicted spectra.  
        Unless otherwise specified, PSMs with estimated FDR (Percolator q-value) **<1%** 
        were selected as ground truth peptides.

        The plots demonstrate performance of *de novo* peptide sequencing algorithms according to **5** metrics:
        - Peptide prediction precision and coverage
        - Amino acid prediction precision and coverage
        - Number of proteome matches vs. number of predictions
        - RT difference between predicted and experimental retention time
        - Spectral angle between predicted and experimental spectra

        Click on algorithm names in the legend to hide/show specific algorithms. Use the plot controls to zoom and pan.
        """
    )
    
    # Load all available datasets
    all_datasets = load_all_datasets()
    
    if not all_datasets:
        st.warning("No results found in the results directory.")
    else:
        # Dataset filter
        st.subheader("Dataset selection")
        selected_datasets = st.multiselect(
            "Select datasets to display:",
            options=all_datasets,
            default=all_datasets,
            help="Choose which datasets to show in the dashboard"
        )
        
        if not selected_datasets:
            st.info("Please select at least one dataset to display.")
        
        # Display each selected dataset
        for dataset_name in selected_datasets:
            st.divider()
            st.subheader(dataset_name)
            
            # Load one metric to get algorithm/version info
            sample_data = load_plot_data(dataset_name, "peptide_precision")
            
            if sample_data is None or sample_data.empty:
                st.warning(f"No plot data found for {dataset_name}")
                continue
            
            # Get available algorithms and their versions
            algorithms = sorted(sample_data["algorithm"].unique())
            latest_versions = get_latest_versions(sample_data)
            
            # Version selection per dataset (use expander to save space)
            with st.expander("⚙️ Algorithm version selection", expanded=False):
                st.caption("Select which versions to display for each algorithm (default: latest)")
                
                # Create a grid of version selectors
                n_cols = 3
                cols = st.columns(n_cols)
                
                selected_versions = {}
                for idx, algo in enumerate(algorithms):
                    col_idx = idx % n_cols
                    with cols[col_idx]:
                        versions = sorted(sample_data[sample_data["algorithm"] == algo]["version"].unique())
                        default_version = latest_versions[algo]
                        selected_versions[algo] = st.multiselect(
                            f"{algo}",
                            options=versions,
                            default=[default_version] if default_version in versions else versions[:1],
                            key=f"{dataset_name}_{algo}_version"
                        )
            
            # If not using expander, use default latest versions
            if not selected_versions:
                selected_versions = {algo: [latest_versions[algo]] for algo in algorithms}
            
            # Define metrics to plot
            metrics = [
                ("peptide_precision", "Peptide precision & coverage", "Coverage", "Precision", True),
                ("AA_precision", "AA precision & coverage", "Coverage", "Precision", True),
                ("number_of_proteome_matches", "Number of proteome matches\nvs. number of peptides", "Number of predicted peptides", "Number of matches", False),
                ("RT_difference", "Absolute difference between\npredicted and experimental RT", "Coverage", "RT difference", False),
                ("SA", "Spectral angle between\npredicted and experimental spectra", "Coverage", "Spectral angle", False),
            ]
            
            # Create tabs for each metric to avoid width issues
            metric_names = [m[1].replace('\n', ' ') for m in metrics]
            plot_tabs = st.tabs(metric_names)
            
            for idx, (metric_name, title, x_label, y_label, show_auc) in enumerate(metrics):
                plot_data = load_plot_data(dataset_name, metric_name)
                
                with plot_tabs[idx]:
                    if plot_data is not None and not plot_data.empty:
                        fig = create_plot(
                            plot_data,
                            selected_versions,
                            title,
                            xaxis_title=x_label,
                            yaxis_title=y_label,
                            show_auc=show_auc
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"{dataset_name}_{metric_name}_plot")
                    else:
                        st.info(f"No data available for {title}")


with tab2:
    st.header("Adding a new algorithm")
    # st.divider()

    st.markdown(
        """
        Make a pull request on [Github](https://github.com/PominovaMS/denovo_benchmarks) 
        to add your algorithm to the benchmarking system.
        """
    )

    main, sidebar = st.columns([2, 1])

    with main:
        st.markdown(
            """
            Add your algorithm in the `denovo_benchmarks/algorithms/algorithm_name` folder by providing  
            `container.def`, `make_predictions.sh`, `input_mapper.py`, `output_mapper.py` files.  
            Detailed files descriptions are given below.  

            Templates for each file implementation can be found in the 
            `algorithms/base/` [folder](https://github.com/PominovaMS/denovo_benchmarks/tree/main/algorithms/base).  
            It also includes the `InputMapperBase` and `OutputMapperBase` base classes for implementing input and output mappers.  
            For examples, you can check 
            [Casanovo](https://github.com/PominovaMS/denovo_benchmarks/tree/main/algorithms/casanovo) 
            and [DeepNovo](https://github.com/PominovaMS/denovo_benchmarks/tree/main/algorithms/deepnovo) implementations. 
            """
        )

        st.subheader("Files description")
        st.markdown(
            """
            - **`container.def`** — definition file of the [Apptainer](https://apptainer.org/docs/user/main/definition_files.html) 
            container image that creates environment and installs dependencies required for running the algorithm.
                
            - **`make_predictions.sh`** — bash script to run the *de novo* algorithm on the input dataset 
            (folder with MS spectra in .mgf files) and generate an output file with per-spectrum peptide predictions.  
                **Input**: path to a dataset folder containing .mgf files with spectra data  
                **Output**: output file (in a common output format) containing predictions for all spectra in the dataset

                To configure the model for specific data properties (e.g. non-tryptic data, data from a particular instrument, etc.), please use **dataset tags**. 
                Current set of tags can be found in the `DatasetTag` in [dataset_config.py](https://github.com/PominovaMS/denovo_benchmarks/blob/main/dataset_config.py) and includes `nontryptic`, `timstof`, `waters`, `sciex`.
                Example usage can be found in `algorithms/base/make_predictions_template.sh`.

            - **`input_mapper.py`** — python script to convert input data 
            from its original representation (**input format**) to the format expected by the algorithm.

                **Input format**
                - Input: a dataset folder with separate .mgf files containing MS spectra with annotations.
                - Keys order for a spectrum in .mgf file:  
                `[TITLE, RTINSECONDS, PEPMASS, CHARGE]` 
            ` `  

            - **`output_mapper.py`** — python script to convert the algorithm output to the common **output format**.

            **Output format**
            - .csv file (with `sep=","`)
            - must contain columns:
                - `"sequence"` — predicted peptide sequence, written in the predefined **output sequence format**
                - `"score"` — *de novo* algorithm "confidence" score for a predicted sequence
                - `"aa_scores"` — per-amino acid scores, if available. If not available, the whole peptide `score` will be used as a score for each amino acid.
                - `"spectrum_id"` — information to match each prediction with its ground truth sequence.  
                    `{filename}:{index}` string, where  
                    `filename` — name of the .mgf file in a dataset,  
                    `index` —  index (0-based) of each spectrum in an .mgf file.
                    ` `  
                
                - **Output sequence format**
                    - 20 amino acid tokens:  
                    `G, A, S, P, V, T, C, L, I, N, D, Q, K, E, M, H, F, R, Y, W`
                    - Amino acids with post-translational modifications (PTMs) are written in 
                    **[ProForma](https://github.com/HUPO-PSI/ProForma/tree/master) format** with **Unimod accession codes** for PTMs:  
                    `C[UNIMOD:4]` for Cysteine Carbamidomethylation, `M[UNIMOD:35]` for Methionine Oxidation, etc.
                    - N-terminus and C-terminus modifications, if supported by the algorithm, are also written in **ProForma notation** with **Unimod accession codes**:  
                    `[UNIMOD:xx]-PEPTIDE-[UNIMOD:yy]`
            """
        )
