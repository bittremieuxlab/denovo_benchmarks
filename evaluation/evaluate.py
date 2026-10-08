"""Evaluating collected algorithms predictions with respect to the 
ground truth labels."""

import argparse
import os
import shutil
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from functools import partial
from sklearn.metrics import auc

from . import utils
from . import mmseqs
from . import ground_truth_mapper
from .results import save_results
from .spectrum_prediction import (
    N_CALIBRATION_PSMS,
    FRAGMENT_MASS_TOL,
    WINDOW_SIZE,
    get_intensity_model_mods,
    get_RT_model_mods,
    map_mods_delta_mass_to_unimod,
    check_supported_by_model,
    predict_intensities,
    predict_RT,
    get_calibration_model,
    calculate_SA,
)
from .metrics import aa_match_metrics, aa_match_batch
from .token_masses import AA_MASSES


DATASET_TAGS_PATH = os.environ['DATASET_TAGS_PATH'] 
PROTEOMES_DIR = os.environ['PROTEOMES_DIR']
MMSEQS2_ARGS = [
    "--seed-sub-mat VTML40.out",
    "--comp-bias-corr 0 --mask 0",
    "-k 5",
    "--spaced-kmer-mode 0",
    "--exact-kmer-matching 1",
]


parser = argparse.ArgumentParser()
parser.add_argument(
    "output_root_dir",
    help="""
    The path to the output directory containing all tools predictions, 
    stored in `algorithm_name/algorithm_version/dset_name/output.csv` files.
    """,
)
parser.add_argument(
    "data_dir", help="The path to the input data with ground truth labels."
)
parser.add_argument(
    "--results_dir",
    default="results/",
    help="The path to save evaluation results (default: 'results/').",
)
parser.add_argument(
    "--skip_proteome_matches",
    help="Skip calculation of proteome matches.",
    action="store_true",
)
parser.add_argument(
    "--algorithms",
    nargs="+",
    default=None,
    help="""
    Only evaluate these algorithms, given as `algorithm_name` (all versions)
    or `algorithm_name:algorithm_version`.
    """,
)
parser.add_argument(
    "--quiet",
    help="Hide MMseqs2 output unless an MMseqs2 command fails.",
    action="store_true",
)
args = parser.parse_args()

# Parse selected algorithms: {algorithm_name: algorithm_version or None (all versions)}
selected_algorithms = None
if args.algorithms is not None:
    selected_algorithms = {}
    for algorithm in args.algorithms:
        algo_name, _, algo_version = algorithm.partition(":")
        selected_algorithms.setdefault(algo_name, set()).add(algo_version or None)


# Define dataset name and path to store evaluation results
dataset_name = os.path.basename(os.path.normpath(args.data_dir))
print(f"Evaluating results for {dataset_name}.")
print(f"Skip proteome matches metric: {args.skip_proteome_matches}.")

# Get dataset tags and database_path (proteome column, by dataset_name)
tags_df = pd.read_csv(DATASET_TAGS_PATH, sep='\t').set_index("dataset")
database_path = tags_df.loc[dataset_name, "proteome"]
dataset_tags = tags_df.loc[dataset_name]
dataset_tags = tuple(dataset_tags.index[dataset_tags == 1])
# Get corresponding spectrum prediction models and supported modifications
model_name_I, supported_mods_I = get_intensity_model_mods(dataset_tags)
model_name_rt, supported_mods_rt = get_RT_model_mods(dataset_tags)
print("Use prediction models:")
print(f"- Intensity: {model_name_I}, supported PTMs: {supported_mods_I}")
print(f"- RT: {model_name_rt}, supported PTMs: {supported_mods_rt}\n")

if not args.skip_proteome_matches:
    # Create directories for MMseqs2 proteome matches search
    (
        search_tmp_dir, tmp_files_dir, target_db_dir, query_db_dir, search_result_dir, search_result_path
    ) = mmseqs.setup_mmseqs_dirs(search_tmp_dir = "./mmseqs2_tmp")
    # create a database from a reference proteome
    reference_proteome_path = os.path.join(PROTEOMES_DIR, database_path)
    contam_path = os.path.join(PROTEOMES_DIR, "crap.fasta")
    target_fasta_path = os.path.join(search_tmp_dir, "proteome.fasta")
    mmseqs.create_target_fasta(reference_proteome_path, contam_path, target_fasta_path)


# Load GT peptide labels
labels_path = os.path.join(args.data_dir, "labels.csv")
sequences_true = pd.read_csv(labels_path)
sequences_true["seq"] = sequences_true["seq"].apply(ground_truth_mapper.format_sequence)

# Get experimental spectra params
dataset_path = os.path.join(args.data_dir, "mgf")
spectra_params = utils.extract_spectra_params(dataset_path)

# Predict intensities and RT for GT peptides
# TODO: maybe find a better name for this dataframe (true_psms is misleading?)
true_psms = sequences_true.join(spectra_params[
    ["spectrum_id", "charge", "precursor_mass", "true_RT", "filename", "idx", "run"]
].set_index("spectrum_id"), on="spectrum_id")
true_psms["seq_unimod"] = true_psms["seq"].apply(map_mods_delta_mass_to_unimod) # only for Prosit models(? for MS2PIP too?)
# true_psms["seq_unimod"] = true_psms["seq"].copy() # for other models, keep original format

print("DEBUG: all spectra params: spectra_params", spectra_params.shape)
print("DEBUG: DB annotated GT peptides: sequences_true", sequences_true.shape)
print("DEBUG: GT peptides with spectra params: true_psms", true_psms.shape, "\n")

# Find predicted sequences supported by the intensity prediction model
true_psms_supported_I_idx = true_psms.apply(
    lambda row: check_supported_by_model(row["seq_unimod"], row["charge"], supported_mods_I),
    axis=1,
)
# Find predicted sequences supported by the RT prediction model
true_psms_supported_rt_idx = true_psms.apply(
    lambda row: check_supported_by_model(row["seq_unimod"], row["charge"], supported_mods_rt),
    axis=1,
)
print("DEBUG: GT peptides supported by intensity prediction model")
print(true_psms_supported_I_idx.value_counts(), "\n")
print("DEBUG: GT peptides supported by RT prediction model")
print(true_psms_supported_rt_idx.value_counts(), "\n")

# Get intensity predictions for GT peptides
gt_predictions_mz, gt_predictions_I = predict_intensities(
    model_name_I,
    true_psms[true_psms_supported_I_idx].rename({"seq_unimod": "sequence"}, axis=1),
)
# Get RT predictions for GT peptides and store them in the dataframe
true_psms.loc[true_psms_supported_rt_idx, "pred_RT"] = predict_RT(
    model_name_rt,
    true_psms[true_psms_supported_rt_idx].rename({"seq_unimod": "sequence"}, axis=1),
)

# Calculate spectral angles
print("Calculate spectral angles for GT peptides")
true_psms["SA"] = np.nan
for filename in true_psms["filename"].value_counts().index:
    print("FIle:", filename)
    # Calculate spectral angles (with original spectra loaded from mgf file)
    true_psms_file_mask_I = (true_psms["filename"] == filename) & true_psms_supported_I_idx
    # spec_idx: index - psm idx in dataframe, value - 0-based spectrum idx in mgf file
    spec_idxs = true_psms[true_psms_file_mask_I]["idx"].astype(np.int64)
    # Load mgf file, iterate through experimental spectra, calculate SA
    mgf_path = os.path.join(dataset_path, filename + ".mgf")
    true_psms.loc[true_psms_file_mask_I, "SA"] = calculate_SA(spec_idxs, gt_predictions_mz, gt_predictions_I, mgf_path)
print()

# Calculate RT differences (with calibrated RT)
print("Calculate RT differences for GT peptides")
true_psms["true_RT_calib"] = np.nan
spectra_params["true_RT_calib"] = np.nan

# Calibrate true RT to iRT based on GT PSMs from the same run
for run in true_psms["run"].value_counts().index:
    print("Run:", run)
    # Select calibration PSMs (from GT PSMs)
    true_psms_run_mask_rt = (true_psms["run"] == run) & true_psms_supported_rt_idx
    calib_psms = true_psms[true_psms_run_mask_rt]
    calib_psms = calib_psms.sample(n=min(N_CALIBRATION_PSMS, len(calib_psms)), replace=False, random_state=0)
    # Train calibration model (for this particular file)
    rt_calib_reg = get_calibration_model(calib_psms)
    # Calibrate true_RT to iRT
    true_psms.loc[true_psms_run_mask_rt, "true_RT_calib"] = rt_calib_reg.predict(
        true_psms.loc[true_psms_run_mask_rt, "true_RT"].values[:, None]
    )[:, 0]
    # Calibrate true_RT for all spectra in the file
    all_spectra_run_mask = (spectra_params["run"] == run)
    spectra_params.loc[all_spectra_run_mask, "true_RT_calib"] = rt_calib_reg.predict(
        spectra_params.loc[all_spectra_run_mask, "true_RT"].values[:, None]
    )[:, 0]

# Calibrate true_RT for remaining spectra (w/o GT PSMs from the same run)
# (with our FDR criteria, it can happen that some runs don't have any DB PSMs)
if spectra_params["true_RT_calib"].isnull().any():
    print("Spectra with no GT PSMs from the same run:", spectra_params["true_RT_calib"].isnull().sum())
    # Select calibration PSMs (from GT PSMs in all files)
    calib_psms = true_psms[true_psms_supported_rt_idx]
    calib_psms = calib_psms.sample(n=min(N_CALIBRATION_PSMS, len(calib_psms)), replace=False, random_state=0)
    # Train calibration model (for this particular file)
    rt_calib_reg = get_calibration_model(calib_psms)
    # Calibrate true_RT for all spectra (spectra_params)
    all_spectra_run_mask = (spectra_params["true_RT_calib"].isnull())
    spectra_params.loc[all_spectra_run_mask, "true_RT_calib"] = rt_calib_reg.predict(
        spectra_params.loc[all_spectra_run_mask, "true_RT"].values[:, None]
    )[:, 0]

# Calculate RT differences (on calibrated RT)
max_true_irt = true_psms["true_RT_calib"].max()
true_psms["RT_diff"] = (true_psms["pred_RT"] - true_psms["true_RT_calib"]).abs() / max_true_irt

# Evaluate every algorithm & prepare plotting data. 
# Create plots and metrics for ALL algorithms for a given dataset.
# - skip algorithm that doesn't have output.csv or SA/pred_RT columns

# args.data_dir = path/to/dataset/folder # with labels.csv and mgf/
# args.output_dir="$output_root_dir/$algorithm_name/$algorithm_version/$dset_name"

output_metrics = {}

# Number of points to represent the curve (for all algorithms)
PLOT_N_POINTS = 200

def _downsample_curve(coverage, metric):
    """Downsample coverage and metric arrays to PLOT_N_POINTS."""
    if len(coverage) == 0:
        return np.empty(0), np.empty(0)
    plot_idxs = np.linspace(0, len(coverage) - 1, min(PLOT_N_POINTS, len(coverage))).astype(np.int64)
    return coverage[plot_idxs], metric[plot_idxs]

def _append_plot_data(plot_dict, algo_name, algo_version, coverage, metric, **extras):
    """Append downsampled plot data to a plot dictionary (in-place)."""
    plot_dict["algorithm"].append(algo_name)
    plot_dict["version"].append(algo_version)
    plot_dict["coverage"].append(coverage)
    plot_dict["metric"].append(metric)
    for key, value in extras.items():
        if key in plot_dict:
            plot_dict[key].append(value)

# for each plot, collect a dataframe containing 
# dataset, algorithm (algo_name), algo_version, coverage, metric_value (at given coverage)
# coverage and metric value are stored as lists of values
aa_precision_plot_data = {"algorithm": [], "version": [], "coverage": [], "metric": [], "auc": []}
pep_precision_plot_data = {"algorithm": [], "version": [], "coverage": [], "metric": [], "auc": []}
n_proteome_matches_plot_data = {"algorithm": [], "version": [], "coverage": [], "metric": []} # TODO: only if not args.skip_proteome_matches?
rt_diff_plot_data = {"algorithm": [], "version": [], "coverage": [], "metric": []}
sa_plot_data = {"algorithm": [], "version": [], "coverage": [], "metric": []}
# TODO: do we want to add also denovo_rt_diff_plot_data and denovo_SA_plot_data (only denovo peptides without GT)?

for algo_name in os.listdir(args.output_root_dir):
    algo_dir = os.path.join(args.output_root_dir, algo_name)
    if not os.path.isdir(algo_dir):
        continue
    if selected_algorithms is not None and algo_name not in selected_algorithms:
        continue

    for algo_version in os.listdir(algo_dir):
        version_dir = os.path.join(algo_dir, algo_version)
        if not os.path.isdir(version_dir):
            continue
        if selected_algorithms is not None and not (
            None in selected_algorithms[algo_name] or algo_version in selected_algorithms[algo_name]
        ):
            continue

        full_algo_name = f"{algo_name}_{algo_version}"
        print("EVALUATE", full_algo_name)

        output_path = os.path.join(version_dir, dataset_name, "output.csv")
        if not os.path.isfile(output_path):
            print(f"Predictions file not found for {algo_name} {algo_version}, skipping...")
            continue

        with open(output_path) as f:
            header = f.readline().strip().split(",")
        if "SA" not in header or "pred_RT" not in header:
            print("Output file columns:", header)
            print(f"Predictions do not contain SA and predicted RT, skipping...")
            continue

        # Load tool predictions, match with ground truth
        output_data = utils.load_predictions(output_path, sequences_true)
        print("DEBUG: predicted & GT dataframe: output_data", output_data.shape)

        # Add experimental spectra data (covering all spectra in the dataset)
        output_data = output_data.join(spectra_params.set_index("spectrum_id"), on="spectrum_id", how="outer").reset_index(drop=True) # how="left" is default
        print("DEBUG: predicted with spectra_params added: output_data", output_data.shape)

        # Get idxs of GT labeled peptides & sequenced peptides (in correct output format)
        print("NaN sequences:", output_data["score"].isnull().sum())
        output_data = output_data.sort_values("score", ascending=False)
        n_spectra = len(output_data)
        labeled_idx = output_data["sequence_true"].notnull()
        sequenced_idx = utils.get_sequenced_idx(output_data)
        output_data.loc[~sequenced_idx, "sequence"] = ""
        output_data.loc[~sequenced_idx, "aa_scores"] = ""
        print("DEBUG: n_spectra (used as total number of spectra)", n_spectra)
        print("DEBUG: n_sequenced (number of de novo predicted peptides)", sequenced_idx.sum())

        # Find predicted sequences supported by the model (Prosit or other), assuming SA and pred_RT precalculated
        supported_I_idx = output_data["SA"].notnull()
        supported_rt_idx = output_data["pred_RT"].notnull()
        # Calculate RT differences (on calibrated RT) and normalize by max iRT
        output_data["RT_diff"] = (output_data["pred_RT"] - output_data["true_RT_calib"]).abs() / max_true_irt

        # Calculate amino acid and peptide-level precision and recall
        # Prepare output sequences for metrics calculation
        output_data.loc[sequenced_idx, ["sequence", "aa_scores"]] = output_data.loc[sequenced_idx].apply(
            lambda row: utils.ptms_to_delta_mass(row["sequence"], row["aa_scores"]),
            axis=1,
            result_type="expand",
        ).values
        # Calculate metrics (aa precision, recall, peptide precision)
        aa_matches_batch, n_aa_pred, n_aa_true = aa_match_batch(
            output_data["sequence"][labeled_idx],
            output_data["sequence_true"][labeled_idx],
            AA_MASSES,
        )
        aa_precision, aa_recall, pep_precision = aa_match_metrics(aa_matches_batch, n_aa_true, n_aa_pred)

        if not args.skip_proteome_matches:
            # Calculate number of proteome matches
            # Create query sequences without any modifications and I/L indistinguishable
            output_data.loc[sequenced_idx, "sequence_no_ptm"] = output_data.loc[sequenced_idx, "sequence"].apply(
                partial(utils.remove_ptms, ptm_pattern='[^A-Z]')
            )
            output_data.loc[sequenced_idx, "sequence_query"] = output_data.loc[sequenced_idx, "sequence_no_ptm"].apply(
                mmseqs.isoleucine_to_leucine
            )
            print(
                output_data["sequence"].value_counts().size, 
                output_data["sequence_no_ptm"].value_counts().size,
                output_data["sequence_query"].value_counts().size
            )
            # Write the list of unique sequences to the query database
            unique_sequences = output_data["sequence_query"].unique().tolist()
            query_fasta_path = os.path.join(search_tmp_dir, "denovo_predicted_peptides.fasta")
            mmseqs.create_query_fasta(unique_sequences, query_fasta_path)
            # Run mmseqs search
            search_df = mmseqs.run_mmseqs(
                target_fasta_path,
                query_fasta_path,
                target_db_dir,
                query_db_dir,
                search_result_dir,
                search_result_path,
                tmp_files_dir,
                args=MMSEQS2_ARGS,
                quiet=args.quiet,
            )
            # Map matches back to original de novo sequences
            matched_sequences = search_df["qseq"].tolist() 
            output_data["proteome_match"] = False
            output_data.loc[sequenced_idx, "proteome_match"] = output_data.loc[sequenced_idx, "sequence_query"].isin(matched_sequences)
            n_proteome_matches = output_data["proteome_match"].sum()

        # [Debug] Check number of GT peptide matches
        pep_matches = np.array([aa_match[1] for aa_match in aa_matches_batch])
        output_data["pep_match"] = False
        output_data.loc[labeled_idx, "pep_match"] = pep_matches
        
        # Collect metrics
        output_metrics[full_algo_name] = {
            "N sequences": sequenced_idx.size,
            "N predicted": sequenced_idx.sum(), # how many sequences were predicted by de novo tool
            "AA precision": aa_precision,
            "AA recall": aa_recall,
            "Pep precision": pep_precision,
            "N proteome matches": n_proteome_matches if not args.skip_proteome_matches else None,
        }

        # Collect plotting data
        # RT difference curve
        rt_diff = output_data[supported_rt_idx].sort_values("score", ascending=False)["RT_diff"]
        rt_diff_wma = np.convolve(rt_diff, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
        coverage = np.arange(1, len(rt_diff_wma) + 1) / n_spectra
        coverage, rt_diff_wma = _downsample_curve(coverage, rt_diff_wma)
        _append_plot_data(
            rt_diff_plot_data, algo_name, algo_version, 
            coverage.tolist(), rt_diff_wma.tolist()
        )

        # SA curve
        SA = output_data[supported_I_idx].sort_values("score", ascending=False)["SA"]
        SA_wma = np.convolve(SA, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
        coverage = np.arange(1, len(SA_wma) + 1) / n_spectra
        coverage, SA_wma = _downsample_curve(coverage, SA_wma)
        _append_plot_data(
            sa_plot_data, algo_name, algo_version, 
            coverage.tolist(), SA_wma.tolist()
        )

        # Proteome matches vs number of predictions curve
        if not args.skip_proteome_matches:
            prot_matches = output_data["proteome_match"][sequenced_idx].values
            n_matches = np.cumsum(prot_matches)
            coverage = np.arange(1, sequenced_idx.sum() + 1) / n_spectra
            coverage, n_matches = _downsample_curve(coverage, n_matches)
            _append_plot_data(
                n_proteome_matches_plot_data, algo_name, algo_version, 
                coverage.tolist(), n_matches.tolist()
            )

        # Peptide precision-coverage curve
        pep_matches = np.array([aa_match[1] for aa_match in aa_matches_batch])
        precision = np.cumsum(pep_matches) / np.arange(1, len(pep_matches) + 1)
        coverage = np.arange(1, len(pep_matches) + 1) / len(pep_matches)
        coverage, precision = _downsample_curve(coverage, precision)
        _append_plot_data(
            pep_precision_plot_data, algo_name, algo_version, 
            coverage.tolist(), precision.tolist(), 
            auc=auc(coverage, precision)
        )

        # Amino acid precision-coverage curve
        aa_scores = np.concatenate(list(map(utils.parse_scores, output_data["aa_scores"][labeled_idx].values.tolist())))
        sort_idx = np.argsort(aa_scores)[::-1]
        aa_matches_pred = np.concatenate([aa_match[2][0] for aa_match in aa_matches_batch])
        precision = np.cumsum(aa_matches_pred[sort_idx]) / np.arange(1, len(aa_matches_pred) + 1)
        coverage = np.arange(1, len(aa_matches_pred) + 1) / len(aa_matches_pred)
        coverage, precision = _downsample_curve(coverage, precision)
        _append_plot_data(
            aa_precision_plot_data, algo_name, algo_version, 
            coverage.tolist(), precision.tolist(), 
            auc=auc(coverage, precision) if len(coverage) > 0 else 0
        )
        
        if not args.skip_proteome_matches:
            # [Debug] display number of peptide matches & proteome matches
            print("DEBUG: N GT peptide matches:", output_data["pep_match"].sum())
            print("DEBUG: N proteome matches:", n_proteome_matches)
            idx = output_data["pep_match"] & ~output_data["proteome_match"]
            print("DEBUG: GT peptide matches w/o proteome match:", idx.sum())
            idx = ~output_data["pep_match"] & output_data["proteome_match"]
            print("DEBUG: Proteome matches w/o GT peptide match:", idx.sum())
        
        print("\n", "=" * 100, "\n")

# Database search baseline n_proteome_matches plot
if not args.skip_proteome_matches:
    gt_n_prot_matches = len(true_psms) # total number of database search annotations
    _append_plot_data(
        n_proteome_matches_plot_data, 
        algo_name="database search", 
        algo_version="", # no version for database search 
        coverage=[0., 1.], # full coverage 
        metric=[gt_n_prot_matches, gt_n_prot_matches],
    )
    # FIXME: coverage=[0., 1.] - full coverage, line crosses the entire plot, 
    # and doesn't reflect the real number of DB annotations.
    # Alternative would be: coverage=[0., n_gt_sequences / n_spectra]


# TODO: add database search baseline to RT_diff plot
# gt_rt_diff = true_psms[true_psms_supported_rt_idx]["RT_diff"]
# gt_rt_diff_wma = np.convolve(gt_rt_diff, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
# gt_coverage = np.arange(1, len(gt_rt_diff_wma) + 1) / len(gt_rt_diff_wma)
# plot_idxs = np.linspace(0, len(gt_coverage) - 1, PLOT_N_POINTS).astype(np.int64)
# # _append_plot_data(rt_diff_plot_data, algo_name, algo_version, gt_coverage, gt_rt_diff_wma)
# axs[3].plot(gt_coverage[plot_idxs], gt_rt_diff_wma[plot_idxs], label="database search", color="k")

# TODO: add shuffled baseline to RT_diff plot
# true_RT_shuffled = np.random.permutation(true_psms.loc[true_psms_supported_rt_idx, "true_RT_calib"])
# rt_shuffled_diff = (true_psms.loc[true_psms_supported_rt_idx, "true_RT_calib"] - true_RT_shuffled).abs() / max_true_irt
# rt_shuffled_diff_wma = np.convolve(rt_shuffled_diff, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
# plot_idxs = np.linspace(0, len(gt_coverage) - 1, PLOT_N_POINTS).astype(np.int64)
# axs[3].plot(gt_coverage[plot_idxs], rt_shuffled_diff_wma[plot_idxs], label=f"random baseline", color="tab:gray")

# TODO: add database search baseline to SA plot
# gt_SA = true_psms[true_psms_supported_I_idx]["SA"]
# gt_sa_wma = np.convolve(gt_SA, np.ones(WINDOW_SIZE) / WINDOW_SIZE, mode='valid')
# gt_coverage = np.arange(1, len(gt_sa_wma) + 1) / len(gt_sa_wma)
# plot_idxs = np.linspace(0, len(gt_coverage) - 1, PLOT_N_POINTS).astype(np.int64)
# axs[4].plot(gt_coverage[plot_idxs], gt_sa_wma[plot_idxs], label="database search", color="k")

# Save results
dataset_results_dir = os.path.join(args.results_dir, dataset_name)
os.makedirs(dataset_results_dir, exist_ok=True)

# (merge into existing results files: rows of evaluated algorithms are replaced, other rows are kept)
plot_key_cols = ["algorithm", "version"]
pep_precision_plot_data = pd.DataFrame(pep_precision_plot_data)
save_results(pep_precision_plot_data, os.path.join(dataset_results_dir, "peptide_precision_plot_data.csv"), plot_key_cols)
aa_precision_plot_data = pd.DataFrame(aa_precision_plot_data)
save_results(aa_precision_plot_data, os.path.join(dataset_results_dir, "AA_precision_plot_data.csv"), plot_key_cols)
if not args.skip_proteome_matches:
    n_proteome_matches_plot_data = pd.DataFrame(n_proteome_matches_plot_data)
    save_results(n_proteome_matches_plot_data, os.path.join(dataset_results_dir, "number_of_proteome_matches_plot_data.csv"), plot_key_cols)
rt_diff_plot_data = pd.DataFrame(rt_diff_plot_data)
save_results(rt_diff_plot_data, os.path.join(dataset_results_dir, "RT_difference_plot_data.csv"), plot_key_cols)
sa_plot_data = pd.DataFrame(sa_plot_data)
save_results(sa_plot_data, os.path.join(dataset_results_dir, "SA_plot_data.csv"), plot_key_cols)

output_metrics = pd.DataFrame(output_metrics).T
save_results(output_metrics, os.path.join(dataset_results_dir, "metrics.csv"))


if not args.skip_proteome_matches:
    # Clean tmp folders
    shutil.rmtree(search_tmp_dir)
