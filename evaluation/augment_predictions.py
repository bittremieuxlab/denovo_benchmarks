"""Precompute parameters of de novo predicted spectra for faster evaluation."""

import argparse
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from functools import partial

from . import utils
from . import mmseqs
from .spectrum_prediction import (
    get_intensity_model_mods,
    get_RT_model_mods,
    check_supported_by_model,
    predict_intensities,
    predict_RT,
    calculate_SA,
)

USE_COLS = ["sequence", "score", "aa_scores", "spectrum_id", "pred_RT", "SA", "proteome_match"]
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
    "--output_dir",
    help="The path to the output directory containing the algorithm output.",
)
parser.add_argument(
    "--data_dir", help="The path to the input data directory with spectra in mgf/ subdirectory."
)
parser.add_argument(
    "--force",
    help="Force re-computation even if pred_RT and SA already exist in the output file.",
    action="store_true",
)
parser.add_argument(
    "--quiet",
    help="Hide MMseqs2 output unless an MMseqs2 command fails.",
    action="store_true",
)
args = parser.parse_args()


# output_dir="$output_root_dir/$algorithm_name/$algorithm_version/$dset_name" 
# always contains dataset_name as the last part
dataset_name = os.path.basename(os.path.normpath(args.output_dir))

# Get dataset tags (by dataset_name)
tags_df = pd.read_csv(DATASET_TAGS_PATH, sep='\t').set_index("dataset")
database_path = tags_df.loc[dataset_name, "proteome"]
dataset_tags = tags_df.loc[dataset_name]
dataset_tags = tuple(dataset_tags.index[dataset_tags == 1])
print(f"Use reference proteome: {database_path}\n")
# Get corresponding spectrum prediction models and supported modifications
model_name_I, supported_mods_I = get_intensity_model_mods(dataset_tags)
model_name_rt, supported_mods_rt = get_RT_model_mods(dataset_tags)
print("Use prediction models:")
print(f"- Intensity: {model_name_I}, supported PTMs: {supported_mods_I}")
print(f"- RT: {model_name_rt}, supported PTMs: {supported_mods_rt}\n")

output_path = os.path.join(args.output_dir, "output.csv")
output_data = pd.read_csv(output_path)

use_cols = [col for col in USE_COLS if col in output_data.columns]
output_data = output_data[use_cols]

# 1) Calculate the predicted RT and SA of the predicted spectrum. 
# Skip if output_data already has pred_RT and SA columns and not args.force
if not (("pred_RT" in output_data.columns) and ("SA" in output_data.columns)) or args.force:
    print("Calculating predicted RT and SA of the predicted spectrum...")

    # Add experimental spectra data (we only need charge and precursor mass)
    dataset_path = os.path.join(args.data_dir, "mgf")
    spectra_params = utils.extract_spectra_params(dataset_path)
    output_data = output_data.join(spectra_params.set_index("spectrum_id"), on="spectrum_id")
    # Find predicted sequences supported by the intensity prediction model
    # (Limit max number of modified residies as models can be unstable on peptides with too many modifications)
    supported_I_idx = output_data.apply(
        lambda row: check_supported_by_model(row["sequence"], row["charge"], supported_mods_I, max_number_mods=10),
        axis=1,
    )
    # Find predicted sequences supported by the RT prediction model
    supported_rt_idx = output_data.apply(
        lambda row: check_supported_by_model(row["sequence"], row["charge"], supported_mods_rt, max_number_mods=10),
        axis=1,
    )
    print("DEBUG: de novo peptides supported by intensity prediction model")
    print(supported_I_idx.value_counts(), "\n")
    print("DEBUG: de novo peptides supported by RT prediction model")
    print(supported_rt_idx.value_counts(), "\n")

    output_data["SA"] = np.nan
    if any(supported_I_idx):
        # Get intensity predictions for de novo peptides
        predictions_mz, predictions_I = predict_intensities(model_name_I, output_data[supported_I_idx])
        # Calculate spectral angles with experimental spectra
        for filename in output_data["filename"].value_counts().index:
            print(filename)
            file_mask = (output_data["filename"] == filename) & supported_I_idx
            # spec_idx: index - psm idx in dataframe, value - 0-based spectrum idx in mgf file
            spec_idxs = output_data[file_mask]["idx"].astype(np.int64)

            # Load mgf file, iterate through experimental spectra, calculate SA
            mgf_path = os.path.join(dataset_path, filename + ".mgf")
            output_data.loc[file_mask, "SA"] = calculate_SA(spec_idxs, predictions_mz, predictions_I, mgf_path)
    else:
        print("No de novo peptides supported by intensity prediction model. Skipping intensity and SA prediction.")

    output_data["pred_RT"] = np.nan
    if any(supported_rt_idx):
        # Get RT predictions for de novo peptides and store them in output_data
        output_data.loc[supported_rt_idx, "pred_RT"] = predict_RT(
            model_name_rt, output_data[supported_rt_idx], chunk_size=100000,
        )
    else:
        print("No de novo peptides supported by RT prediction model. Skipping RT prediction.")

else:
    print(f"Output data {output_path} already has pred_RT and SA columns. Skipping...")

# 2) Check matches to the reference proteome
if not ("proteome_match" in output_data.columns) or args.force:
    print("Checking matches to the reference proteome...")

    # Create directories for MMseqs2 proteome matches search
    # Use a unique tmp dir per job (per output_dir + random suffix) so that
    # parallel jobs don't collide on shared MMseqs2 DB/result files.
    job_tmp_dir = tempfile.mkdtemp(
        prefix=f"mmseqs2_tmp_{dataset_name}_", dir=os.path.abspath(args.output_dir)
    )
    (
        search_tmp_dir, tmp_files_dir, target_db_dir, query_db_dir, search_result_dir, search_result_path
    ) = mmseqs.setup_mmseqs_dirs(search_tmp_dir=job_tmp_dir)
    # create a database from a reference proteome
    reference_proteome_path = os.path.join(PROTEOMES_DIR, database_path)
    contam_path = os.path.join(PROTEOMES_DIR, "crap.fasta")
    target_fasta_path = os.path.join(search_tmp_dir, "proteome.fasta")
    mmseqs.create_target_fasta(reference_proteome_path, contam_path, target_fasta_path)

    # Get idxs of de novo sequences that have a predicted sequence
    sequenced_idx = utils.get_sequenced_idx(output_data)

    # Normalize PTM notation to delta masses (same as in evaluate.py)
    output_data.loc[sequenced_idx, ["sequence", "aa_scores"]] = output_data.loc[sequenced_idx].apply(
        lambda row: utils.ptms_to_delta_mass(row["sequence"], row["aa_scores"]),
        axis=1,
        result_type="expand",
    ).values

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
    print(f"{dataset_name}: {n_proteome_matches} de novo predicted sequences match the reference proteome.\n")

    # Clean tmp folders
    shutil.rmtree(search_tmp_dir)
else:
    print(f"Output data {output_path} already has proteome_match column. Skipping...")

# Save updated output data
use_cols = [col for col in USE_COLS if col in output_data.columns]
output_data[use_cols].to_csv(output_path, index=False)
