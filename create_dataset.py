import argparse
import os
import re
import shutil
from tqdm import tqdm
from pyteomics import mzml
from dataset_utils import *
from dataset_config import get_config, DatasetTag

MAX_SPECTRA_PER_FILE = 20000


def run_db_search(search_tool, dset_name, db_search_config, files_list, mzml_files_dir):
    """Run the appropriate DB search tool if results do not already exist."""
    if check_db_search_results_exist(search_tool, files_list, mzml_files_dir):
        print(f"Database search results already exist for {search_tool}. Skipping search.\n")
        return

    if search_tool == "msfragger":
        if db_search_config.n_db_splits > 1:
            print(f"Running MSFragger search with database split into {db_search_config.n_db_splits} parts...")
            run_msfragger_search_split(dset_name, db_search_config)
        else:
            run_msfragger_search(dset_name, db_search_config)
    elif search_tool == "msgf":
        run_msgf_search(dset_name, db_search_config)
    elif search_tool == "comet":
        run_comet_search(dset_name, db_search_config)
    else:
        raise ValueError(f"Unsupported search tool: {search_tool}")


def run_rescoring(search_tool, dset_name, rescoring_config, files_list, rescored_files_dir, rescore_file_prefix="rescore"):
    """Run Percolator rescoring for the specified search tool."""
    # TODO: check_rescoring_results_exist
    if f"{rescore_file_prefix}.percolator.psms.txt" in os.listdir(rescored_files_dir):
        print(f"Rescoring results already exist for {search_tool}. Skipping rescoring.\n")
        return

    tool_rescoring_features_dirs = {
        "msfragger": os.path.join(MZML_DATA_DIR, dset_name),
        "msgf": os.path.join(MZML_DATA_DIR, dset_name, "msgf_features"),
        "comet": os.path.join(MZML_DATA_DIR, dset_name, "comet_features"),
    }
    features_dir = tool_rescoring_features_dirs[search_tool]
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Rescoring features directory not found for {search_tool}: {features_dir}")

    # Create a single merged rescoring features file here (from files in features_dir)
    file_paths = [os.path.join(features_dir, f"{fname}.pin") for fname in files_list]
    # [! uncomment to use deep learning-based features in rescoring]
    if search_tool == "msfragger":
        get_msbooster_rescoring_features(dset_name, rescoring_config)
        file_paths = [os.path.join(features_dir, f"{fname}_rescore.pin") for fname in files_list]
    
    skiprows = [1] if search_tool == "msgf" else None
    droprows = [i - 1 for i in skiprows] if skiprows else []
    dfs = []
    for file_path in file_paths:
        print(file_path)
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
        # Split the first line into column names
        column_names = first_line.split("\t")
        try:
            df = pd.read_csv(file_path, sep="\t", usecols=column_names)
        except:
            df = pd.read_csv(file_path, sep="\t", usecols=column_names, skipfooter=1)
        df = df.drop(droprows, axis=0).reset_index(drop=True)
    #     df = pd.read_csv(file_path, sep="\t", usecols=column_names, skiprows=skiprows) 
        dfs.append(df)
        print()
    print("Found rescoring features files:", len(dfs))

    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df = df.fillna(0)
    print("Merged rescoring features dataframe:", df.shape)
    # Save merged PSMs features df to be used by Percolator
    df.to_csv(
        os.path.join(rescored_files_dir, f"{rescore_file_prefix}.pin"), 
        sep="\t", 
        index=False
    )
    # Run rescoring
    run_psm_rescoring(dset_name, rescoring_config, rescored_files_dir, rescore_file_prefix)


def get_mgf_files_spectra_idxs(files_list, mgf_files_dir, raw_files_dir, dset_id, download_config):
    """Ensure chunked MGF files exist and extract spectra indices."""
    if not check_chunked_mgf_files_exist(files_list, mgf_files_dir):
        # Convert raw files to MGF if needed
        if not check_full_mgf_files_exist(files_list, mgf_files_dir):
            # Download raw files if needed
            if not check_raw_files_exist(files_list, raw_files_dir, download_config.ext):
                download_files(download_config, files_list)
            convert_raw(dset_id, files_list, mgf_files_dir, target_ext=".mgf")

        # Filter and chunk MGF files
        spectra_idxs_0 = []
        for fname in tqdm(files_list):
            input_path = os.path.join(mgf_files_dir, fname + ".mgf")
            spectra = mgf.read(input_path)
            spectra_filtered = [spectrum for spectrum in spectra if "charge" in spectrum["params"]]
            n_splits = len(spectra_filtered) // MAX_SPECTRA_PER_FILE + int(len(spectra_filtered) % MAX_SPECTRA_PER_FILE > 0)
            for k in range(n_splits):
                chunk = spectra_filtered[k * MAX_SPECTRA_PER_FILE:(k + 1) * MAX_SPECTRA_PER_FILE]
                output_fname = f"{fname}_{k}"
                output_path = os.path.join(mgf_files_dir, output_fname + ".mgf")
                if not os.path.exists(output_path):
                    mgf.write(chunk, output_path)
                idxs_0 = {idx: spectrum["params"]["title"] for idx, spectrum in enumerate(chunk)}
                idxs_0 = pd.DataFrame.from_dict(idxs_0, orient="index", columns=["title"]).reset_index()
                idxs_0 = idxs_0.rename(columns={"index": "idx_0"})
                idxs_0["filename"] = output_fname
                spectra_idxs_0.append(idxs_0)
        return pd.concat(spectra_idxs_0, axis=0).reset_index(drop=True)

    # If chunked MGF files already exist, just read them & extract spectra idxs
    spectra_idxs_0 = []
    for fname in tqdm(files_list):
        print(fname)
        chunk_files = [f for f in os.listdir(mgf_files_dir) if re.fullmatch(fr"{fname}_[\d]+.mgf", f)]
        # chunk_files = [f for f in os.listdir(mgf_files_dir) if f.startswith(fname + "_") and f.endswith(".mgf")]
        for chunk_file in chunk_files:
            print("Reading chunk file:", chunk_file)
            chunk_path = os.path.join(mgf_files_dir, chunk_file)
            spectra = mgf.read(chunk_path)
            idxs_0 = {idx: spectrum["params"]["title"] for idx, spectrum in enumerate(spectra)}
            idxs_0 = pd.DataFrame.from_dict(idxs_0, orient="index", columns=["title"]).reset_index()
            idxs_0 = idxs_0.rename(columns={"index": "idx_0"})
            idxs_0["filename"] = chunk_file[:-len(".mgf")]
            spectra_idxs_0.append(idxs_0)
    return pd.concat(spectra_idxs_0, axis=0).reset_index(drop=True)


def collect_labels(
    search_tool, rescored_files_dir, rescore_file_prefix, spectra_idxs_0, labels_path, config,
):
    """Collect PSM labels from Percolator results and save them."""
    results_path = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.psms.txt")
    # results_df = pd.read_csv(results_path, sep="\t")
    results_df = pd.read_csv(
        results_path, sep="\t", 
        usecols=['PSMId', 'score', 'q-value', 'posterior_error_prob', 'peptide', 'proteinIds']
    ) 

    # filter by q-value
    q_val_threshold = config.rescoring.q_val_threshold
    print("Filtering PSMs by q-value threshold:", q_val_threshold)
    print("Before filtering:", results_df.shape)
    results_df = results_df[results_df["q-value"] < q_val_threshold]
    print("After filtering:", results_df.shape)

    # filter by PSM beloging to the corresponding pool
    if config.db_search.pool_proteomes_dir is not None: # TODO: mb use a different check for peptides that need to be filtered?
        # For synthetic peptides, generate separate databased for each file
        pool_proteomes_dir = os.path.join(PROTEOMES_DIR, config.db_search.pool_proteomes_dir)
        PT_pools_name = f"{config.name}.csv"
        PT_pools_path = os.path.join(PROTEOMES_DIR, PT_pools_name)
        PT_pools_df = pd.read_csv(PT_pools_path)
        sample_pools = PT_pools_df.set_index("sample")["pool"].to_dict()

        results_df["filename"] = results_df["PSMId"].apply(lambda x: get_filename(x, search_tool=search_tool))
        results_df["pools"] = results_df["proteinIds"].str.strip(";").str.split(";")
        results_df["psm_in_pool"] = results_df.apply(
            lambda row: check_psm_in_sample_pool(sample_pools[row.filename], row.pools), 
            axis=1
        )
        results_df = results_df[results_df["psm_in_pool"]].reset_index(drop=True)
        print("After filtering by sample pool:", results_df.shape)
    elif config.rescoring.filter_pools:
        results_df["filename"] = results_df["PSMId"].apply(lambda x: get_filename(x, search_tool=search_tool))
        results_df["pools"] = results_df["proteinIds"].str.strip(";").str.split(";")
        results_df["psm_in_pool"] = results_df.apply(
            lambda row: check_psm_in_sample_pool(config.rescoring.filter_pools, row.pools), 
            axis=1
        )
        results_df = results_df[results_df["psm_in_pool"]].reset_index(drop=True)
        print("After filtering by sample pool:", results_df.shape)

    results_df = results_df[["PSMId", "peptide", "q-value"]]
    if search_tool == "msgf" and DatasetTag.agilent in config.tags:
        results_df = map_psm_id_index_to_scan_id(results_df)
    elif search_tool == "msgf" and DatasetTag.timstof in config.tags:
        results_df = fix_msgf_timstof_psm_id(results_df)
    results_df["title"] = results_df["PSMId"].apply(lambda x: get_spectrum_title(x, search_tool=search_tool))
    results_df = pd.merge(results_df, spectra_idxs_0, on="title")

    results_df["spectrum_id"] = results_df["filename"] + ":" + results_df["idx_0"].astype(str)
    results_df["peptide"] = results_df["peptide"].apply(format_peptide_notation)
    sequences_true = results_df[["peptide", "spectrum_id"]].rename(columns={"peptide": "seq"})
    sequences_true.to_csv(labels_path, index=False)


# Main script logic
if __name__ == "__main__":
    # Paths parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to the dataset config file")
    parser.add_argument("--search_tool", type=str, default="msfragger", help="database search tool to use (e.g., msfragger, msgf, comet)")
    args = parser.parse_args()

    # Config setup
    config = get_config(args.config_path)
    dset_id = config.download.dset_id
    dset_name = config.name
    search_tool = args.search_tool
    rescore_file_prefix = f"{search_tool}_rescore" if search_tool != "msfragger" else "rescore"

    # Main output dirs setup
    for data_dir in [RAW_DATA_DIR, MZML_DATA_DIR, RESCORED_DATA_DIR, DATASET_STORAGE_DIR]:
        os.makedirs(data_dir, exist_ok=True)

    # Create dirs for intermediate & final dataset files
    raw_files_dir = os.path.join(RAW_DATA_DIR, dset_id)
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    rescored_files_dir = os.path.join(RESCORED_DATA_DIR, dset_name, rescore_file_prefix)
    mgf_files_dir = os.path.join(DATASET_STORAGE_DIR, dset_name, "mgf")
    for data_dir in [raw_files_dir, mzml_files_dir, rescored_files_dir, mgf_files_dir]:
        os.makedirs(data_dir, exist_ok=True)

    print("Creating dataset at:", mgf_files_dir)
    files_list = get_files_list(dset_name, config)
    print("Processing files:\n", files_list)

    # Prepare mzML files
    # TODO: make optional: we don't need mzml files if DB search outputs for search_tool already exist
    # if config.db_search.ext == ".mzml":
    prepare_mzml_files(dset_id, files_list, raw_files_dir, mzml_files_dir, config.download)

    # Run DB search
    run_db_search(search_tool, dset_name, config.db_search, files_list, mzml_files_dir)

    # Run rescoring
    run_rescoring(search_tool, dset_name, config.rescoring, files_list, rescored_files_dir, rescore_file_prefix)

    # Prepare MGF files and extract spectra indices
    spectra_idxs_0 = get_mgf_files_spectra_idxs(files_list, mgf_files_dir, raw_files_dir, dset_id, config.download)

    # Collect labels
    # labels_fname = "labels.csv" if search_tool == "msfragger" else f"{search_tool}_labels.csv"
    labels_fname = f"{search_tool}_labels.csv"
    labels_path = os.path.join(DATASET_STORAGE_DIR, dset_name, labels_fname)
    collect_labels(search_tool, rescored_files_dir, rescore_file_prefix, spectra_idxs_0, labels_path, config)

    # Add dataset tags
    collect_dataset_tags(config)
