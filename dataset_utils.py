import os
import re
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm
from pyteomics import fasta, mgf

from dataset_config import Config, DataDownloadConfig

from dotenv import load_dotenv
load_dotenv()

DATA_DIR =  os.environ['DATA_DIR']
WORK_DIR = os.environ['WORK_DIR'] 
ROOT = os.environ['ROOT']
FRAGPIPE_DIR = os.environ['FRAGPIPE_DIR']

# Path to msconvert apptainer container
MSCONVERT_PATH = os.path.join(WORK_DIR, "benchmarking", "pwiz-skyline-i-agree-to-the-vendor-licenses_latest.sif")
# Path to MSFragger DB split script (for running DB search with large DB)
SPLIT_SCRIPT_PATH = os.path.join(FRAGPIPE_DIR, "FragPipe/21.1-Java-11/tools/msfragger_pep_split.py")
# Path to MSFragger executable file (jar)
MSFRAGGER_PATH = os.path.join(DATA_DIR, "easybuild", "build", "MSFragger-4.0", "MSFragger-4.0.jar")
MSFRAGGER_BASE_PARAMS = os.path.join(WORK_DIR, "benchmarking", "configs", "default_closed.params")
# Path to MSBooster executable file (jar)
MSBOOSTER_PATH = os.path.join(FRAGPIPE_DIR, "MSBooster", "1.2.31-Java-11", "MSBooster-1.2.31.jar")
DIANN_PATH = os.path.join(FRAGPIPE_DIR, "FragPipe/21.1-Java-11/tools/diann/1.8.2_beta_8/linux/diann-1.8.1.8")
KOINA_URL = "https://koina.wilhelmlab.org:443/v2/models/"
MSBOOSTER_BASE_PARAMS = os.path.join(WORK_DIR, "benchmarking", "params_rescore", "msbooster_base.params")

# Add MSGF+ path
MSGF_PATH = os.path.join(DATA_DIR, "tools", "MSGFPlus.jar")

# Add Comet path
COMET_PATH = os.path.join(DATA_DIR, "tools", "comet.linux.exe")
COMET_BASE_PARAMS = os.path.join(WORK_DIR, "benchmarking", "configs", "comet.params")

# Spectrum params order for saving labeled mgf files
MGF_KEY_ORDER = ["title", "pepmass", "rtinseconds", "charge", "scans", "seq"]
# Path to the file with datasets properties (tags)
DATASET_TAGS_PATH = os.path.join(ROOT, "denovo_benchmarks", "dataset_tags.tsv")


PROTEOMES_DIR = os.path.join(ROOT, "proteomes")

RAW_DATA_DIR = os.path.join(ROOT, "raw")
MZML_DATA_DIR = os.path.join(ROOT, "mzml")
RESCORED_DATA_DIR = os.path.join(ROOT, "rescored")
DATASET_STORAGE_DIR = os.path.join(DATA_DIR, "benchmarking", "datasets")


def get_files_list(dset_name: str, config: Config):
    """
    Select raw spectra files for the dataset based on 
    selection rules defined in the download_config such as:
    - file extension;
    - number of files;
    - inclusion/exclusion keywords in the filename. 

    Args:
        dset_name (str): Dataset name.
        config (Config): Config with dataset 
            selection criteria, including spectral dataset ID,
            file extension for download and for DB search, 
            the inclusion/exclusion keywords, 
            or file download links, and the maximum number of files.

    Returns:
        dict: A dictionary where keys are file names (without extensions)
            and values are their corresponding paths 
            to the raw or mzML spectra file within the dataset folder.
    """

    def check_file(file_path, ext, links=None):
        """Check if file matches criteria."""
        if links is not None:
            # If downloading links provided, check if filename is in the list
            if file_path not in links:
                return False
        if not file_path.lower().endswith(ext.lower()):
            return False
        if any(keyword not in file_path for keyword in config.download.keywords):
            return False
        if any(keyword in file_path for keyword in config.download.exclude_keywords):
            return False
        return True

    dset_id = config.download.dset_id
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    raw_files_dir = os.path.join(RAW_DATA_DIR, dset_id)
    download_ext = config.download.ext
    db_search_ext = config.db_search.ext

    # Step 1: Check for existing mzML files
    if os.path.exists(mzml_files_dir):
        links = config.download.links
        links = [os.path.basename(link) for link in links] if links else None
        links = [link[:-len(download_ext)] + db_search_ext for link in links] if links else None
        mzml_files = [
            os.path.join(mzml_files_dir, f)
            for f in os.listdir(mzml_files_dir)
            if check_file(f, ext=db_search_ext)
        ]
        if len(mzml_files) >= config.download.n_files:
            files_list = {
                os.path.basename(file_path)[:-len(db_search_ext)]: file_path  # Remove ".mzML"
                for file_path in mzml_files[:config.download.n_files]
            }
            return files_list

    # Step 2: Check for existing raw files
    if os.path.exists(raw_files_dir):
        links = config.download.links
        links = [os.path.basename(link) for link in links] if links else None
        raw_files = [
            os.path.join(raw_files_dir, f)
            for f in os.listdir(raw_files_dir)
            if check_file(f, ext=download_ext, links=links)
        ]
        if len(raw_files) >= config.download.n_files:
            files_list = {
                os.path.basename(file_path)[:-len(download_ext)]: file_path
                for file_path in raw_files[:config.download.n_files]
            }
            return files_list

    # Step 3: Connect to Pride/Massive repository
    if "PXD" in dset_id or "MSV" in dset_id:
        import ppx 

        proj = ppx.find_project(
            dset_id, 
            local=raw_files_dir,
        )
        files_list = [
            file_path
            for file_path 
            in proj.remote_files() 
            if check_file(file_path, ext=download_ext)
        ]
    else:
        # If load via wget, file_path is just fname.ext
        files_list = [
            os.path.basename(file_link) 
            for file_link 
            in config.download.links
            # if check_file(file_link,  ext=download_ext)
        ]
    files_list = files_list[:config.download.n_files]
    files_list = {
        os.path.basename(file_path)[:-len(download_ext)]: file_path
        for file_path
        in files_list
    }
    return files_list


### Checks for existing files before downloading/unpacking/processing

def check_mzml_files_exist(files_list, mzml_files_dir):
    """Check if all mzML files for the given files_list exist in mzml_files_dir."""
    mzml_files = [fname.lower() + ".mzml" for fname in files_list]
    return set(mzml_files).issubset(set(fname.lower() for fname in os.listdir(mzml_files_dir)))

def check_raw_files_exist(files_list, raw_files_dir, ext=".raw"):
    """Check if all raw files for the given files_list exist in raw_files_dir."""
    raw_files = [fname.lower() + ext for fname in files_list]
    return set(raw_files).issubset(set(fname.lower() for fname in os.listdir(raw_files_dir)))

def check_unpacked_files_exist(files_list, unpack_dir, base_file_ext):
    # Check if unpacked files exist
    unpacked_files = [fname.lower() + base_file_ext for fname in files_list]
    return set(unpacked_files).issubset(set(fname.lower() for fname in os.listdir(unpack_dir)))

def check_db_search_results_exist(search_tool, files_list, mzml_files_dir):
    """Check if database search results already exist."""
    if search_tool == "msfragger":
        # expected_files = [os.path.join(mzml_files_dir, f"{fname}.pin") for fname in files_list]
        expected_files = [os.path.join(mzml_files_dir, f"{fname}_rescore.pin") for fname in files_list]
    elif search_tool == "msgf":
        expected_files = [os.path.join(mzml_files_dir, "msgf_features", f"{fname}.pin") for fname in files_list]
    elif search_tool == "comet":
        expected_files = [os.path.join(mzml_files_dir, "comet_features", f"{fname}.pin") for fname in files_list]
    else:
        raise ValueError(f"Unsupported search tool: {search_tool}")
    
    return all(os.path.exists(file) for file in expected_files)

def check_chunked_mgf_files_exist(files_list, mgf_files_dir):
    """Check if chunked MGF files exist."""
    chunked_mgf_files = [fname.lower() + "_0.mgf" for fname in files_list]
    return set(chunked_mgf_files).issubset(set(fname.lower() for fname in os.listdir(mgf_files_dir)))

def check_full_mgf_files_exist(files_list, mgf_files_dir):
    """Check if full MGF files exist."""
    full_mgf_files = [fname.lower() + ".mgf" for fname in files_list]
    return set(full_mgf_files).issubset(set(fname.lower() for fname in os.listdir(mgf_files_dir)))


### Methods to prepare spectra files for database search

def download_files(download_config, files_list):
    """
    Download files from the `files_list` to a local folder. 

    Args:
        download_config (DataDownloadConfig): config with dataset details.
        files_list (dict): Dictionary where keys are filenames (without extensions) 
            and values are full file paths.
    """
    dset_id = download_config.dset_id
    dset_dir = os.path.join(RAW_DATA_DIR, dset_id)
    print(f"Loading dataset {dset_id} to the folder {dset_dir}")

    if "PXD" in dset_id or "MSV" in dset_id:
        import ppx 
        
        proj = ppx.find_project(
            dset_id, 
            local=dset_dir,
        )
        print("Local files:", proj.local_files())
    
        # select files to download
        fnames = list(files_list.values())
        if download_config.ext == ".wiff":
            fnames += [fname + ".scan" for fname in fnames]
        proj.download(fnames)
        print("Loaded files:\n", proj.local_files()) # TODO: remove

    else:
        # if load via wget, need to take download_links from downloag_config for each file
        file_links = {
            os.path.basename(file_link)[:-len(download_config.ext)]: file_link
            for file_link 
            in download_config.links
        }
        print("Local files:", os.listdir(dset_dir))
        for fname, file_path in files_list.items():
            if not os.path.exists(os.path.join(dset_dir, file_path)):
                cmd = [
                    "wget",
                    "-P",
                    dset_dir,
                    file_links[fname]
                ]
                subprocess.run(" ".join(cmd), shell=True, check=True)
        print("Loaded files:", os.listdir(dset_dir))

def convert_raw(dset_id, files_list, target_dir, target_ext=".mzml"):
    os.makedirs(target_dir, exist_ok=True)    

    dset_dir = os.path.join(RAW_DATA_DIR, dset_id)
    raw_file_pathes = {
        fname: os.path.join(dset_dir, file_path)
        for fname, file_path
        in files_list.items()
    }
    print("Files:\n", list(raw_file_pathes.values()))
    print(f"Converting to {target_ext}. Storing to {target_dir}")

    ext_flag = "--mgf" if target_ext == ".mgf" else "--mzML"
    filter_ms_level = 2 if target_ext == ".mgf" else 1

    for fname, file_path in tqdm(raw_file_pathes.items()):
        # Skip the conversion:
        #  - for .mzML: if the target file filename.mzML already exists in files_dir, 
        if target_ext == ".mzml":
            out_fname = fname + ".mzML"
            target_file = os.path.join(target_dir, out_fname)
            if os.path.exists(target_file):
                print(f"Skipping {fname}.mzML, already exists in {target_dir}")
                continue

        #  - for .mgf: if any file matching the pattern filename_{i}.mgf 
        #    exists in files_dir
        elif target_ext == ".mgf":
            out_fname = fname + ".mgf"
            target_file = os.path.join(target_dir, out_fname)
            if os.path.exists(target_file):
                print(f"Skipping {fname}.mgf, already exists in {target_dir}")
                continue

        out_fname = fname + target_ext
        cmd = [
            "apptainer",
            "exec",
            "--cleanenv",
            MSCONVERT_PATH,
            "wine msconvert",
            ext_flag,
            "-z",
            "-o",
            target_dir,
            "--outfile",
            out_fname,
            "--filter",
            '"peakPicking vendor"',
            "--filter",
            '"precursorRefine"',
            "--filter",
            f'"msLevel {filter_ms_level}-"',
            "--filter",
            '"titleMaker <RunId>.<ScanNumber>.<ScanNumber>.<ChargeState>"'
            # '"titleMaker <RunId>.<Index>.<ChargeState>.<MsLevel>"'
        ]
        cmd += [file_path]
        subprocess.run(" ".join(cmd), shell=True, check=True)
    print(os.listdir(target_dir))

def prepare_mzml_files(dset_id, files_list, raw_files_dir, mzml_files_dir, download_config):
    """Ensure mzML files exist by downloading and converting raw files if needed."""
    
    if not check_mzml_files_exist(files_list, mzml_files_dir):
        # Download raw files if needed
        if not check_raw_files_exist(files_list, raw_files_dir, download_config.ext):
            download_files(download_config, files_list)

        # Unpack raw files if needed
        if download_config.ext.endswith(".zip"):
            base_file_ext = download_config.ext[:-len(".zip")]
            # If files can be directly processed by MSFragger (.d, .mzml), unpack to mzml_files_dir
            unpack_dir = mzml_files_dir if base_file_ext == ".d" else raw_files_dir
            if not check_unpacked_files_exist(files_list, unpack_dir, base_file_ext):
                for fname in files_list.values():
                    file_path = os.path.join(raw_files_dir, fname)
                    shutil.unpack_archive(filename=file_path, extract_dir=unpack_dir)

        # Convert raw files to mzML
        if download_config.ext in [".raw", ".wiff"]:
            convert_raw(dset_id, files_list, mzml_files_dir, target_ext=".mzml")
    
    mzml_files = [fname for fname in os.listdir(mzml_files_dir) if fname.lower().endswith(".mzml")]
    print("mzML files are ready in", mzml_files_dir, ":\n", mzml_files)


### Methods to prepare reference databases for real datasets (add decoys and contaminants)

def generate_decoys_fasta(dset_name, db_file, contam_only=False):
    """
    Add decoys and common contaminants to the reference database
    with FragPipe Philosopher. 
    Only run if there is no existing database with decoys.
    """

    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    db_path = os.path.join(PROTEOMES_DIR, db_file)

    def check_fasta_filename(fname):
        db_fname = db_file.split("/")[-1]
        if contam_only:
            return (f"contam-{db_fname}" in fname) and ("decoys-" not in fname)
        return f"decoys-contam-{db_fname}" in fname

    # Check if database with decoys already exists
    existing_db_w_decoys = [
        fname for fname in os.listdir(mzml_files_dir)
        if fname.endswith(".fas") and check_fasta_filename(fname)
    ]
    print("Existing databases with decoys:\n", existing_db_w_decoys)

    if existing_db_w_decoys:
        db_w_decoys_path = existing_db_w_decoys[0]
    
    else:
        cmd = [
            "cd",
            mzml_files_dir,
            "&&",
            "philosopher workspace --init",
            "&&",
            "philosopher database",
            "--custom",
            db_path,
            "--contam",
        ]
        if contam_only:
            cmd += ["--nodecoys"]
        subprocess.run(" ".join(cmd), shell=True, check=True)
        db_w_decoys_path = [
            fname for fname in os.listdir(mzml_files_dir) 
            if fname.endswith(".fas") and check_fasta_filename(fname)
        ][0]
    
    db_w_decoys_path = os.path.join(mzml_files_dir, db_w_decoys_path)
    return db_w_decoys_path

### Methods to prepare reference databases for synthetic peptides

def get_extended_db_path(db_path):
    db_path = db_path.split("/")
    db_path[-1] = "extended-" + db_path[-1]
    return "/".join(db_path)

def append_pool_peptides(add_pool_path, db_path):
    """Add peptides from another pool to the reference database to enrich search space for FDR calculation."""
    with open(add_pool_path, 'r') as f_in, open(db_path, 'a') as f_out:
        skip = True
        for line in f_in:    
            if line.startswith('>'): # a header
                skip = line.startswith('>QC')
                if not skip: # write the header
                    f_out.write(line)   
            elif not skip: # write the following sequence
                f_out.write(line)
    print(f"Pool peptides from '{add_pool_path}' appended to '{db_path}'.")

def prepare_synthetic_fasta(dset_name, files_list, pool_proteomes_dir, PT_pools_df):
    files_db_w_decoys = {}
    for fname in files_list:
        print("File:", fname)
        sample_idx = PT_pools_df[PT_pools_df["sample"] == fname].index[0]
        pool_fasta = PT_pools_df.loc[sample_idx, "fasta"]
        print(sample_idx, pool_fasta)

        mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
        db_path = os.path.join(mzml_files_dir, pool_fasta)
        if os.path.exists(db_path):
            print("Original DB already exists:", db_path)
        else:
            # if DB doesn't exist yet, copy pool database to the mzml_files_dir
            source_db_path = os.path.join(pool_proteomes_dir, pool_fasta)
            cmd = f"cp {source_db_path} {db_path}"
            subprocess.run(cmd, shell=True, check=True)
            print("Original DB", db_path)
        
        # Generate extended database and extended decoy database
        db_extended_path = get_extended_db_path(db_path)
        if os.path.exists(db_extended_path):
            print("Extended DB already exists:", db_extended_path)
        else:
            # if extentded DB doesn't exist yet, create it (add 10 pools to DB)
            cmd = f"cat {db_path} >> {db_extended_path}"
            print(f"Extended DB: {db_extended_path}.")
            subprocess.run(cmd, shell=True, check=True)
            
            pool_fastas_df = PT_pools_df["fasta"].drop_duplicates().reset_index(drop=True).to_frame()
            idx = pool_fastas_df[pool_fastas_df["fasta"] == pool_fasta].index[0]
            add_pool_idxs = slice((idx + 1) % len(pool_fastas_df), (idx + 11) % len(pool_fastas_df))
            add_pool_fastas = pool_fastas_df["fasta"].iloc[add_pool_idxs].values.tolist()
            for add_pool_fasta in add_pool_fastas:
                add_pool_path = os.path.join(pool_proteomes_dir, add_pool_fasta)
                append_pool_peptides(add_pool_path, db_extended_path)
            
        # Add decoys to extended database
        db_w_decoys_extended_path = generate_decoys_fasta(
            dset_name=dset_name, db_file=db_extended_path, contam_only=False
        )
        print("Extended DB with decoys and contaminants:", db_w_decoys_extended_path)   
        files_db_w_decoys[fname] = db_w_decoys_extended_path
    return files_db_w_decoys


### Methods to run database search with MSFragger (with or without splitting the database)

def generate_msfragger_params(db_w_decoys_path, config, template_path, output_path):
    """
    Generates a MSFragger parameter file from a given configuration and template.

    Args:
    config (dict): The configuration dictionary for the dataset.
    template_path (str): Path to the default template .params file.
    output_path (str): Path to save the modified .params file.
    """
    
    def replace_param_value(line, new_value):
        if "#" in line:
            param, desc = line.split("#")
        else:
            param = line
            desc = ""
        
        param_key, param_value = param.split(" = ", 1)
        
        if "variable_mod" in param_key:
            new_value = new_value.replace("_", " ")
        
        param = f"{param_key} = {new_value}"
        return f"{param}        # {desc}"
    
    # Step 1: Load the default template params file
    with open(template_path, 'r') as template_file:
        template_data = template_file.readlines()
    
    # Step 2: Modify template based on the provided config
    search_params = {key.lstrip("-"): value for key,  value in config.search_params.items()}
    search_params["database_name"] = db_w_decoys_path
    
    modified_params = []
    for line in template_data:
        line = line.strip()
        
        for key, value in search_params.items():
            # Search for the parameter in the template file and replace if exists
            if key in line and not line.startswith("#"):                
                line = replace_param_value(line, value)
                break  # No need to check other keys once matched
        
        modified_params.append(line)
    modified_params.append("\n")

    # Step 3: Add missing parameters from config (not in the template)
    existing_params = {line.split('=')[0].strip() for line in modified_params if '=' in line}
    for key, value in search_params.items():
        if key not in existing_params:
            if "variable_mod" in key:
                value = value.replace("_", " ")
            modified_params.append(f"{key} = {value}")

    # Step 4: Save the modified params file
    with open(output_path, 'w') as output_file:
        output_file.write("\n".join(modified_params))
    print(f"Generated MSFragger params file at: {output_path}")

def run_msfragger_search(dset_name, db_search_config):
    """Run MSFragger database search."""
    # Search for mzml files in the directory
    search_ext = db_search_config.ext
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mzml_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == search_ext]
    mzml_files = {os.path.splitext(f)[0]: os.path.join(mzml_files_dir, f) for f in mzml_files}

    # Generate decoys database
    if db_search_config.pool_proteomes_dir is not None: # TODO: mb use a different check for synthetic peptides?
        # For synthetic peptides, generate separate databased for each file
        pool_proteomes_dir = os.path.join(PROTEOMES_DIR, db_search_config.pool_proteomes_dir)
        PT_pools_name = f"{dset_name}.csv"
        PT_pools_path = os.path.join(PROTEOMES_DIR, PT_pools_name)
        PT_pools_df = pd.read_csv(PT_pools_path)
        files_db_w_decoys = prepare_synthetic_fasta(dset_name, mzml_files, pool_proteomes_dir, PT_pools_df)
    else:
        db_w_decoys_path = generate_decoys_fasta(dset_name, db_search_config.database_path)
        files_db_w_decoys = {fname: db_w_decoys_path for fname in mzml_files}

    # Combine mzml_files with the same database
    db_w_decoys_files = {}
    for fname, db_w_decoys_path in files_db_w_decoys.items():
        if db_w_decoys_path not in db_w_decoys_files:
            db_w_decoys_files[db_w_decoys_path] = []
        db_w_decoys_files[db_w_decoys_path].append(mzml_files[fname])
    
    # Iterate over each database and run the msfragger command
    for db_w_decoys_path, mzml_files in db_w_decoys_files.items():
        print("\nDB with decoys:\n", db_w_decoys_path)
        print("Files:\n", mzml_files)
        options = [
            "--database_name",
            db_w_decoys_path,
            "--decoy_prefix",
            "rev_",
            "--output_format",
            "pepxml_pin", # .pin outputs for MSBooster
        ]
        if db_search_config.ext == ".d":
            options += [
                "--write_uncalibrated_mgf", "1",
                "--write_calibrated_mzml", "1",
            ]
        # Parse additional search params from config if provided
        for arg in [*db_search_config.search_params.items()]:
            options += list(map(str, arg))
    
        cmd = [
            "java",
            "-Xmx160G",
            "-jar",
            MSFRAGGER_PATH,
            *options,
            *mzml_files,
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)

        if search_ext == ".d":
            # Use uncalibrated mzml files to get "proxy" mzml files
            for file_path in mzml_files:
                fname = os.path.splitext(file_path)[0]
                src_fname = fname + "_uncalibrated.mzML"
                dst_fname = fname + ".mzML"
                shutil.move(src_fname, dst_fname)
                print(f"Created mzML: {dst_fname}")
    print("DB search results (.pepXML, .pin):\n", os.listdir(mzml_files_dir))

def run_msfragger_search_split(dset_name, db_search_config):
    # TODO: does not support search with multiple per-file databases (synthetic peptides)
    """Run MSFragger database search with split database."""
    # Generate decoys database
    db_w_decoys_path = generate_decoys_fasta(dset_name, db_search_config.database_path)

    # Search for mzml files in the directory
    search_ext = db_search_config.ext
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mzml_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == search_ext]
    mzml_files = [os.path.join(mzml_files_dir, f) for f in mzml_files]

    # Create MSFragger params file from config
    params_file = os.path.join(mzml_files_dir, "closed.params")
    generate_msfragger_params(
        db_w_decoys_path,
        db_search_config, 
        MSFRAGGER_BASE_PARAMS, 
        params_file
    )

    n_db_splits = db_search_config.n_db_splits
    # Iterate over each mzML file and run the msfragger command
    for mzml_file in mzml_files:
        output_file = os.path.basename(mzml_file).replace("mzML", "pepXML")
        if output_file in os.listdir(mzml_files_dir):
            print(f"{output_file} already exists.")
            continue

        print(f"PROCESSING {mzml_file}")
        cmd = [
            "python",
            SPLIT_SCRIPT_PATH,
            f"{n_db_splits}",
            '"java -jar -Dfile.encoding=UTF-8 -Xmx160G"',
            MSFRAGGER_PATH,
            params_file,
            mzml_file,
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)
        print(f"PROCESSED {mzml_file}\n\n")

    # Print results
    print("DB search results (.pepXML, .pin):\n", os.listdir(mzml_files_dir))

### Methods to generate PSM rescoring features for MSFragger search results with MSBooster 

def get_msbooster_rescoring_features(dset_name, rescoring_config):
    """Create PSMs rescoring features with MSBooster."""
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    rescored_files_dir = os.path.join(RESCORED_DATA_DIR, dset_name)
 
    # select all the .mzml files from mzml_files_dir (# only select with fnames in files_list?)
    mzml_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == ".mzml"]
    mzml_files = [os.path.join(mzml_files_dir, f) for f in mzml_files]
    print(".mzML files available for rescoring:\n", mzml_files)

    # select .pin files with fnames in files_list
    pin_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == ".pin"]
    pin_files = [os.path.join(mzml_files_dir, f) for f in pin_files]
    print(".pin files available for rescoring:\n", pin_files)

    file_prefix = "rescore" # TODO: move outside?

    options = [
        "--DiaNN",
        DIANN_PATH,
        "--KoinaURL",
        KOINA_URL,
        "--editedPin",
        file_prefix,
        "--paramsList",
        MSBOOSTER_BASE_PARAMS,
        "--mzmlDirectory",
        *mzml_files,
        "--pinPepXMLDirectory",
        *pin_files,
        "--outputDirectory",
        rescored_files_dir,
    ]

    for arg in [*rescoring_config.feat_pred_params.items()]:
        options += list(map(str, arg))
    
    cmd = [
        "java",
        "-Xmx64G",
        "-jar",
        MSBOOSTER_PATH,
        *options,
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    print("Created PSMs features (_rescore.pin):\n", os.listdir(mzml_files_dir))


### Methods to run database search with MSGF+

def prepare_msgf_fasta(dset_name, db_file=None, db_w_decoys_file=None):
    """Prepare target and decoy databases for MSGF+."""
    if db_w_decoys_file is None:
        print("Prepare database with reversed decoys.")
        db_w_decoys_file = generate_decoys_fasta(dset_name, db_file)
    print("Database with decoys:\n", db_w_decoys_file)
    
    print("Split into separate target & decoys databases")
    target_db_file = db_w_decoys_file.replace("-decoys", "").split(".")[0] + ".fasta"
    decoys_db_file = db_w_decoys_file.replace("-contam", "").split(".")[0] + ".fasta"
    print("Target:", target_db_file)
    print("Decoys:", decoys_db_file)
    
    cmd = [
        "awk",
        "'/^>/ {is_decoy = ($0 ~ /^>rev_/)} {if (is_decoy) print > \"" + decoys_db_file + "\"; else print > \"" + target_db_file + "\"}'",
        db_w_decoys_file
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return target_db_file, decoys_db_file

def prepare_msgf_modifications_config(dset_name, db_search_config, mods_file="MSGF_Mods.txt"):
    """Write PTM modifications config file for MSGF+ search."""

    ptm_psims_names = {
        -17.0265: "Ammonia-loss", # or Gln->pyro-Glu
        0.984: "Deamidated",
        4.0251: "Label:2H(4)", # SILAC
        6.0201: "Label:13C(6)", # SILAC
        8.0142: "Label:13C(6)15N(2)", # SILAC
        10.0083: "Label:13C(6)15N(4)", # SILAC
        12.0: "Thiazolidine", # not PSI-MS, Formaldehyde adduct # UNIMOD:1009
        14.0157: "Methyl",
        15.9949: "Oxidation",
        21.9819: "Cation:Na", # Sodium adduct
        26.0156: "Delta:H(2)C(2)", # Acetaldehyde +26
        27.9949: "Formyl",
        28.0313: "Dimethyl",
        42.0106: "Acetyl",
        42.0470: "Trimethyl",
        44.9851: "Nitro", # Oxidation to nitro
        52.9115: "Cation:Fe[III]", # not PSI-MS, Replacement of 3 protons by iron
        53.9193: "Cation:Fe[II]", # not PSI-MS, Replacement of 2 protons by iron
        55.9197: "Cation:Ni[II]", # not PSI-MS, Replacement of 2 protons by nickel
        56.0262: "Propionyl", # Propionate labeling reagent light form (N-term & K)
        56.0626: "Diethyl",
        57.0215: "Carbamidomethyl",
        68.0262: "Crotonyl", # not PSI-MS, Crotonylation
        70.0419: "Crotonaldehyde", # Butyryl
        79.9663: "Phospho",
        86.0004: "Malonyl", # Malonylation
        86.0368: "Hydroxyisobutyryl", # not PSI-MS, 2-hydroxyisobutyrylation
        100.016: "Succinyl",
        114.0317: "Glutaryl", # not PSI-MS, Glutarylation
        114.0429: "GG", # Ubiquitinylation residue
        226.0776: "Biotin",
        229.1629: "TMT6plex", # not PSI-MS, Sixplex Tandem Mass Tag
    }
    term_residues_notation = {
        "n": "N-term",
        "c": "C-term",
    }

    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mods_file_path = os.path.join(mzml_files_dir, mods_file)

    search_params = db_search_config.search_params

    fixed_ptm_strs = {"C": "C2H3N1O1,C,fix,any,Carbamidomethyl"}
    # Parse fixed modifications
    for key, mass in search_params.items():
        if key.startswith("--add_"):
            residue = key.split("_")[1]
            mass = round(float(mass), 4)
            if mass == 0.0:
                del fixed_ptm_strs[residue]
            else:
                fixed_ptm_strs[residue] = f"{mass},{residue},fix,any,{ptm_psims_names.get(mass, 'Unknown')}"

    with open(mods_file_path, "w") as f:
        for fixed_ptm_str in fixed_ptm_strs.values():
            f.write(fixed_ptm_str + "\n")

    config_ptm_strs = []
    for k in search_params:
        if "variable_mod_" in k:
            config_ptm_str = search_params[k]
            config_ptm_strs.append(config_ptm_str)

    with open(mods_file_path, "a") as f:
        for config_ptm_str in config_ptm_strs:
            mass, residues, n = config_ptm_str.split("_")
            mass = round(float(mass), 4)
            if mass == 0:
                # No modification
                continue
            
            # Split residues if they include both N/C-term and amino acids
            term_residues = re.findall("[cn].", residues)
            aa_residues = re.sub("[cn].", "", residues)
            
            if term_residues:
                for term_residue in term_residues:
                    term, residue = list(term_residue) # len(term_residue) == 2
                    position = term_residues_notation[term]
                    residue = "*" if residue == "^" else residue
                    f.write(",".join([str(mass), residue, "opt", position, ptm_psims_names[mass]]) + "\n")
            
            if aa_residues:
                position = "any"
                f.write(",".join([str(mass), aa_residues, "opt", position, ptm_psims_names[mass]]) + "\n")
    return mods_file_path

def run_msgf_search(dset_name, db_search_config):
    """Run MSGF+ database search."""

    msgf_enzyme_id = {
        "nonspecific": 0, # unspecific cleavage
        "trypsin": 1, # Trypsin
        "chymotrypsin": 2, # Chymotrypsin
        "lysc": 3, # Lys-C
        "lysn": 4, # Lys-N
        "gluc": 5, # Glu-C, glutamyl endopeptidase
        "argc": 6, # Arg-C
        "aspn": 7, # Asp-N
    #     "": 8, # alphaLP
    #     "": 9, no cleavage
    }
    msgf_instrument_id = {
        "LCQ": 0, # Linear Quadrupole Ion Trap
        "LTQ": 0, # Linear Trap Quadrupole
        "Orbitrap": 1, # Orbitrap Elite, Fusion, Lumos
        "FTICR": 1, # Fourier Transform Ion Cyclotron Resonance
        "Lumos": 1, # a high-end version of the Orbitrap Fusion
        "TOF": 2, # Bruker, Sciex TripleTOF, Agilent TOF
        "QExactive": 3, # Hybrid Quadrupole-Orbitrap (QE, QE Plus / HF)
    }
    msgf_fragmentation_id = {
        "not_given": 0, # as written in the spectrum or CID if no info
        "CID": 1,
        "ETD": 2,
        "HCD": 3,
        "UVPD": 4,
    }

    # mass tol params are passed in the MSFragger expected format
    # default mass tolerance units are ppm (1)
    # for fragment mass, units are set to Da (0)

    def format_mass_tolerance(mass_lower, mass_upper, units="ppm"):
        mass_lower = abs(mass_lower)
        mass_tol_str = f"{mass_lower}{units},{mass_upper}{units}"
        return mass_tol_str

    # Fragment mass tolerance -- can not be specified. 
    # Although you can specify the precursor tolerance with -t 
    # you cannot specify the fragment ion tolerance. 
    # -- the tolerance is determined automatically by the training algorithm
    # Source: https://github.com/MSGFPlus/msgfplus/issues/17#issuecomment-348549563

    def get_precursor_mass_tol(db_search_config):
        mass_lower = db_search_config.search_params.get("--precursor_mass_lower", -20)
        mass_upper = db_search_config.search_params.get("--precursor_mass_upper", 20)
        return format_mass_tolerance(mass_lower, mass_upper, units="ppm")

    def get_instrument_id(db_search_config):
        # 0: Low-res LCQ/LTQ
        # 1: Orbitrap/FTICR/Lumos
        # 2: TOF
        # 3: Q-Exactive
        instrument_id = db_search_config.instrument
        return msgf_instrument_id.get(instrument_id, 0)

    def get_fragmentation_method(db_search_config):
        # 0: As written in the spectrum or CID if no info
        # 1: CID
        # 2: ETD
        # 3: HCD
        # 4: UVPD
        fragmentation_method = db_search_config.fragmentation
        return msgf_fragmentation_id.get(fragmentation_method, 0)

    def get_enzyme_id(db_search_config):
        return msgf_enzyme_id[
            db_search_config.search_params.get("--search_enzyme_name_1", "trypsin")
        ]

    search_ext = db_search_config.ext
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mzml_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == search_ext]
    mzml_files = {os.path.splitext(f)[0]: os.path.join(mzml_files_dir, f) for f in mzml_files}

    target_res_dir = os.path.join(mzml_files_dir, "target_res")
    decoys_res_dir = os.path.join(mzml_files_dir, "decoys_res")
    os.makedirs(target_res_dir, exist_ok=True)
    os.makedirs(decoys_res_dir, exist_ok=True)

    # Write MSGF+ modification config file
    mods_file_path = prepare_msgf_modifications_config(dset_name, db_search_config, mods_file="MSGF_Mods.txt")
    # Define MSGF+ search params
    options = [
        "-decoy", "rev_",
        "-t", format_mass_tolerance(
            db_search_config.search_params.get("--precursor_mass_lower", -20),
            db_search_config.search_params.get("--precursor_mass_upper", 20),
            "ppm"
        ),
        "-tda", 0,
        "-m", get_fragmentation_method(db_search_config), # Fragmentation Method; Default: 0
        "-inst", get_instrument_id(db_search_config), # Instrument ID; Default: 0
        "-e", msgf_enzyme_id[db_search_config.search_params.get("--search_enzyme_name_1", "trypsin")],
        "-ntt", db_search_config.search_params.get("--num_enzyme_termini", 2), # Number of Tolerable Termini [0/1/2]; Default: 2
        "-minLength", db_search_config.search_params.get("--digest_min_length", 7),
        "-maxLength", db_search_config.search_params.get("--digest_max_length", 40),
        "-addFeatures", 1, # add features for Percolator to the output
        "-numMods", db_search_config.search_params.get("--max_variable_mods_per_peptide", 3),
        "-mod", mods_file_path,
    ]
    if get_enzyme_id(db_search_config) != 0: # not nonspecific cleavage
        options += ["-maxMissedCleavages", db_search_config.search_params.get("--allowed_missed_cleavage_1", 2)]
    options = list(map(str, options))

    if db_search_config.pool_proteomes_dir is not None:
        # For synthetic peptides, generate separate databased for each file
        pool_proteomes_dir = os.path.join(PROTEOMES_DIR, db_search_config.pool_proteomes_dir)
        PT_pools_name = f"{dset_name}.csv"
        PT_pools_path = os.path.join(PROTEOMES_DIR, PT_pools_name)
        PT_pools_df = pd.read_csv(PT_pools_path)
        files_db_w_decoys = prepare_synthetic_fasta(dset_name, mzml_files, pool_proteomes_dir, PT_pools_df)
    else:
        # Generate a single database with decoys
        db_w_decoys_file = generate_decoys_fasta(dset_name, db_search_config.database_path)
        files_db_w_decoys = {fname: db_w_decoys_file for fname in mzml_files}

    # Combine mzML files with the same database
    db_w_decoys_files = {}
    for fname, db_w_decoys_path in files_db_w_decoys.items():
        if db_w_decoys_path not in db_w_decoys_files:
            db_w_decoys_files[db_w_decoys_path] = []
        db_w_decoys_files[db_w_decoys_path].append(mzml_files[fname])

    # Iterate over each database and run MSGF+ search
    for db_w_decoys_path, mzml_files in db_w_decoys_files.items():
        print("\nDB with decoys:\n", db_w_decoys_path)
        print("Files:\n", mzml_files)

        # Prepare target and decoy databases
        target_db_file, decoys_db_file = prepare_msgf_fasta(dset_name, db_w_decoys_file=db_w_decoys_path)

        print("Run MSGF+ for target database:")
        for fname in mzml_files:
            res_fname =  os.path.splitext(fname)[0] + ".mzid"
            cmd = [
                "java",
                "-Xmx160G",
                "-jar",
                MSGF_PATH,
                *options,
                "-d", target_db_file, # DatabaseFile
                "-s", fname,
                "&&",
                "mv", res_fname, target_res_dir,
            ]
            print(" ".join(cmd))
            subprocess.run(" ".join(cmd), shell=True, check=True)
            print("Target DB search results:\n", os.listdir(target_res_dir), "\n")

        print("Run MSGF+ for decoys database:")
        for fname in mzml_files:
            res_fname =  os.path.splitext(fname)[0] + ".mzid"
            cmd = [
                "java",
                "-Xmx160G",
                "-jar",
                MSGF_PATH,
                *options,
                "-d", decoys_db_file, # DatabaseFile
                "-s", fname,
                "&&",
                "mv", res_fname, decoys_res_dir,
            ]
            print(" ".join(cmd))
            subprocess.run(" ".join(cmd), shell=True, check=True)
            print("Decoys DB search results:\n", os.listdir(decoys_res_dir), "\n")

    # Collect target and decoy results into features file
    msgf2pin_enzymes = [
        "no_enzyme",
        "trypsin",
        "chymotrypsin",
        "lys-c",
        "lys-n",
        "glu-c",
        "arg-c",
        "asp-n",
    ]
    res_features_dir = os.path.join(mzml_files_dir, "msgf_features")
    os.makedirs(res_features_dir, exist_ok=True)

    for fname in os.listdir(target_res_dir):
        if fname.endswith(".mzid"):
            feats_fname = fname.replace(".mzid", ".pin")
            cmd = [
                "msgf2pin",
                "-o", os.path.join(res_features_dir, feats_fname),
                "-e", msgf2pin_enzymes[get_enzyme_id(db_search_config)],
                os.path.join(target_res_dir, fname),
                os.path.join(decoys_res_dir, fname),
            ]
            subprocess.run(" ".join(cmd), shell=True, check=True)

    # Cleanup temporary files created by MSGF+
    tmp_file_patterns = [".canno", ".cnlcp", ".csarr", ".cseq"]
    for pattern in tmp_file_patterns:
        for tmp_file in tqdm([f for f in os.listdir(mzml_files_dir) if f.endswith(pattern)]):
            os.remove(os.path.join(mzml_files_dir, tmp_file))
            print(f"Removed temporary file: {tmp_file}")


### Methods to run database search with Comet

# Dict to map MSFragger search params to analogous Comet search params
msfragger2comet_param_mapper = {
    "precursor_mass_upper": "peptide_mass_tolerance_upper",
    "precursor_mass_lower": "peptide_mass_tolerance_lower",
    "fragment_mass_tolerance": "fragment_bin_tol",
    # Enzyme name needs to be converted to id
    "search_enzyme_name_1": "search_enzyme_number",
    "search_enzyme_name_2": "search_enzyme2_number",
    "allowed_missed_cleavage_1": "allowed_missed_cleavage",
    "allowed_missed_cleavage_2": "allowed_missed_cleavage",
    "num_enzyme_termini": "num_enzyme_termini",
    # Digest length needs to be formatted separately into peptide_length_range
    "digest_min_length": "digest_min_length",
    "digest_max_length": "digest_max_length",
    # Up to 15 variable_mod entries are supported for a standard search
    "variable_mod_01": "variable_mod01",
    "variable_mod_02": "variable_mod02",
    "variable_mod_03": "variable_mod03",
    "variable_mod_04": "variable_mod04",
    "variable_mod_05": "variable_mod05",
    "variable_mod_06": "variable_mod06",
    "variable_mod_07": "variable_mod07",
    "variable_mod_08": "variable_mod08",
    "variable_mod_09": "variable_mod09",
    "variable_mod_10": "variable_mod10",
    "variable_mod_11": "variable_mod11",
    "variable_mod_12": "variable_mod12",
    "variable_mod_13": "variable_mod13",
    "variable_mod_14": "variable_mod14",
    "variable_mod_15": "variable_mod15",
    # fragment ions
    "use_A_ions": "use_A_ions",
    "use_B_ions": "use_B_ions", 
    "use_C_ions": "use_C_ions",
    "use_X_ions": "use_X_ions",
    "use_Y_ions": "use_Y_ions",
    "use_Z_ions": "use_Z_ions",
}
# Dict to map MSFragger search_enzyme_name_1/2 to Comet comet_enzyme_id
msfragger2comet_enzymes = {
    "nonspecific": 0,     # 0.  Cut_everywhere    0    -    -
    "trypsin": 1,         # 1.  Trypsin    1    KR    P
    "stricttrypsin": 2,   # 2.  Trypsin/P    1    KR    -
    "lysc": 3,            # 3.  Lys_C    1    K    P
    "lysn": 4,            # 4.  Lys_N    0    K    -
    "argc": 5,            # 5.  Arg_C    1    R    P
    "aspn": 6,            # 6.  Asp_N    0    DN    -
    # 7.  CNBr    1    M    -
    "gluc": 8,            # 8.  Glu_C    1    DE    P
    # 9.  PepsinA    1    FL    -
    "chymotrypsin": 10,   # 10. Chymotrypsin    1    FWYL    P
    # 11. No_cut    1    @    @
}
# Method to map PTMs from MSFragger format to Comet format
def msfragger2comet_ptm(ptm):
    """
    Comet modifications format:  
    <mass> 
    <residues> - If more than a single residue is modified, list them all as a string.
        Use ‘n’ for N-terminal modfication and ‘c’ for C-terminal modification.
    <binary> - If 0, the modification is a variable modification, else a binary modification 
        (binary = all residues are either modified or all residues are not modified).
    <max_mods> - maximum number of modified residues possible for this modification entry.
    <term_distance> - the distance the modification is applied to from the respective terminus.
        -2 = anywhere except C-term residue
        -1 = no distance contraint
        0 = only terminal residue
        N=1... = only applies to terminal residue and next N residues(?)
    <n/c-term> - which terminus the distance constraint is applied to:
        0 = protein N-terminus
        1 = protein C-terminus
        2 = peptide N-terminus
        3 = peptide C-terminus
    <required> - whether peptides must contain this modification. 
    If 1, only peptides that contain this modification will be analyzed.
    <neutral_loss>
    
    The default value is “0.0 X 0 3 -1 0 0 0.0”
    """

    ptm_binary = 0
    ptm_term_distance = -1
    ptm_term = 0
    ptm_required = 0
    ptm_neutral_loss = 0.0
    
    ptm_mass, ptm_residues, ptm_max_mods = ptm.split("_")

    if ptm_mass == "0":
        # No modification -- set to default params
        ptm_residues = "X"
        ptm_max_mods = 3

    else:
        ptm_residues = ptm_residues.replace("^", "")

    ptm = f"{ptm_mass} {ptm_residues} {ptm_binary} {ptm_max_mods} " \
        + f"{ptm_term_distance} {ptm_term} {ptm_required} {ptm_neutral_loss}"
    return ptm

def generate_comet_params(db_path, config, template_path, output_path):
    """
    Generates a Comet parameter file from a given dataset config and template.
    
    Args:
    db_path (str): Path to the target protein database 
        (with contaminants, without decoys!).
    config (dict): The dataset DB search config object.
    template_path (str): Path to the default comet.params file.
    output_path (str): Path to write the generated comet.params file.
    """

    # Read in default template
    with open(template_path, 'r') as f:
        lines = f.readlines()

    # Extract search params from db_search_config
    search_params = {k.lstrip('-'): v for k, v in config.search_params.items()}

    # Map fragment_ion_series param to Comet format
    if "fragment_ion_series" in search_params:
        fragment_ions = search_params["fragment_ion_series"].upper().split(",")
        # Reset default ion settings
        search_params[f"use_B_ions"] = 0
        search_params[f"use_Y_ions"] = 0
        # Update with specified ions
        for fragment_ion in fragment_ions:
            search_params[f"use_{fragment_ion}_ions"] = 1
    
    # Map search params from MSFragger to Comet format
    search_params = {
        msfragger2comet_param_mapper.get(k, k): v 
        for k, v in search_params.items()
        if k in msfragger2comet_param_mapper or "add_" in k # static mods
    }
    # Map search_enzyme value (name to number)
    if "search_enzyme_number" in search_params:
        search_params["search_enzyme_number"] = msfragger2comet_enzymes[search_params["search_enzyme_number"]]
    if "search_enzyme2_number" in search_params:
        search_params["search_enzyme2_number"] = msfragger2comet_enzymes[search_params["search_enzyme2_number"]]
    if "search_enzyme_number" in search_params:
        search_params["sample_enzyme_number"] = search_params["search_enzyme_number"]
    # Map peptide_length_range
    peptide_length_range = [7, 40] # peptide_length_range_default
    if "digest_min_length" in search_params:
        peptide_length_range[0] = search_params["digest_min_length"]
    if "digest_max_length" in search_params:
        peptide_length_range[1] = search_params["digest_max_length"]
    search_params["peptide_length_range"] = " ".join(list(map(str, peptide_length_range)))

    # Inject the database path
    search_params['database_name'] = db_path
    print("Comet search params to update:\n", search_params)

    # Replace parameters in template
    updated_lines = []
    for line in lines:
        stripped = line.strip()

        # Replace if it's a known param
        if '=' in stripped and not stripped.startswith("#"):
            param_name = stripped.split("=")[0].strip()
            
            if param_name in search_params:
                value = search_params[param_name]
                
                # Format PTM
                if "variable_mod" in param_name:
                    value = msfragger2comet_ptm(value)
                
                updated_lines.append(f"{param_name} = {value}")
                print("UPDATE:", f"{param_name} = {value}")
                continue

        updated_lines.append(line.rstrip())

    # Save output
    with open(output_path, 'w') as out:
        out.write("\n".join(updated_lines) + "\n")
    print(f"✅ Generated Comet params file at: {output_path}")

def get_comet_static_mods(params_path):
    static_mods = {}
    with open(params_path, 'r') as params_file:
        for line in params_file:
            if line.startswith("add_"):
                stripped = line.split("#")[0].strip()
                mod_name, mod_val = stripped.split("=")
                mod_name, mod_val = mod_name.strip(), mod_val.strip()

                if float(mod_val) != 0:
                    # TODO: this doesn't parse terminal statis mods correctly!
                    mod_residue = mod_name.split("_")[1]
                    static_mods[mod_residue] = mod_val
    return static_mods

def add_comet_static_mods_to_peptide(sequence, static_mods):
    """
    Add static modifications used in Comet search to the peptide sequence found
    (Comet doesn't include static mods in the output peptide sequence by itself, 
    but they are expected in the downstream analysis).
    """
    for residue, mod in static_mods.items():
        sequence = sequence.replace(residue, f"{residue}[{mod}]")
    return sequence

def run_comet_search(dset_name, db_search_config):
    """
    Run Comet database search and organize results.

    Args:
        dset_name (str): Dataset name.
        db_search_config (object): Configuration object for database search.
    """
    # Search for mzml files in the directory
    search_ext = db_search_config.ext
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mzml_files = [f for f in os.listdir(mzml_files_dir) if os.path.splitext(f)[1].lower() == search_ext]
    mzml_files = {os.path.splitext(f)[0]: os.path.join(mzml_files_dir, f) for f in mzml_files}
    features_dir = os.path.join(mzml_files_dir, "comet_features")
    os.makedirs(features_dir, exist_ok=True)

    # Generate decoys database
    if db_search_config.pool_proteomes_dir is not None: # TODO: mb use a different check for synthetic peptides?
        # For synthetic peptides, generate separate databased for each file
        pool_proteomes_dir = os.path.join(PROTEOMES_DIR, db_search_config.pool_proteomes_dir)
        PT_pools_name = f"{dset_name}.csv"
        PT_pools_path = os.path.join(PROTEOMES_DIR, PT_pools_name)
        PT_pools_df = pd.read_csv(PT_pools_path)
        files_db_w_decoys = prepare_synthetic_fasta(dset_name, mzml_files, pool_proteomes_dir, PT_pools_df)
    else:
        db_w_decoys_path = generate_decoys_fasta(dset_name, db_search_config.database_path)
        files_db_w_decoys = {fname: db_w_decoys_path for fname in mzml_files}

    # Combine mzml_files with the same database
    db_w_decoys_files = {}
    for fname, db_w_decoys_path in files_db_w_decoys.items():
        if db_w_decoys_path not in db_w_decoys_files:
            db_w_decoys_files[db_w_decoys_path] = []
        db_w_decoys_files[db_w_decoys_path].append(mzml_files[fname])

    # Iterate over each database and run Comet search
    for db_w_decoys_path, mzml_files in db_w_decoys_files.items():
        print("\nDB with decoys:\n", db_w_decoys_path)
        print("Files:\n", mzml_files)

        # Generate Comet params file
        params_path = os.path.join(mzml_files_dir, "comet_search.params")
        generate_comet_params(db_w_decoys_path, db_search_config, COMET_BASE_PARAMS, params_path)

        # Run Comet search
        cmd = [
            COMET_PATH,
            f"-P{params_path}",
            *mzml_files
            # mzml_files_dir + "/*.mzML"
        ]
        subprocess.run(" ".join(cmd), shell=True, check=True)

        # Organize results
        for fname in mzml_files:
            output_fname = os.path.splitext(fname)[0] + ".pin"
            if os.path.exists(os.path.join(mzml_files_dir, output_fname)):
                cmd = ["mv", os.path.join(mzml_files_dir, output_fname), features_dir]
                subprocess.run(" ".join(cmd), shell=True, check=True)

    # Post-process Comet output files (.pin)
    # Comet does not annotate static modifications in the peptide sequences within its output files. 
    static_mods = get_comet_static_mods(params_path)
    print("Static modification to add to Comet DB search result:\n", static_mods)
    
    if static_mods:
        features_files = [fname for fname in os.listdir(features_dir) if fname.endswith(".pin")]
        # Get number of columns
        file_path = os.path.join(features_dir, features_files[0])
        with open(file_path, 'r') as file:
            first_line = file.readline().strip()
        # Split the first line into column names
        column_names = first_line.split("\t")
        n_cols = len(column_names)

        for fname in features_files:
            file_path = os.path.join(features_dir, fname)
            feat_df = pd.read_csv(file_path, sep="\t", usecols=list(range(n_cols)))
            feat_df["Peptide"] = feat_df["Peptide"].apply(lambda x: add_comet_static_mods_to_peptide(x, static_mods))
            feat_df.to_csv(file_path, index=False, sep="\t")

    print("Comet search results (.pin):\n", os.listdir(features_dir))


### Methods to run PSM rescoring with Percolator

def run_psm_rescoring(dset_name, rescoring_config, rescored_files_dir, rescore_file_prefix="rescore"):
    """
    Run Percolator for PSMs rescoring (using MSBooster features for MSFragger).
    Rescoring is performed on all search engine results, 
    but only MSFragger results support RT and spectral features. 
    """
    # TODO: move outside (to constants?)
    num_threads = 3
    test_fdr = 0.01
    train_fdr = 0.01

    input_file = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.pin")
    weights_file = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.weights.csv")
    target_psms = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.psms.txt")
    decoy_psms = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.decoy.psms.txt")
    target_peptides = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.peptides.txt")
    decoy_peptides = os.path.join(rescored_files_dir, f"{rescore_file_prefix}.percolator.decoy.peptides.txt")

    cmd = f"percolator --weights {weights_file} \
            --num-threads {num_threads} \
            --subset-max-train 500000 \
            --post-processing-tdc \
            --testFDR {test_fdr} \
            --trainFDR {train_fdr} \
            --results-psms {target_psms} \
            --decoy-results-psms {decoy_psms} \
            --results-peptides {target_peptides} \
            --decoy-results-peptides {decoy_peptides} \
            {input_file}"
    subprocess.run(cmd, shell=True, check=True)
    print(
        "PSMs rescoring results (.percolator.psms.txt):\n", 
        os.listdir(rescored_files_dir)
    )


### Utility methods to parse and perform checks on PSM IDs (after rescoring)

def get_filename(psm_id, search_tool="msfragger"):
    """
    Extract spectrum file name from the PSM ID after rescoring.
    Supports PSM IDs from MSFragger, MSGF+ and Comet.
    Assumes that there are no `.` in the file name.
    """

    if search_tool == "comet":
        title = psm_id.split("/")[-1]
        title = title.split("_")
        filename = "_".join(title[:-3])
        return filename
    
    elif search_tool == "msgf":
        title = psm_id.split("_")
        filename = "_".join(title[:-6])
        return filename
    
    # else (msfragger):
    filename = psm_id.split(".")[0]
    return filename

def get_spectrum_title(psm_id, search_tool="msfragger"):
    if search_tool == "comet":
        title = psm_id.split("/")[-1]
        title = title.split("_")
        filename, scan_id, charge, rank = title[:-3], title[-3], title[-2], title[-1]
        filename = "_".join(filename)
        title = ".".join([filename, scan_id, scan_id, charge])
        return title
    
    elif search_tool == "msgf":
        title = psm_id.split("_")
        # PSM "index": SII_13752_1 = title[-6:-3], 
        filename, scan_id, charge, rank = title[:-6], title[-5], title[-2], title[-1]
        filename = "_".join(filename)
        title = ".".join([filename, scan_id, scan_id, charge])
        return title
    
    # else (msfragger, msgf):
    title = "_".join(psm_id.split("_")[:-1])
    return title

def check_psm_in_sample_pool(sample_pool, psm_pools): # TODO: should be psm_proteins    
    if isinstance(sample_pool, list):
        return len(set(sample_pool) & set(psm_pools)) > 0
    
    if sample_pool in psm_pools:
        return True
    
    if "QC_JPT_QC_Peptide" in psm_pools or "QC_JPT_RT_Peptide" in psm_pools:
        return True
    return False


### --- TODO: organise utility methods --- 

def format_peptide_notation(sequence: str):
    """TODO: PTMs may need conversion to ProForma notation."""
    # remove cleavage sites
    if (
        re.match(r"[A-Z-_].*.[A-Z-_]", sequence) is not None
    ):  # check is not mandatory
        sequence = sequence[2:-2]
    return sequence

def map_psm_id_index_to_scan_id(results_df):
    
    def get_filename(psm_id):
        return psm_id.split("_SII_")[0]
    
    def map_mzml_index_to_scan_id(psm_id):
        filename = psm_id.split("_SII_")[0]

        psm_id = psm_id.split("_")
        mzml_index = int(psm_id[-5]) - 1
        scan_id = mzml_index_to_scan_id[filename][mzml_index]

        psm_id[-5] = str(scan_id)
        psm_id = "_".join(psm_id)
        return psm_id
    
    results_df["filename"] = results_df["PSMId"].apply(get_filename)
    mzml_files_dir = os.path.join(MZML_DATA_DIR, dset_name)
    mzml_files = results_df.filename.unique().tolist()
    mzml_index_to_scan_id = {}
    for fname in mzml_files:
        file_path = os.path.join(mzml_files_dir, fname + ".mzML")
        mzml_spectra = mzml.MzML(file_path)
        print(fname, len(mzml_spectra))

        index_to_scan_id_mapping = {}
        for spectrum in tqdm(mzml_spectra):
            idx = spectrum["index"]
            scan_id = int(spectrum["id"].split("=")[-1]) # ! Assume id="scanId=X" -- id doesn't contain any other information
            index_to_scan_id_mapping[idx] = scan_id
        mzml_index_to_scan_id[fname] = index_to_scan_id_mapping
        
    results_df["PSMId"] = results_df["PSMId"].apply(map_mzml_index_to_scan_id)
    results_df = results_df.drop("filename", axis=1)
    return results_df

def fix_msgf_timstof_psm_id(results_df):

    def fix_timstof_psm_id(psm_id):
        psm_id = psm_id.split("_")
        psm_id[-5] = psm_id[-3]
        psm_id = "_".join(psm_id)
        return psm_id

    results_df["PSMId"] = results_df["PSMId"].apply(fix_timstof_psm_id)
    return results_df

### --- 


def collect_dataset_tags(config):
    if os.path.exists(DATASET_TAGS_PATH):
        tags_df = pd.read_csv(DATASET_TAGS_PATH, sep="\t").to_dict('records')
    else:
        tags_df = []
        
    # add reference proteome information (for each dataset)
    dset_tags = {"proteome": config.db_search.database_path}
    # add dataset property tags
    dset_tags.update({tag.name: 1 for tag in config.tags})
    print(f"Dataset {config.name} tags:\n", dset_tags)
    
    tags_df.append({"dataset": config.name, **dset_tags})
    tags_df = pd.DataFrame(tags_df).fillna(0)
    tags_df = tags_df.drop_duplicates(subset="dataset", keep="last")
    tags_df.to_csv(DATASET_TAGS_PATH, index=False, sep="\t")
    print(f"Written to {DATASET_TAGS_PATH}")