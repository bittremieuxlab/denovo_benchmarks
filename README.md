# Benchmarking de novo peptide sequencing algorithms

## Adding a new algorithm

Make a pull request to add your algorithm to the benchmarking system.

Add your algorithm in the `denovo_benchmarks/algorithms/algorithm_name` folder by providing  
`container.def`, `make_predictions.sh`, `input_mapper.py`, `output_mapper.py` and `versions.log` files.  
Detailed files descriptions are given below.  

Templates for each file implementation can be found in the 
`algorithms/base/` [folder](https://github.com/bittremieuxlab/denovo_benchmarks/tree/main/algorithms/base).  
It also includes the `InputMapperBase` and `OutputMapperBase` base classes for implementing input and output mappers.  
For examples, you can check 
[Casanovo](https://github.com/bittremieuxlab/denovo_benchmarks/tree/main/algorithms/casanovo) 
and [DeepNovo](https://github.com/bittremieuxlab/denovo_benchmarks/tree/main/algorithms/deepnovo) implementations. 


- **`container.def`** — definition file of the [Apptainer](https://apptainer.org/docs/user/main/definition_files.html) 
container image that creates environment and installs dependencies required for running the algorithm.
    
- **`make_predictions.sh`** — bash script to run the *de novo* algorithm on the input dataset 
(folder with MS spectra in .mgf files) and generate an output file with per-spectrum peptide predictions.  
    **Input**: path to a dataset folder containing .mgf files with spectra data  
    **Output**: output file (in a common output format) containing predictions for all spectra in the dataset

    To configure the model for specific data properties (e.g. non-tryptic data, data from a particular instrument, etc.), please use **dataset tags**. 
    Current set of tags can be found in the `DatasetTag` in [dataset_config.py](https://github.com/bittremieuxlab/denovo_benchmarks/blob/main/dataset_config.py) and includes `nontryptic`, `timstof`, `waters`, `sciex`.
    The tags of each dataset are listed in `dataset_tags.tsv`, together with its reference proteome.
    Example usage can be found in `algorithms/base/make_predictions_template.sh`.

- **`input_mapper.py`** — python script to convert input data 
from its original representation (**input format**) to the format expected by the algorithm.

    **Input format**
    - Input: a dataset folder with separate .mgf files containing MS spectra.
    - Keys order for a spectrum in .mgf file:  
    `[TITLE, RTINSECONDS, PEPMASS, CHARGE]`


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
            `index` — index (0-based) of each spectrum in an .mgf file.
        
    
    - **Output sequence format**
        - 20 amino acid tokens:  
        `G, A, S, P, V, T, C, L, I, N, D, Q, K, E, M, H, F, R, Y, W`
        - Amino acids with post-translational modifications (PTMs) are written in 
        **[ProForma](https://github.com/HUPO-PSI/ProForma/tree/master) format** with **Unimod accession codes** for PTMs:  
        `C[UNIMOD:4]` for Cysteine Carbamidomethylation, `M[UNIMOD:35]` for Methionine Oxidation, etc.
        - N-terminus and C-terminus modifications, if supported by the algorithm, are also written in **ProForma notation** with **Unimod accession codes**:  
        `[UNIMOD:xx]-PEPTIDE-[UNIMOD:yy]`

- **`versions.log`** — list of container versions of the algorithm, newest first. 
The benchmark always runs the newest `container_version`, and outputs and results are stored per version. 
A template with the required fields (`container_version`, `date`, `repo_commit`, `notes`) can be found in `algorithms/base/versions_template.log`.

To check that your container runs and produces predictions in the output format, use `run_test.sh` (see [Running the benchmark](#running-the-benchmark)).
`./run_test.sh [-k] algorithm_name` runs the algorithm on `sample_data/9_species_human` and checks its output. 
It prints the number of valid predictions (`Predictions: N rows, M predicted sequences, K valid.`) and warns about invalid ones 
(sequence not in ProForma format, or number of `aa_scores` not matching the sequence tokens), which the evaluation counts as no prediction.


## System requirements

Building containers and running the benchmark locally requires the following:

- Operating System: Linux (required for Apptainer).
- Dependencies:
    - [Apptainer](https://apptainer.org/docs/user/main/quick_start.html).
    
    Make sure the [Apptainer dependencies](https://github.com/apptainer/apptainer/blob/main/INSTALL.md) are installed. 

    You may also need to install the following packages:
    ```bash
    sudo apt install squashfuse gocryptfs fuse-overlayfs  
    ```

    - Python 3 with the packages in `requirements.txt` ([Streamlit](https://docs.streamlit.io/get-started/installation), pandas, plotly).

    The dashboard was checked with Python 3.11, Streamlit 1.65, pandas 3.0 and plotly 6.5.2.

We run the tools on a high-performance computing (HPC) system using the **Suse Linux Enterprise Server** operating system, equipped with two **Intel Xeon Gold 6526Y processors**, **512 GB of RAM**, and **four NVIDIA L40S GPUs**.

The current source code does not include containerized implementations for **PEAKS** and **GraphNovo** due to their integration within the **PEAKS Studio 12** software, which is only available as a graphical user interface (GUI) tool compatible with the Windows operating system. These tools will be executed manually on a desktop computer running **Windows 11**, equipped with an **Intel Core i9 processor** and **128 GB of RAM**.


## Input data structure

The benchmark expects input data to follow a specific folder structure. 

- Each dataset is stored in a separate folder with unique name.
- **Spectra** are stored as `.mgf` files inside the `mgf/` subfolder.
- **Ground truth labels** (PSMs found via database search) are contained in `labels.csv` file within each dataset folder.

Below is an example layout for our evaluation datasets stored on the HPC:

```
datasets/
    9_species_human/
        labels.csv
        mgf/
            151009_exo3_1.mgf
            151009_exo3_2.mgf
            151009_exo3_3.mgf
            ...
    9_species_solanum_lycopersicum/
        labels.csv
        mgf/...
    9_species_mus_musculus/
        labels.csv
        mgf/...
    9_species_methanosarcina_mazei/
        labels.csv
        mgf/...
    ...
```

Note that algorithm containers only get as input the `/mgf` subfolder with spectra files and **do not** have access to the `labels.csv` file. 
Only the evaluation container accesses the `labels.csv` file to evaluate algorithm predictions.

We provide two small sample datasets (`9_species_human`, `9_species_mus_musculus`) and their reference proteomes (`proteomes/`) 
in the `sample_data/` directory for testing the benchmarking pipeline locally.

However, running the full benchmark, especially on larger spectra files, is **not recommended** on a local computer, as de novo prediction can be computationally intensive and time-consuming. Additionally, while some containerized tool versions support flexible switching between CPU and GPU devices, others strictly require GPU access and will fail to run if a compatible GPU is unavailable.


## Running the benchmark

To run the benchmark locally:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/bittremieuxlab/denovo_benchmarks.git
    cd denovo_benchmarks
    ```

2. **Build containers for algorithms and evaluation**:
    To build all apptainer images, make sure you have [apptainer installed](https://apptainer.org/docs/user/main/quick_start.html#installation). Then run:

    ```bash
    chmod +x build_apptainer_images.sh
    ./build_apptainer_images.sh
    ```

    This will build the apptainer images for all algorithms and the evaluation apptainer image.

    If an apptainer image already exists, the script will ask if you want to rebuild it.

    ```bash
    A .sif image for casanovo already exists. Force rebuild? (y/N) 
    ```

    If a container is missing, that algorithm will be skipped during benchmarking. We don't share or store containers publicly yet due to ongoing development and their large size.

3. **Configure paths:**
    In order to configure the project environment to run the benchmark locally, you need to make a copy of the `.env.template` file and rename it to `.env`. This file contains the necessary environment variables for the project to run properly. 
    
    After renaming the file, update the file paths within the `.env` file to reflect the correct locations on your system.
    Variables:
    - `DATASET_TAGS_PATH` — path to `dataset_tags.tsv` (required)
    - `PROTEOMES_DIR` — folder with the reference proteomes (required)
    - `DATA_DIR`, `WORK_DIR`, `ROOT` — base folders used to build the paths above and by the dataset creation scripts
    - `FRAGPIPE_DIR` — FragPipe installation, only needed for creating datasets

4. **Run benchmark on a dataset**:
    <!-- Make sure the required packages are installed:

    ```bash
    sudo apt install squashfuse gocryptfs fuse-overlayfs  
    ``` -->

    Run the benchmark of an algorithm on a dataset:

    ```bash
    ./run.sh [-r] [-q] [--no-eval] /path/to/dataset/dir algorithm_name
    ```
    Example:
    ```bash
    ./run.sh sample_data/9_species_human casanovo
    ```

    Options:
    - `-r` — recalculate the algorithm output, even if it already exists
    - `-q` — quiet mode: hide the output of the algorithm and of MMseqs2 unless they fail
    - `--no-eval` — skip the evaluation step

    The script runs the latest algorithm container version (from `algorithms/<algorithm>/versions.log`), 
    augments the predictions and evaluates them. 
    Outputs are stored as:
    ```
    outputs/<algorithm>/<version>/<dataset>/
        output.csv   # predictions
        time.log     # algorithm run time
    ```
    Evaluation results are stored in `results/<dataset>/`. Only the rows of the evaluated algorithm version 
    are added or replaced, results of other algorithms are kept.

5. **Run several datasets and algorithms**:

    ```bash
    ./run_array.sh [-r] [-q] -d /path/to/dataset1 [-d /path/to/dataset2 ...] [-a algorithm1 -a algorithm2 ...]
    ```
    Runs `run.sh` for every dataset and algorithm (default: all algorithms with a `container.def`), 
    then evaluates each dataset. 

6. **Run an algorithm on parts of a dataset** (e.g. for large datasets):

    ```bash
    ./run_split.sh /path/to/dataset/dir algorithm_name number_of_parts
    ```
    Runs the algorithm separately on parts of the dataset's `.mgf` files, merges the part outputs 
    into `outputs/<algorithm>/<version>/<dataset>/output.csv`, then augments and evaluates them. 
    If some part fails, outputs are not merged. A rerun only runs the missing parts.
```


## Benchmark results

Evaluation results are stored in `results/<dataset>/` (one plot data file per metric, one row per algorithm version) and are shown in the dashboard.

## Running Streamlit dashboard locally:
To view the Streamlit dashboard for the benchmark locally, run:
```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

