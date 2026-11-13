# Methods to run MMseqs2 and calculate number of proteome matches

import os
import subprocess
import pandas as pd


def isoleucine_to_leucine(sequence):
    return sequence.replace("I", "L") if isinstance(sequence, str) else sequence


def run_command(cmd, quiet=False):
    """Run a shell command. If quiet, show its output only when it fails."""
    if not quiet:
        subprocess.run(" ".join(cmd), shell=True, check=True)
        return
    result = subprocess.run(
        " ".join(cmd), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, result.args)


def setup_mmseqs_dirs(search_tmp_dir="./mmseqs2_tmp"):
    """Create directories for MMseqs2 proteome matches search."""
    # dir for MMseqs2 files
    os.makedirs(search_tmp_dir, exist_ok=True) 
    # dir for MMseqs2 runs tmp files
    tmp_files_dir = os.path.join(search_tmp_dir, "tmp")
    os.makedirs(tmp_files_dir, exist_ok=True)
    # dir target DB (multiple files)
    target_db_dir = os.path.join(search_tmp_dir, "targetDB")
    os.makedirs(target_db_dir, exist_ok=True)
    # dir query DB (multiple files)
    query_db_dir = os.path.join(search_tmp_dir, "queryDB")
    os.makedirs(query_db_dir, exist_ok=True)
    # dir raw search results (multiple files)
    search_result_dir = os.path.join(search_tmp_dir, "results")
    os.makedirs(search_result_dir, exist_ok=True)
    # path to search results in .m8 format
    search_result_path = os.path.join(search_tmp_dir, "search_result.m8") 
    return search_tmp_dir, tmp_files_dir, target_db_dir, query_db_dir, search_result_dir, search_result_path


# def create_query_fasta(output_data, query_fasta_path):
#     """
#     Create query database of de novo predicted peptides 
#     (with I replaced by L). De novo peptide sequences are
#     treated as non-unique and associated with their original
#     index in `output_data` dataframe. 
#     """
    
#     with open(query_fasta_path, "w") as f:
#         f.write(
#             "\n".join([
#                 f">{idx}\n{isoleucine_to_leucine(peptide)}" # I -> L
#                 for idx, peptide 
#                 in output_data.sequence_no_ptm.reset_index().values
#             ])
#         )
#     print("queryDB (fasta):", query_fasta_path)

def create_query_fasta(query_sequences, query_fasta_path):
    """
    Write the list of sequences to the query database. 
    List of sequences is assumed to contain unique peptide sequences,
    without modifications and with I replaced by L. 
    """
    
    with open(query_fasta_path, "w") as f:
        f.write(
            "\n".join([
                f">{idx}\n{sequence}"
                for idx, sequence 
                in enumerate(query_sequences)
            ])
        )
    print("queryDB (fasta):", query_fasta_path)


def create_target_fasta(reference_proteome_path, contam_path, target_fasta_path):
    """
    Create target database for MMseqs2 search:
    - join reference proteome with common contaminants
    - replace I with L (indistinguishable by mass)
    """
    
    cmd = [
        "cat",
        reference_proteome_path,
        contam_path,
        "| sed -e 's/I/L/g'",
        ">",
        target_fasta_path,
    ]
    subprocess.run(" ".join(cmd), shell=True, check=True)
    print("targetDB (fasta):", target_fasta_path)


def run_mmseqs(
    target_fasta_path,
    query_fasta_path,
    target_db_dir,
    query_db_dir,
    search_result_dir,
    search_result_path,
    tmp_files_dir,
    args=[],
    quiet=False,
):
    SEARCH_COLS = "query,target,qseq,qaln,taln,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    QUERY_KEY = "query"
    
    print("\n CLEAN EXISTING RESULTS")
    cmd = ["rm -rf", search_result_dir + "/*"]
    run_command(cmd, quiet)
    
    print("\n CREATE TARGET DB")
    target_db_path = os.path.join(target_db_dir, "targetDB")
    cmd = [
        "mmseqs",
        "createdb",
        target_fasta_path,
        target_db_path,
        "-v 1",
    ]
    print(" ".join(cmd))
    run_command(cmd, quiet)
    
    print("\n CREATE QUERY DB")
    query_db_path = os.path.join(query_db_dir, "queryDB")
    cmd = [
        "mmseqs",
        "createdb",
        query_fasta_path,
        query_db_path,
        "-v 1",
    ]
    print(" ".join(cmd))
    run_command(cmd, quiet)
    
    print("\n SEARCH/MAP SEQUENCES")
    # mmseqs map <i:queryDB> <i:targetDB> <o:alignmentDB> <tmpDir> [options]
    cmd = [
        "mmseqs",
        "map",
        query_db_path,
        target_db_path,
        os.path.join(search_result_dir, "search_result"),
        tmp_files_dir,
        "--cov-mode 2", # coverage of query
        "-c 1.0",
        "-s 4", # sensitivity # 7.5?
        "-a", # backtrace alignments 
        "-e inf",
    ] + args
    print(" ".join(cmd))
    run_command(cmd, quiet)

    print("\n CONVERT RESULTS TO .m8")
    # mmseqs convertalis queryDB targetDB resultDB resultDB.m8
    cmd = [
        "mmseqs",
        "convertalis",
        query_db_path,
        target_db_path,
        os.path.join(search_result_dir, "search_result"),
        search_result_path,
        f"--format-output {SEARCH_COLS}",
        "-v 1"
    ]
    print(" ".join(cmd))
    run_command(cmd, quiet)

    # Load search results
    search_df = pd.read_csv(search_result_path, sep="\t", names=SEARCH_COLS.split(","))
    # Filter out acceptable matches
    search_df = search_df.sort_values(by=[QUERY_KEY, "mismatch"])
    search_df = search_df.drop_duplicates(QUERY_KEY, keep="first")
    search_df = search_df[search_df["mismatch"] <= 2] # filter s.t. n_mismatches <= 2

    return search_df # only contains sequence-match pairs for unique seqeuences
