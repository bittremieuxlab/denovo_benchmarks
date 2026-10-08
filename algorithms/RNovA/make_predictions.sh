#!/bin/bash
# RNovA SeqFiller for the denovo_benchmarks harness.
#
# $1: folder holding the dataset .mgf files (run.sh passes the dataset name,
#     mounted at /algo/<name>; paths are resolved from the current directory).
#
# Settings (tuned for amino-acid AUC on the 9-species human sample, checked on the
# mouse and tomato samples):
#   - candidate residues: candidates.txt, the 20 amino acids with carbamidomethyl C
#     (+57.021) and oxidised M (+15.995), for every dataset whatever its PTM tags;
#   - precursor tolerance 10 ppm;
#   - precursor isotope shifts -1,0: each spectrum is also decoded with its precursor one
#     13C lighter (a mis-picked monoisotopic peak) and the higher-scoring result kept;
#   - 8 refinement iterations;
#   - RNovA does not report sequences whose mean residue score is below -2.0 (fixed in
#     Inference_Sequence.py, not configurable).
# Tolerance and iterations are also RNovA's own defaults; they are spelled out here so
# every run logs them.
#
# Environment (all optional, e.g. via the harness --env-file):
#   RNOVA_CANDIDATES     ';'-separated candidate list overriding candidates.txt,
#                        e.g. "A;C|UniMod:4;...;Y;M|UniMod:35;N[0.984]".
#   RNOVA_BATCH_SIZE     SeqFiller batch size (default 128, ~16 GB of GPU memory).
#   RNOVA_PRECURSOR_PPM  precursor tolerance (default 10).
#   RNOVA_MAX_ITER       refinement iterations (default 8).
#   RNOVA_ISOTOPES       precursor isotope shifts (default "-1,0"; "0" disables;
#                        each shift adds one full decode).
export RNOVA_PRECURSOR_PPM="${RNOVA_PRECURSOR_PPM:-10}"
export RNOVA_MAX_ITER="${RNOVA_MAX_ITER:-8}"
RNOVA_ISOTOPES="${RNOVA_ISOTOPES:--1,0}"
set -euo pipefail

ALGO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$(cd "$1" && pwd)"
CANDIDATES="${RNOVA_CANDIDATES:-$(cat "$ALGO_DIR/candidates.txt")}"
export RNOVA_KNAPSACK_CACHE="${RNOVA_KNAPSACK_CACHE:-$ALGO_DIR/knapsack_cache}"

# never leave a previous run's predictions behind for the harness to collect
rm -f "$ALGO_DIR/outputs.csv"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rnova.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Using RNovA SeqFiller with candidates: $CANDIDATES"
echo "Precursor tolerance ${RNOVA_PRECURSOR_PPM} ppm, isotope shifts ${RNOVA_ISOTOPES}, ${RNOVA_MAX_ITER} iterations"

# Convert input data to model format (all files merged, spectrum ids in SCANS)
python "$ALGO_DIR/input_mapper.py" \
    --input_dir "$DATASET_DIR" \
    --isotopes="$RNOVA_ISOTOPES" \
    --output_path "$WORK_DIR/spectra.mgf"

# Run de novo algorithm on the input data
# (SeqFiller loads configs/ and save/ relative to its own folder)
(cd "$ALGO_DIR/RNovA/RNovA_SeqFiller_Inference" && \
    python Inference_Sequence.py "$WORK_DIR/spectra.mgf" "$CANDIDATES")

# Convert predictions to the general output format
(cd "$ALGO_DIR" && \
    python output_mapper.py \
        --output_path "$WORK_DIR/spectra_rnova_denovo_seq.csv" \
        --result_path "$ALGO_DIR/outputs.csv")
