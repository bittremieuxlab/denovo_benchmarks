#!/bin/bash

# Get dataset property tags
DSET_TAGS=$(python /algo/base/dataset_tags_parser.py --dataset "$@")
while IFS='=' read -r key value; do
    export "$key"="$value"
done <<< "$DSET_TAGS"

# Choose the device  
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        device=0 # GPU:0
    else
        device=-1 # CPU
    fi
else
    device=-1 # CPU
fi

# Use tag variables to specify de novo algorithm
# for the particular dataset properties
cd /algo
mkdir -p /algo/input_data

# Initialize output file to collect per-file outputs and add output file header
echo -e "sequence,score,aa_scores,spectrum_id" > /algo/denovo_outputs.csv

# Iterate through files in the dataset
for input_file in "$@"/*.mgf; do

    # Clean input dir (previous input files)
    rm -rf /algo/input_data/*

    # Extract just the filename without the path
    filename=$(basename "$input_file")
    echo "Processing file: $input_file"

    # Convert input data to model format
    python input_mapper.py \
        --input_path "$input_file" \
        --output_path "/algo/input_data/$filename" \
        --config_path "pi-HelixNovo/config.yaml"

    # Run de novo algorithm on the input data
    python pi-HelixNovo/main.py \
        --mode=denovo \
        --config=pi-HelixNovo/config.yaml \
        --gpu=$device \
        --output=denovo_output.csv \
        --peak_path="/algo/input_data/$filename" \
        --model=pi-HelixNovo/pi-helixnovo_massivekb.ckpt

    # Collect predictions (from algorithm output file denovo_output.csv)
    tail -n+2 denovo_output.csv >> /algo/denovo_outputs.csv
    cd /algo
done

# Convert predictions to the general output format
echo "Converting outputs:"
python ./output_mapper.py --output_path=denovo_outputs.csv
