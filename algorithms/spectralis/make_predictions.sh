#!/bin/bash

# Make pyenvs available
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Run Casanovo to obtain initial starting sequences
pyenv activate casanovo
casanovo_output="outputs.mztab"

# Get dataset property tags
DSET_TAGS=$(python /algo/base/dataset_tags_parser.py --dataset "$@")
while IFS='=' read -r key value; do
    export "$key"="$value"
done <<< "$DSET_TAGS"

# Run de novo algorithm on the input data
echo "Using general model."
casanovo sequence -m /algo/casanovo_v4_2_0.ckpt -c casanovo_config.yml -o $casanovo_output "$@"/*.mgf

# Change to spectralis pyenv
pyenv deactivate
pyenv activate spectralis

# Write the initial starting sequences to the input MGFs
spectralis_data_dir="./seq_data"
mkdir -p $spectralis_data_dir
python /algo/intermediate_mapper.py --mztab_path $casanovo_output --mgf_in_dir "$@" --mgf_out_dir $spectralis_data_dir

# Run Spectralis on every MGF file in the input directory
input_mgf_paths=$spectralis_data_dir/*.mgf
mkdir -p ./spectralis_outputs
spectralis_outputs=./spectralis_outputs/
for mgf_file in $input_mgf_paths; do
    file_name=$(basename -- "$mgf_file")
    spectralis --mode="rescoring" --config="spectralis_config.yml" --input_path="$mgf_file" --output_path="${spectralis_outputs}${file_name%.mgf}.csv"
done

# Format the output into the target format
python /algo/output_mapper.py --mgf_in_dir "$@" --spectralis_output_dir $spectralis_outputs --casanovo_output $casanovo_output