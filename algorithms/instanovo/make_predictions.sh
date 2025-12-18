#!/bin/bash

# Get dataset property tags
DSET_TAGS=$(python /algo/base/dataset_tags_parser.py --dataset "$@")
# Parse tags and set individual environment variables for each of them
# (variable names are identical to tag names
#  -- check DatasetTag values in dataset_config.py)
while IFS='=' read -r key value; do
    echo "exporting: $key = $value"
    export "$key"="$value"
done <<< "$DSET_TAGS"

# Run de novo algorithm on the input data
# python -m instanovo.transformer.predict data_path="$1/*.mgf" model_path="/algo/instanovo_extended.ckpt" denovo=True output_path=denovo_outputs.csv 
# TODO: add refinement? (takes long)
instanovo predict --data-path="$1/*.mgf" --instanovo-model="/algo/instanovo-v1.2.0.ckpt" --denovo --no-refinement --output-path=/algo/denovo_outputs.csv batch_size=64 num_workers=4 fp16=false

# Convert predictions to the general output format
python output_mapper.py --output_path=/algo/denovo_outputs.csv
