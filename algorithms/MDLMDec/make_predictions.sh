#!/bin/bash
conda init
source /opt/conda/etc/profile.d/conda.sh
conda activate temp

if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    device=0
    accelerator=gpu
else
    device=-1
    accelerator=cpu
fi
echo "Using device $device with accelerator $accelerator."

cd /algo

# Turn mgf files into parquet inputs
python mgf_to_parquet.py --input /algo/"$@"/ --divisions 2

#TODO: add checkpoint
python apply_standalone_model.py

cd /algo

# Placeholder for output mapper:
python output_mapper.py --input_dir /algo/"$@"/
