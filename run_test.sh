#!/bin/bash
algorithm_name="$1"
test_dset_dir="sample_data/9_species_human"
test_dset_name=$(basename "$test_dset_dir")
test_spectra_dir="$test_dset_dir/mgf"
overlay_size=512
test_output_dir="./test_outputs"

# Create the output directory if it doesn't exist
mkdir -p "${test_output_dir}"

# List input files
echo "Processing test dataset"
ls "$test_spectra_dir"/*.mgf

# Check if the output file does not exist
echo "Processing algorithm: $algorithm_name"

# [Optional] Create a test container (from container.def)
# apptainer build --fakeroot \
#     "algorithms/${algorithm_name}/test_container.sif" \
#     "algorithms/${algorithm_name}/container.def"

# Create writable overlay for the container
apptainer overlay create --fakeroot --size $overlay_size \
    --sparse "algorithms/${algorithm_name}/test_overlay.img"

# Calculate predictions
echo "RUN ALGORITHM"
apptainer exec --fakeroot --nv \
    --overlay "algorithms/${algorithm_name}/test_overlay.img" \
    -B "${test_spectra_dir}":"/algo/${test_dset_name}" \
    --env-file .env \
    "algorithms/${algorithm_name}/container.sif" \
    bash -c "cd /algo && ./make_predictions.sh ${test_dset_name}"

# Collect predictions in output_dir
echo "EXPORT PREDICTIONS"
apptainer exec --fakeroot \
    --overlay "algorithms/${algorithm_name}/test_overlay.img" \
    -B "${test_output_dir}":/algo/outputs \
    --env-file .env \
    "algorithms/${algorithm_name}/container.sif" \
    bash -c "cp /algo/outputs.csv /algo/outputs/test_output.csv"

# Check predictions output format
# Use evaluation.sif container for running output format tests
echo "VALIDATE PREDICTIONS OUTPUT FORMAT"
apptainer exec --fakeroot "evaluation.sif" \
    bash -c "python test_output_format.py"
validation_status=$?

if [ $validation_status -eq 0 ]; then
    echo "OUTPUT FORMAT VALIDATED."
else
    echo "OUTPUT FORMAT VALIDATION FAILED."
fi

# Remove test container image and overlay
# TODO: make a flag to not remove container if needed
# rm -rf "algorithms/${algorithm_name}/test_container.sif"
rm -rf "algorithms/${algorithm_name}/test_overlay.img"
rm -rf "${test_output_dir}"

exit $validation_status
