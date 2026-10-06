#!/bin/bash
# Run a given algorithm on a given dataset (without splitting)

recalculate=false
evaluate=true
quiet=false

positional=()
while [ $# -gt 0 ]; do
    case "$1" in
        -r) recalculate=true ;;
        --no-eval) evaluate=false ;;
        -q|--quiet) quiet=true ;;
        -*) echo "Error: invalid option $1" >&2; exit 1 ;;
        *) positional+=("$1") ;;
    esac
    shift
done
set -- "${positional[@]}"

if [ $# -ne 2 ]; then
    echo "Error: expected <dataset_dir> <algorithm_name>, got $# argument(s)" >&2
    exit 1
fi

dset_dir="$1"
algorithm_name="$2"
dset_name=$(basename "$dset_dir")
spectra_dir="$dset_dir/mgf"
output_root_dir="./outputs"
overlay_size=16384 #8192 # in MB, adjust as needed

# Check if algorithm exists and is not "base"
if [ ! -d "algorithms/${algorithm_name}" ]; then
    echo "Error: Algorithm '${algorithm_name}' not found in algorithms/" >&2
    exit 1
fi

if [ "${algorithm_name}" = "base" ]; then
    echo "Error: 'base' is not an algorithm" >&2
    exit 1
fi

# Extract latest container version
# (always runs latest container version. 
# Switch to earlier container state via git history if older version is needed).
algorithm_version=$(grep -m 1 "container_version:" "algorithms/${algorithm_name}/versions.log" | awk -F'"' '{print $2}')
# Validate version extraction worked
if [ -z "$algorithm_version" ]; then
    echo "Error: Could not extract container_version from algorithms/${algorithm_name}/versions.log" >&2
    exit 1
fi

echo "Running benchmark with $algorithm_name on dataset $dset_name."
echo "Using algorithm version: $algorithm_version."
echo "Recalculate the algorithm output: $recalculate."
echo "Evaluate predictions: $evaluate."
echo "Quiet mode (show tool output only on failure): $quiet."

output_dir="$output_root_dir/$algorithm_name/$algorithm_version/$dset_name"

if [ "$recalculate" = true ]; then
    # Clean output dir 
    rm -rf "$output_dir"
fi

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# List input files
echo "Processing dataset: $dset_name ($dset_dir)"
ls "$spectra_dir/"*.mgf

# 1. Run algorithm & get predictions
time_log_file="$output_dir/time.log"
output_file="$output_dir/output.csv"
echo "Output file: $output_file"

# Check if the output file does not exist
if [ ! -e "$output_file" ]; then
    echo "Processing algorithm: $algorithm_name"

    # Remove an existing container overlay, if any
    rm -rf "algorithms/${algorithm_name}/overlay_${dset_name}.img"
    # Create writable overlay for the container
    apptainer overlay create --fakeroot --size $overlay_size --sparse "algorithms/${algorithm_name}/overlay_${dset_name}.img"

    # Calculate predictions
    echo "RUN ALGORITHM $algorithm_name"
    tool_log="$output_dir/predict.log"
    if [ "$quiet" = true ]; then
        exec 3> "$tool_log"
    else
        exec 3>&1
    fi
    { time ( apptainer exec --fakeroot --nv \
        --overlay "algorithms/${algorithm_name}/overlay_${dset_name}.img" \
        -B "${spectra_dir}":"/algo/${dset_name}" \
        --env-file .env \
        "algorithms/${algorithm_name}/container.sif" \
        bash -c "cd /algo && ./make_predictions.sh ${dset_name}" 2>&1 ) >&3; } 2> "$time_log_file"
    predict_status=$?
    exec 3>&-
    if [ "$quiet" = true ]; then
        if [ $predict_status -ne 0 ]; then
            echo "Algorithm $algorithm_name failed (exit $predict_status). Last 50 lines of tool output:"
            tail -n 50 "$tool_log"
            echo "Full tool output: $tool_log"
        else
            rm -f "$tool_log"
        fi
    fi
    
    # Collect predictions in output_dir
    echo "EXPORT PREDICTIONS"
    apptainer exec --fakeroot \
        --overlay "algorithms/${algorithm_name}/overlay_${dset_name}.img" \
        -B "${output_dir}":/algo/outputs \
        --env-file .env \
        "algorithms/${algorithm_name}/container.sif" \
        bash -c "cp /algo/outputs.csv /algo/outputs/output.csv"

else
    echo "Skipping running algorithm: $algorithm_name. Output file already exists."
fi

# Remove the container overlay once predictions are exported (keep it for debugging if export failed)
if [ -e "$output_file" ]; then
    rm -rf "algorithms/${algorithm_name}/overlay_${dset_name}.img"
fi

# 2. Augment predictions with predicted RT and SA between predictied and experimental spectra
echo "Output file: $output_file"
# Augment algorithm predictions with RT and SA (if not already present)
echo "AUGMENT PREDICTIONS"
apptainer exec --fakeroot --env-file .env "evaluation.sif" \
    bash -c "python -m evaluation.augment_predictions --output_dir ${output_dir} --data_dir ${dset_dir}"
# TODO: fix augment_predictions semantics, only pass necessary information

# 3. Evaluate predictions
# (only this algorithm version is evaluated; results of other algorithms are kept)
# TODO: add results_dir explicit definition
if [ "$evaluate" = true ]; then
    echo "EVALUATE PREDICTIONS"
    eval_args="--algorithms ${algorithm_name}:${algorithm_version}"
    if [ "$quiet" = true ]; then
        eval_args="$eval_args --quiet"
    fi
    apptainer exec --fakeroot --env-file .env "evaluation.sif" \
        bash -c "python -m evaluation.evaluate ${output_root_dir}/ ${dset_dir} ${eval_args}"
    # apptainer exec --fakeroot --env-file .env "evaluation.sif" \
    #     bash -c "python -m evaluation.evaluate ${output_root_dir}/ ${dset_dir} --skip_proteome_matches"
fi
