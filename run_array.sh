#!/bin/bash
# Run run.sh for every combination of the given datasets and algorithms,
# then evaluate each dataset once for all algorithms that produced output.

datasets=()
algorithms=()
run_args=()
eval_args=""
recalculate=false
while [ $# -gt 0 ]; do
    case "$1" in
        -d) datasets+=("$2"); shift ;;
        -a) algorithms+=("$2"); shift ;;
        -q) run_args+=(-q); eval_args="--quiet" ;;
        -r) run_args+=(-r); recalculate=true ;;
        *) echo "Error: invalid argument $1" >&2; exit 1 ;;
    esac
    shift
done

if [ ${#datasets[@]} -eq 0 ]; then
    echo "Error: at least one dataset is required (-d <dataset_dir>)" >&2
    exit 1
fi
if [ ${#algorithms[@]} -eq 0 ]; then
    for container_def in algorithms/*/container.def; do
        algorithms+=("$(basename "$(dirname "$container_def")")")
    done
fi

logs_dir="logs/run_array_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$logs_dir"
summary_file="$logs_dir/summary.tsv"
printf "dataset\talgorithm\tstatus\tdetails\tlog\n" > "$summary_file"
echo "Datasets: ${datasets[*]}"
echo "Algorithms: ${algorithms[*]}"
echo "Logs: $logs_dir"

for dset_dir in "${datasets[@]}"; do
    dset_name=$(basename "$dset_dir")
    evaluate_algorithms=""

    # 1. Run each algorithm
    for algorithm_name in "${algorithms[@]}"; do
        log_file="$logs_dir/${dset_name}__${algorithm_name}.log"
        algorithm_version=$(grep -m 1 "container_version:" "algorithms/${algorithm_name}/versions.log" 2>/dev/null | awk -F'"' '{print $2}')
        output_file="outputs/${algorithm_name}/${algorithm_version}/${dset_name}/output.csv"

        if [ ! -d "algorithms/${algorithm_name}" ] || [ "$algorithm_name" = "base" ]; then
            status="FAILED"; details="unknown algorithm"; log_file="-"
        elif [ -z "$algorithm_version" ]; then
            status="FAILED"; details="no container_version in versions.log"; log_file="-"
        elif { [ "$recalculate" = true ] || [ ! -e "$output_file" ]; } \
                && [ ! -f "algorithms/${algorithm_name}/container.sif" ]; then
            status="SKIPPED"; details="no container"; log_file="-"
        else
            details="new output"
            if [ -e "$output_file" ] && [ "$recalculate" = false ]; then
                details="existing output reused"
            fi
            echo "RUN $algorithm_name on $dset_name"
            bash run.sh "${run_args[@]}" --no-eval "$dset_dir" "$algorithm_name" > "$log_file" 2>&1

            # Status from the output file: missing -> algorithm failed, no SA/pred_RT columns -> augmentation failed
            header=",$(head -n 1 "$output_file" 2>/dev/null),"
            if [ ! -s "$output_file" ]; then
                status="FAILED"; details="algorithm failed"
            elif [[ "$header" != *",SA,"* || "$header" != *",pred_RT,"* ]]; then
                status="FAILED"; details="augmentation failed"
            else
                status="OK"
                evaluate_algorithms="$evaluate_algorithms ${algorithm_name}:${algorithm_version}"
            fi
        fi
        echo "  $status: $details"
        printf "%s\t%s\t%s\t%s\t%s\n" \
            "$dset_name" "$algorithm_name" "$status" "$details" "$log_file" >> "$summary_file"
    done

    # 2. Evaluate the dataset
    log_file="$logs_dir/${dset_name}__evaluation.log"
    if [ -z "$evaluate_algorithms" ]; then
        status="SKIPPED"; details="no algorithm output to evaluate"; log_file="-"
    else
        echo "EVALUATE $dset_name:$evaluate_algorithms"
        apptainer exec --fakeroot --env-file .env -B "$(realpath "$dset_dir")" "evaluation.sif" \
            bash -c "python -m evaluation.evaluate ./outputs/ ${dset_dir} --algorithms${evaluate_algorithms} ${eval_args}" \
            > "$log_file" 2>&1
        eval_status=$?
        if [ $eval_status -eq 0 ]; then
            status="OK"; details="evaluated:$evaluate_algorithms"
        else
            status="FAILED"; details="evaluation failed (exit $eval_status)"
        fi
    fi
    echo "  evaluation $status: $details"
    printf "%s\t%s\t%s\t%s\t%s\n" \
        "$dset_name" "(evaluation)" "$status" "$details" "$log_file" >> "$summary_file"
done

# Summary table
echo
echo "SUMMARY ($summary_file)"
column -t -s $'\t' "$summary_file"
! grep -q "	FAILED	" "$summary_file"
