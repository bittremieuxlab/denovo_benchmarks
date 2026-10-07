#!/usr/bin/env bats

# Integration tests for run.sh
# Verifies run.sh works end-to-end with mocked external dependencies

bats_require_minimum_version 1.5.0

setup() {
    export TEST_DIR="$(mktemp -d)"
    export ORIG_DIR="$PWD"
    
    # Create mock project structure
    mkdir -p "$TEST_DIR/sample_data/test_dataset/mgf"
    mkdir -p "$TEST_DIR/algorithms/mock_algo"
    mkdir -p "$TEST_DIR/algorithms/base"
    
    # Create mock MGF files
    cat > "$TEST_DIR/sample_data/test_dataset/mgf/spectrum1.mgf" <<EOF
BEGIN IONS
TITLE=Spectrum 1
PEPMASS=500.25
CHARGE=2+
100.0 1000.0
200.0 2000.0
END IONS
EOF
    
    cat > "$TEST_DIR/sample_data/test_dataset/mgf/spectrum2.mgf" <<EOF
BEGIN IONS
TITLE=Spectrum 2
PEPMASS=650.30
CHARGE=3+
150.0 1500.0
250.0 2500.0
END IONS
EOF
    
    # Create versions.log for mock algorithm
    cat > "$TEST_DIR/algorithms/mock_algo/versions.log" <<EOF
- container_version: "mock-1.0.0"
  date: "2025-10-06"
  repo_commit: "abc123"
  notes: "Mock algorithm for testing"
EOF
    
    # Create mock make_predictions.sh
    cat > "$TEST_DIR/algorithms/mock_algo/make_predictions.sh" <<'SCRIPT'
#!/bin/bash
# Mock prediction script - writes to current directory (algorithm dir)
# In real containers, /algo is bind-mounted to this directory
dataset_name="$1"

echo "scan,sequence,score" > outputs.csv
echo "1,PEPTIDE,0.95" >> outputs.csv
echo "2,SEQUENCE,0.87" >> outputs.csv
exit 0
SCRIPT
    chmod +x "$TEST_DIR/algorithms/mock_algo/make_predictions.sh"
    
    # Create mock container (dummy file)
    touch "$TEST_DIR/algorithms/mock_algo/container.sif"
    
    # Create mock .env file
    touch "$TEST_DIR/.env"
    
    # Setup mock apptainer and environment from the centralized script
    mkdir -p "$TEST_DIR/mock_env"
    cp "$ORIG_DIR/tests/mock_env/setup_mock.sh" "$TEST_DIR/mock_env/"
    cp "$ORIG_DIR/tests/mock_env/teardown_mock.sh" "$TEST_DIR/mock_env/"
    
    # Source the centralized mock environment setup with error checking
    if ! source "$TEST_DIR/mock_env/setup_mock.sh"; then
        echo "Failed to setup mock environment" >&2
        exit 1
    fi
    
    # Extend PATH to also include test directory bin (for additional test-specific mocks)
    mkdir -p "$TEST_DIR/bin"
    export PATH="$TEST_DIR/bin:$PATH"
    
    # Create mock python to skip evaluation steps in tests
    cat > "$TEST_DIR/bin/python" <<'EOF'
#!/bin/bash
exit 0
EOF
    chmod +x "$TEST_DIR/bin/python"
    
    # Create mock evaluation.sif (run.sh calls this container for augment_predictions and evaluate)
    touch "$TEST_DIR/evaluation.sif"
    
    # Copy run.sh to test environment
    cp "$ORIG_DIR/run.sh" "$TEST_DIR/"
    
    # Change to test directory for test execution
    cd "$TEST_DIR"
}

teardown() {
    # Restore to original directory
    cd "$ORIG_DIR"
    
    # Source centralized teardown if mock environment was used
    if [ -f "$TEST_DIR/mock_env/teardown_mock.sh" ]; then
        source "$TEST_DIR/mock_env/teardown_mock.sh" 2>/dev/null || true
    fi
    
    # Cleanup test directory (preserve if DEBUG is set)
    if [ -z "${DEBUG:-}" ]; then
        rm -rf "$TEST_DIR"
    else
        echo "Test directory preserved at: $TEST_DIR" >&2
    fi
}

# Test 1: Script runs successfully with valid dataset and algorithm
@test "integration: run.sh executes successfully with mocked environment" {
    run bash run.sh sample_data/test_dataset mock_algo
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"Running benchmark with mock_algo"* ]]
    [[ "$output" == *"Using algorithm version: mock-1.0.0"* ]]
}

# Test 2: Script creates correct output directory structure
@test "integration: run.sh creates algorithm/version/dataset directory structure" {
    bash run.sh sample_data/test_dataset mock_algo
    
    [ -d "outputs/mock_algo/mock-1.0.0/test_dataset" ]
}

# Test 3: Script creates expected output files with correct names
@test "integration: run.sh creates output.csv and time.log" {
    bash run.sh sample_data/test_dataset mock_algo
    
    [ -f "outputs/mock_algo/mock-1.0.0/test_dataset/output.csv" ]
    [ -f "outputs/mock_algo/mock-1.0.0/test_dataset/time.log" ]
}

# Test 4: Output file contains expected data
@test "integration: run.sh generates valid CSV output" {
    bash run.sh sample_data/test_dataset mock_algo
    
    output_file="outputs/mock_algo/mock-1.0.0/test_dataset/output.csv"
    
    # Check header exists
    head -n 1 "$output_file" | grep -q "scan,sequence,score"
    
    # Check data rows exist
    [ "$(wc -l < "$output_file")" -gt 1 ]
}

# Test 4b: Verify run.sh uses correct file paths internally
@test "integration: run.sh defines correct output_file and time_log_file variables" {
    # Extract variable definitions from run.sh and verify they match expected paths
    time_log_def=$(grep '^time_log_file=' run.sh | head -n 1)
    output_file_def=$(grep '^output_file=' run.sh | head -n 1)
    
    # These should match the expected pattern
    [[ "$time_log_def" == 'time_log_file="$output_dir/time.log"' ]]
    [[ "$output_file_def" == 'output_file="$output_dir/output.csv"' ]]
}

# Test 5: Script skips existing outputs without -r flag
@test "integration: run.sh skips calculation when output exists" {
    # First run
    bash run.sh sample_data/test_dataset mock_algo
    
    # Create marker in output file
    output_file="outputs/mock_algo/mock-1.0.0/test_dataset/output.csv"
    echo "MARKER_LINE" >> "$output_file"
    
    # Second run without -r
    run bash run.sh sample_data/test_dataset mock_algo
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"Skipping running algorithm"* ]]
    
    # Marker should still exist
    grep -q "MARKER_LINE" "$output_file"
}

# Test 6: Script recalculates with -r flag
@test "integration: run.sh recalculates output with -r flag" {
    # First run
    bash run.sh sample_data/test_dataset mock_algo
    
    # Create marker in output file
    output_file="outputs/mock_algo/mock-1.0.0/test_dataset/output.csv"
    echo "MARKER_LINE" >> "$output_file"
    
    # Second run with -r
    run bash run.sh -r sample_data/test_dataset mock_algo
    
    [ "$status" -eq 0 ]
    [[ "$output" != *"Skipping"* ]]
    
    # Marker should be gone, file should be recreated
    ! grep -q "MARKER_LINE" "$output_file"
    grep -q "scan,sequence,score" "$output_file"
}

# Test 7: Script extracts version from versions.log correctly
@test "integration: run.sh uses version from versions.log in output path" {
    bash run.sh sample_data/test_dataset mock_algo
    
    # Check that the version from versions.log is used in path
    [ -d "outputs/mock_algo/mock-1.0.0/test_dataset" ]
}

# Test 8: Script handles multiple versions correctly (uses latest)
@test "integration: run.sh uses latest version when multiple exist" {
    # Create versions.log with multiple versions
    cat > algorithms/mock_algo/versions.log <<EOF
- container_version: "mock-2.0.0"
  date: "2026-01-01"
  notes: "Latest version"
- container_version: "mock-1.0.0"
  date: "2025-10-06"
  notes: "Older version"
EOF
    
    bash run.sh sample_data/test_dataset mock_algo
    
    # Should use latest version (mock-2.0.0)
    [ -d "outputs/mock_algo/mock-2.0.0/test_dataset" ]
    [ ! -d "outputs/mock_algo/mock-1.0.0/test_dataset" ]
}

# Test 9: Script fails when versions.log is missing
@test "integration: run.sh fails gracefully when versions.log missing" {
    # Create algorithm without versions.log
    mkdir -p algorithms/no_version_algo
    touch algorithms/no_version_algo/container.sif
    
    run bash run.sh sample_data/test_dataset no_version_algo
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"Could not extract container_version"* ]]
}

# Test 10: Script validates algorithm exists
@test "integration: run.sh fails when algorithm doesn't exist" {
    run bash run.sh sample_data/test_dataset nonexistent_algo
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"not found in algorithms/"* ]]
}

# Test 11: Script rejects 'base' as algorithm name
@test "integration: run.sh rejects base algorithm" {
    # Create versions.log for base
    cat > algorithms/base/versions.log <<EOF
- container_version: "base-1.0.0"
  date: "2025-10-06"
EOF
    
    run bash run.sh sample_data/test_dataset base
    
    [ "$status" -eq 1 ]
    [[ "$output" == *"not an algorithm"* ]]
}

# Test 12: Script removes the overlay after exporting predictions
@test "integration: run.sh removes overlay after run" {
    run bash run.sh sample_data/test_dataset mock_algo

    [ "$status" -eq 0 ]
    [ -f "outputs/mock_algo/mock-1.0.0/test_dataset/output.csv" ]
    [ ! -f "algorithms/mock_algo/overlay_test_dataset.img" ]
}

# Test 13: Script handles -r flag with directory cleanup
@test "integration: run.sh -r removes old output directory" {
    # First run
    bash run.sh sample_data/test_dataset mock_algo
    
    # Create extra files in output directory
    output_dir="outputs/mock_algo/mock-1.0.0/test_dataset"
    touch "$output_dir/extra_file.txt"
    touch "$output_dir/old_time.log"
    
    # Run with -r
    bash run.sh -r sample_data/test_dataset mock_algo
    
    # Extra files should be gone
    [ ! -f "$output_dir/extra_file.txt" ]
    [ ! -f "$output_dir/old_time.log" ]
    # But new output files should exist
    [ -f "$output_dir/output.csv" ]
    [ -f "$output_dir/time.log" ]
}

# Test 14: Script processes MGF files correctly
@test "integration: run.sh finds and lists MGF files" {
    run bash run.sh sample_data/test_dataset mock_algo
    
    [ "$status" -eq 0 ]
    [[ "$output" == *"spectrum1.mgf"* ]]
    [[ "$output" == *"spectrum2.mgf"* ]]
}
# Test 15: Script handles absolute paths for datasets
@test "integration: run.sh works with absolute dataset paths" {
    abs_path="$TEST_DIR/sample_data/test_dataset"
    
    run bash run.sh "$abs_path" mock_algo
    [ "$status" -eq 0 ]
    [ -d "outputs/mock_algo/mock-1.0.0/test_dataset" ]
}

# Test 16: Script works with mock apptainer from setup_mock.sh
@test "integration: mock apptainer from setup_mock.sh is used" {
    apptainer_path=$(which apptainer)
    [[ "$apptainer_path" == "$MOCK_ENV_DIR/bin/apptainer" ]]
}

# Test 17: Script recalculates with -r placed after positional arguments
@test "integration: run.sh recalculates output with -r after arguments" {
    bash run.sh sample_data/test_dataset mock_algo

    output_file="outputs/mock_algo/mock-1.0.0/test_dataset/output.csv"
    echo "MARKER_LINE" >> "$output_file"

    run bash run.sh sample_data/test_dataset mock_algo -r

    [ "$status" -eq 0 ]
    [[ "$output" != *"Skipping"* ]]
    ! grep -q "MARKER_LINE" "$output_file"
}

# Test 18: Script recalculates with -r placed between positional arguments
@test "integration: run.sh recalculates output with -r between arguments" {
    bash run.sh sample_data/test_dataset mock_algo

    output_file="outputs/mock_algo/mock-1.0.0/test_dataset/output.csv"
    echo "MARKER_LINE" >> "$output_file"

    run bash run.sh sample_data/test_dataset -r mock_algo

    [ "$status" -eq 0 ]
    [[ "$output" != *"Skipping"* ]]
    ! grep -q "MARKER_LINE" "$output_file"
}

# Test 19: Script runs evaluation by default
@test "integration: run.sh evaluates predictions by default" {
    run bash run.sh sample_data/test_dataset mock_algo

    [ "$status" -eq 0 ]
    [[ "$output" == *"EVALUATE PREDICTIONS"* ]]
}

# Test 20: Script skips evaluation with --no-eval
@test "integration: run.sh skips evaluation with --no-eval" {
    run bash run.sh sample_data/test_dataset mock_algo --no-eval

    [ "$status" -eq 0 ]
    [[ "$output" == *"Evaluate predictions: false"* ]]
    [[ "$output" != *"EVALUATE PREDICTIONS"* ]]
}

# Test 21: Script removes a leftover overlay when the output already exists
@test "integration: run.sh removes leftover overlay when skipping" {
    bash run.sh sample_data/test_dataset mock_algo
    touch "algorithms/mock_algo/overlay_test_dataset.img"

    run bash run.sh sample_data/test_dataset mock_algo

    [ "$status" -eq 0 ]
    [[ "$output" == *"Skipping running algorithm"* ]]
    [ ! -f "algorithms/mock_algo/overlay_test_dataset.img" ]
}

# Test 22: Script keeps the overlay for debugging when the algorithm produced no output
@test "integration: run.sh keeps overlay when algorithm fails" {
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
echo "mock failure" >&2
exit 1
SCRIPT

    run bash run.sh sample_data/test_dataset mock_algo

    [ ! -f "outputs/mock_algo/mock-1.0.0/test_dataset/output.csv" ]
    [ -f "algorithms/mock_algo/overlay_test_dataset.img" ]
}

# Test 23: Script shows tool output by default
@test "integration: run.sh shows tool output by default" {
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
echo "TOOL_NOISE"
echo "scan,sequence,score" > outputs.csv
echo "1,PEPTIDE,0.95" >> outputs.csv
SCRIPT

    run bash run.sh sample_data/test_dataset mock_algo

    [ "$status" -eq 0 ]
    [[ "$output" == *"TOOL_NOISE"* ]]
}

# Test 24: Script hides tool output with -q when the tool succeeds
@test "integration: run.sh -q hides tool output on success" {
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
echo "TOOL_NOISE"
echo "TOOL_STDERR_NOISE" >&2
echo "scan,sequence,score" > outputs.csv
echo "1,PEPTIDE,0.95" >> outputs.csv
SCRIPT

    run bash run.sh sample_data/test_dataset mock_algo -q

    [ "$status" -eq 0 ]
    [[ "$output" != *"TOOL_NOISE"* ]]
    [[ "$output" != *"TOOL_STDERR_NOISE"* ]]
    [ -f "outputs/mock_algo/mock-1.0.0/test_dataset/output.csv" ]
    [ ! -f "outputs/mock_algo/mock-1.0.0/test_dataset/predict.log" ]
    grep -q "^real" "outputs/mock_algo/mock-1.0.0/test_dataset/time.log"
}

# Test 25: Script shows the end of tool output with -q when the tool fails and keeps the full log
@test "integration: run.sh -q shows tool output tail on failure and keeps predict.log" {
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
echo "FIRST_LINE"
for i in $(seq 1 100); do echo "line $i"; done
echo "TOOL_ERROR_MESSAGE" >&2
exit 3
SCRIPT

    run bash run.sh sample_data/test_dataset mock_algo -q

    log_file="outputs/mock_algo/mock-1.0.0/test_dataset/predict.log"
    [[ "$output" == *"Algorithm mock_algo failed (exit 3)"* ]]
    [[ "$output" == *"TOOL_ERROR_MESSAGE"* ]]
    [[ "$output" != *"FIRST_LINE"* ]]
    [[ "$output" == *"Full tool output: ./$log_file"* ]]
    grep -q "FIRST_LINE" "$log_file"
    grep -q "TOOL_ERROR_MESSAGE" "$log_file"
}

# Test 26: Script passes --quiet to evaluation with -q
@test "integration: run.sh -q passes --quiet to evaluation" {
    # Wrap the mock apptainer to record its arguments
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"

    run bash run.sh sample_data/test_dataset mock_algo -q

    [ "$status" -eq 0 ]
    grep -q "evaluation.evaluate .* --quiet" "$TEST_DIR/apptainer_calls.log"
}

# Test 27: Script evaluates only the algorithm version it ran
@test "integration: run.sh passes --algorithms with its algorithm version to evaluation" {
    # Wrap the mock apptainer to record its arguments
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"

    run bash run.sh sample_data/test_dataset mock_algo

    [ "$status" -eq 0 ]
    grep -q "evaluation.evaluate .* --algorithms mock_algo:mock-1.0.0" "$TEST_DIR/apptainer_calls.log"
}

# Test 28: run_split.sh runs each part and merges outputs into the output layout
@test "integration: run_split.sh runs each part and merges outputs" {
    cp "$ORIG_DIR/run_split.sh" .

    run bash run_split.sh sample_data/test_dataset mock_algo 2

    output_dir="outputs/mock_algo/mock-1.0.0/test_dataset"
    [ "$status" -eq 0 ]
    [[ "$output" == *"for test_dataset part 0:"* ]]
    [[ "$output" == *"for test_dataset part 1:"* ]]
    # one header + 2 rows from each of the 2 parts
    [ "$(wc -l < "$output_dir/output.csv")" -eq 5 ]
    [ "$(grep -c "scan,sequence,score" "$output_dir/output.csv")" -eq 1 ]
    [ "$(grep -c "^real" "$output_dir/time.log")" -eq 1 ]
    [ ! -e "outputs/test_dataset_part_0" ]
    [ ! -e "times/test_dataset_part_0" ]
    [ ! -f "algorithms/mock_algo/overlay_test_dataset.img" ]
}

# Test 29: run_split.sh with more parts than files
@test "integration: run_split.sh with more parts than files" {
    cp "$ORIG_DIR/run_split.sh" .

    run bash run_split.sh sample_data/test_dataset mock_algo 5

    [ "$status" -eq 0 ]
    [[ "$output" != *"part 2:"* ]]
    [[ "$output" != *"No such file"* ]]
    [[ "$output" != *"syntax error"* ]]
    [ "$(wc -l < outputs/mock_algo/mock-1.0.0/test_dataset/output.csv)" -eq 5 ]
}

# Test 30: run_split.sh does not merge after a failed part; a rerun only runs missing parts
@test "integration: run_split.sh does not merge after a failed part and reruns only missing parts" {
    cp "$ORIG_DIR/run_split.sh" .
    # First call succeeds, second call fails
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
if [ -e part_done ]; then exit 1; fi
touch part_done
echo "scan,sequence,score" > outputs.csv
echo "1,PEPTIDE,0.95" >> outputs.csv
SCRIPT

    run bash run_split.sh sample_data/test_dataset mock_algo 2

    output_dir="outputs/mock_algo/mock-1.0.0/test_dataset"
    [ "$status" -eq 1 ]
    [[ "$output" == *"part 1 produced no output, not merging"* ]]
    [ ! -e "$output_dir/output.csv" ]
    [ -e "outputs/test_dataset_part_0/mock_algo_output.csv" ]
    [ -f "algorithms/mock_algo/overlay_test_dataset.img" ]

    # Rerun with a working algorithm: part 0 is skipped, part 1 runs, outputs are merged
    rm algorithms/mock_algo/part_done
    run bash run_split.sh sample_data/test_dataset mock_algo 2

    [ "$status" -eq 0 ]
    [[ "$output" == *"Skipping test_dataset part 0"* ]]
    [[ "$output" == *"for test_dataset part 1:"* ]]
    [ "$(wc -l < "$output_dir/output.csv")" -eq 3 ]
}

# Test 31: run_split.sh evaluates only its algorithm version
@test "integration: run_split.sh passes --algorithms with its algorithm version to evaluation" {
    cp "$ORIG_DIR/run_split.sh" .
    # Wrap the mock apptainer to record its arguments
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"

    run bash run_split.sh sample_data/test_dataset mock_algo 2

    [ "$status" -eq 0 ]
    grep -q "evaluation.augment_predictions --output_dir ./outputs/mock_algo/mock-1.0.0/test_dataset" "$TEST_DIR/apptainer_calls.log"
    grep -q "evaluation.evaluate .* --algorithms mock_algo:mock-1.0.0" "$TEST_DIR/apptainer_calls.log"
}

# Test 32: run_split.sh rejects unknown algorithms and 'base'
@test "integration: run_split.sh rejects unknown algorithm and base" {
    cp "$ORIG_DIR/run_split.sh" .

    run bash run_split.sh sample_data/test_dataset no_such_algo 2
    [ "$status" -eq 1 ]
    [[ "$output" == *"'no_such_algo' not found"* ]]

    run bash run_split.sh sample_data/test_dataset base 2
    [ "$status" -eq 1 ]
    [[ "$output" == *"not an algorithm"* ]]
}

# Test 33: run.sh and run_split.sh mount the dataset dir in augmentation and evaluation containers
@test "integration: augmentation and evaluation mount the dataset dir" {
    cp "$ORIG_DIR/run_split.sh" .
    # Wrap the mock apptainer to record its arguments
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"
    abs_dset_dir="$(realpath sample_data/test_dataset)"

    for script in "run.sh" "run_split.sh"; do
        rm -rf outputs "$TEST_DIR/apptainer_calls.log"
        if [ "$script" = "run.sh" ]; then
            run bash run.sh sample_data/test_dataset mock_algo
        else
            run bash run_split.sh sample_data/test_dataset mock_algo 2
        fi
        [ "$status" -eq 0 ]
        [ "$(grep -c -- "-B $abs_dset_dir evaluation.sif" "$TEST_DIR/apptainer_calls.log")" -eq 2 ]
    done
}

# Helper for run_array.sh tests: working algorithms (output with SA and pred_RT columns, as after augmentation,
# which the mock apptainer does not run), a failing algorithm and one without augmentation columns;
# apptainer calls recorded
setup_run_array() {
    cp "$ORIG_DIR/run_array.sh" .
    cat > algorithms/mock_algo/make_predictions.sh <<'SCRIPT'
echo "spectrum_id,sequence,score,aa_scores,SA,pred_RT" > outputs.csv
echo 'test_1:0,PEPTIDE,0.95,"0.95,0.95",0.8,10.0' >> outputs.csv
SCRIPT
    cp -r algorithms/mock_algo algorithms/noaug_algo
    printf 'echo "spectrum_id,sequence,score,aa_scores" > outputs.csv\necho "test_1:0,PEPTIDE,0.95,0.95" >> outputs.csv\n' \
        > algorithms/noaug_algo/make_predictions.sh
    cp -r algorithms/mock_algo algorithms/mock_algo2
    sed -i 's/mock-1.0.0/mock2-1.0.0/' algorithms/mock_algo2/versions.log
    cp -r algorithms/mock_algo algorithms/fail_algo
    printf 'echo "tool crashed" >&2\nexit 1\n' > algorithms/fail_algo/make_predictions.sh
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"
}

# Test 34: run_array.sh reports per dataset/algorithm status and evaluates each dataset once
@test "integration: run_array.sh runs all pairs, reports status, evaluates once per dataset" {
    setup_run_array

    run bash run_array.sh -d sample_data/test_dataset -a mock_algo -a mock_algo2 -a fail_algo -a noaug_algo -a no_such_algo

    [ "$status" -eq 1 ]
    summary=$(ls logs/run_array_*/summary.tsv)
    grep -q "^test_dataset	mock_algo	OK	new output	" "$summary"
    grep -q "^test_dataset	mock_algo2	OK	new output	" "$summary"
    grep -q "^test_dataset	fail_algo	FAILED	algorithm failed	" "$summary"
    grep -q "^test_dataset	noaug_algo	FAILED	augmentation failed	" "$summary"
    grep -q "^test_dataset	no_such_algo	FAILED	unknown algorithm	-$" "$summary"
    grep -q "^test_dataset	(evaluation)	OK	evaluated: mock_algo:mock-1.0.0 mock_algo2:mock2-1.0.0	" "$summary"
    # run.sh is called without evaluation, evaluation runs once for the dataset
    [ "$(grep -c "evaluation.evaluate" "$TEST_DIR/apptainer_calls.log")" -eq 1 ]
    grep -q "evaluation.evaluate ./outputs/ sample_data/test_dataset --algorithms mock_algo:mock-1.0.0 mock_algo2:mock2-1.0.0" "$TEST_DIR/apptainer_calls.log"
    # per pair logs
    grep -q "tool crashed" logs/run_array_*/test_dataset__fail_algo.log
    [[ "$output" == *"SUMMARY"* ]]
}

# Test 35: run_array.sh reports reused outputs on a rerun, -q is passed on
@test "integration: run_array.sh reuses existing output and passes -q" {
    setup_run_array
    bash run_array.sh -d sample_data/test_dataset -a mock_algo
    rm -rf logs "$TEST_DIR/apptainer_calls.log"

    run bash run_array.sh -q -d sample_data/test_dataset -a mock_algo

    [ "$status" -eq 0 ]
    grep -q "^test_dataset	mock_algo	OK	existing output reused	" logs/run_array_*/summary.tsv
    grep -q "Quiet mode (show tool output only on failure): true" logs/run_array_*/test_dataset__mock_algo.log
    grep -q "evaluation.evaluate .* --quiet" "$TEST_DIR/apptainer_calls.log"
}

# Test 36: run_array.sh requires datasets, defaults to algorithms with a container.def
@test "integration: run_array.sh requires datasets and defaults to algorithms with container.def" {
    setup_run_array
    run bash run_array.sh -a mock_algo
    [ "$status" -eq 1 ]
    [[ "$output" == *"at least one dataset is required"* ]]

    touch algorithms/mock_algo/container.def algorithms/mock_algo2/container.def
    run bash run_array.sh -d sample_data/test_dataset
    [ "$status" -eq 0 ]
    [[ "$output" == *"Algorithms: mock_algo mock_algo2"* ]]
}

# Test 37: run_test.sh runs the algorithm on the sample dataset and validates the output format
@test "integration: run_test.sh runs on sample dataset with .env and reports validation" {
    cp "$ORIG_DIR/run_test.sh" .
    mkdir -p sample_data/9_species_human/mgf
    cp sample_data/test_dataset/mgf/spectrum1.mgf sample_data/9_species_human/mgf/
    # Wrap the mock apptainer to record its arguments
    cat > "$TEST_DIR/bin/apptainer" <<SCRIPT
#!/bin/bash
echo "\$@" >> "$TEST_DIR/apptainer_calls.log"
exec "$MOCK_ENV_DIR/bin/apptainer" "\$@"
SCRIPT
    chmod +x "$TEST_DIR/bin/apptainer"

    run bash run_test.sh mock_algo

    [ "$status" -eq 0 ]
    [[ "$output" == *"OUTPUT FORMAT VALIDATED."* ]]
    grep -q -- "-B sample_data/9_species_human/mgf:/algo/9_species_human --env-file .env .*make_predictions.sh 9_species_human" "$TEST_DIR/apptainer_calls.log"
    grep -q "python test_output_format.py" "$TEST_DIR/apptainer_calls.log"
    [ ! -e "algorithms/mock_algo/test_overlay.img" ]
    [ ! -e "test_outputs" ]
}

# Test 38: run.sh skips an algorithm without container, unless its output already exists
@test "integration: run.sh skips an algorithm without container" {
    output_dir="outputs/mock_algo/mock-1.0.0/test_dataset"
    rm algorithms/mock_algo/container.sif

    run bash run.sh sample_data/test_dataset mock_algo
    [ "$status" -eq 1 ]
    [[ "$output" == *"Skipping mock_algo: container algorithms/mock_algo/container.sif not found."* ]]
    [[ "$output" != *"RUN ALGORITHM"* ]]
    [ ! -e "algorithms/mock_algo/overlay_test_dataset.img" ]

    # Existing output: no container needed (augmentation and evaluation only)
    mkdir -p "$output_dir"
    echo "spectrum_id,sequence,score" > "$output_dir/output.csv"
    run bash run.sh sample_data/test_dataset mock_algo
    [ "$status" -eq 0 ]
    [[ "$output" == *"Skipping running algorithm"* ]]

    # Recalculation needs the container: existing output is kept
    run bash run.sh -r sample_data/test_dataset mock_algo
    [ "$status" -eq 1 ]
    [[ "$output" == *"container algorithms/mock_algo/container.sif not found"* ]]
    [ -e "$output_dir/output.csv" ]
}

# Test 39: run_split.sh skips an algorithm without container
@test "integration: run_split.sh skips an algorithm without container" {
    cp "$ORIG_DIR/run_split.sh" .
    rm algorithms/mock_algo/container.sif

    run bash run_split.sh sample_data/test_dataset mock_algo 2

    [ "$status" -eq 1 ]
    [[ "$output" == *"Skipping mock_algo: container algorithms/mock_algo/container.sif not found."* ]]
    [ ! -e "outputs/test_dataset_part_0" ]
}

# Test 40: run_array.sh reports algorithms without container as SKIPPED (not a failure)
@test "integration: run_array.sh reports algorithms without container as skipped" {
    setup_run_array
    rm algorithms/mock_algo2/container.sif

    run bash run_array.sh -d sample_data/test_dataset -a mock_algo -a mock_algo2

    [ "$status" -eq 0 ]
    summary=$(ls logs/run_array_*/summary.tsv)
    grep -q "^test_dataset	mock_algo	OK	new output	" "$summary"
    grep -q "^test_dataset	mock_algo2	SKIPPED	no container	-$" "$summary"
    grep -q "^test_dataset	(evaluation)	OK	evaluated: mock_algo:mock-1.0.0	" "$summary"
}
