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

# Test 12: Script creates overlay files
@test "integration: run.sh creates overlay files" {
    bash run.sh sample_data/test_dataset mock_algo
    
    # Overlay should be created during execution
    [ -f "algorithms/mock_algo/overlay_test_dataset.img" ]
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
