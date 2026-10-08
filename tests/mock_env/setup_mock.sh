#!/bin/bash
# Setup mock environment for testing without containers
# Replaces apptainer with a mock script that executes commands directly
# Usage: source tests/mock_env/setup_mock.sh
# Restore: source tests/mock_env/teardown_mock.sh

if [ -z "$BASH_SOURCE" ]; then
    echo "Error: This script must be sourced, not executed" >&2
    return 1
fi

export MOCK_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Save original PATH before modifying
export ORIGINAL_PATH="$PATH"

# Create mock apptainer command
mkdir -p "$MOCK_ENV_DIR/bin" || {
    echo "Error: Failed to create mock bin directory" >&2
    return 1
}

cat > "$MOCK_ENV_DIR/bin/apptainer" <<'EOF'
#!/bin/bash
# Mock apptainer - simulates container environment for testing

if [[ "$*" == *"overlay create"* ]]; then
    # Extract .img filename from arguments and create empty file
    overlay_file=$(echo "$@" | grep -oE '[^ ]+\.img')
    if [ -n "$overlay_file" ]; then
        touch "$overlay_file"
        exit 0
    fi
    exit 1
    
elif [[ "$*" == *"exec"* ]]; then
    # Extract algorithm directory from container.sif path
    algo_dir=""
    if [[ "$*" =~ algorithms/([^/]+)/container\.sif ]]; then
        algo_name="${BASH_REMATCH[1]}"
        algo_dir="algorithms/${algo_name}"
    fi
    
    # Extract command from bash -c "..."
    if [[ "$*" =~ bash[[:space:]]+-c[[:space:]]+\"(.*)\" ]]; then
        cmd="${BASH_REMATCH[1]}"
    elif [[ "$*" =~ bash[[:space:]]+-c[[:space:]]+(.+)$ ]]; then
        cmd="${BASH_REMATCH[1]}"
        # Remove trailing quote if present
        cmd="${cmd%\"}"
    else
        exit 0
    fi
    
    # Handle two types of commands:
    # 1. "cd /algo && ./make_predictions.sh dataset" - run algorithm
    # 2. "cp /algo/outputs.csv /algo/outputs/output.csv" - copy output
    
    if [[ "$cmd" == *"make_predictions.sh"* ]]; then
        # Execute make_predictions.sh from algorithm directory
        if [ -n "$algo_dir" ] && [ -d "$algo_dir" ]; then
            (
                cd "$algo_dir" || exit 1
                # Extract just the make_predictions.sh command
                pred_cmd="${cmd#*&&}"
                pred_cmd="${pred_cmd#"${pred_cmd%%[![:space:]]*}"}"  # trim leading space
                eval "$pred_cmd"
            )
            exit $?
        fi
    elif [[ "$cmd" == *"cp /algo/outputs.csv"* ]]; then
        # Move outputs.csv from algorithm dir to output dir
        # Extract output directory from -B flag in the command
        if [[ "$*" =~ -B[[:space:]]+\"?([^:\"]+)\"?:/algo/outputs ]]; then
            output_dir="${BASH_REMATCH[1]}"
        elif [[ "$*" =~ -B[[:space:]]+([^:[:space:]]+):/algo/outputs ]]; then
            output_dir="${BASH_REMATCH[1]}"
        fi
        
        if [ -n "$algo_dir" ] && [ -f "$algo_dir/outputs.csv" ] && [ -n "$output_dir" ]; then
            # Use the target file name of the cp command (e.g. output.csv)
            mv "$algo_dir/outputs.csv" "$output_dir/$(basename "${cmd##* }")"
            exit $?
        fi
    else
        # Other commands (like python evaluation) - just exit successfully
        exit 0
    fi
    
    # Fallback: if no bash -c pattern found, just exit successfully
    exit 0
fi

exit 1
EOF
chmod +x "$MOCK_ENV_DIR/bin/apptainer" || {
    echo "Error: Failed to create mock apptainer" >&2
    return 1
}

# Prepend mock bin to PATH so mock apptainer is used
export PATH="$MOCK_ENV_DIR/bin:$PATH"

echo "✓ Mock environment loaded"
echo "  - apptainer command mocked in $MOCK_ENV_DIR/bin"
echo "  - Original PATH saved in \$ORIGINAL_PATH"
echo "  - Restore with: source tests/mock_env/teardown_mock.sh"
