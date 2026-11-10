# Testing run_dataset.sh

This directory contains test suites for the `run_dataset.sh` script.

## Test Framework

We use [BATS (Bash Automated Testing System)](https://github.com/bats-core/bats-core) for testing bash scripts.

## Installation

### macOS
```bash
brew install bats-core
```

### Linux
```bash
git clone https://github.com/bats-core/bats-core.git
cd bats-core
sudo ./install.sh /usr/local
```

## Running Tests

```bash
# Run all tests
./tests/run_tests.sh

# Run unit tests only
bats tests/test_run_dataset.bats

# Run specific test
bats tests/test_run_dataset.bats --filter "script accepts -r flag"

# Run with tap output
bats --tap tests/test_run_dataset.bats
```

## Test Structure

- **`test_run_dataset.bats`** - Unit tests for individual components and logic
- **`test_run_dataset_integration.bats`** - End-to-end integration tests (require mocking)
- **`run_tests.sh`** - Convenient test runner that checks dependencies

## Test Coverage

### Unit Tests (test_run_dataset.bats)
- ✓ Script existence and executability
- ✓ Argument parsing (dataset directory, flags)
- ✓ Directory creation
- ✓ Cleanup logic with `-r` flag
- ✓ Algorithm directory filtering
- ✓ File path construction
- ✓ MGF file detection
- ✓ Error handling

### Integration Tests (test_run_dataset_integration.bats)
- Full script execution with mocked dependencies
- Output file creation
- Skip logic for existing outputs
- Recalculation with `-r` flag

## Writing New Tests

Each test follows this structure:

```bash
@test "description of what is being tested" {
    # Arrange - set up test conditions
    
    # Act - run the code
    run command_to_test
    
    # Assert - verify results
    [ "$status" -eq 0 ]
    [[ "$output" == *"expected"* ]]
}
```

## Best Practices

1. **Isolate tests** - Each test runs in a fresh environment
2. **Clean up** - Use setup/teardown to manage test state
3. **Mock external deps** - Don't rely on actual containers or external services
4. **Test edge cases** - Missing args, invalid inputs, permission errors
5. **Clear descriptions** - Test names should explain what they verify

## CI/CD Integration

Add to your CI pipeline:

```yaml
test:
  script:
    - brew install bats-core  # or apt-get install bats
    - ./tests/run_tests.sh
```

## Python unit tests for the evaluation

`test_evaluation_utils.py` covers the ProForma handling in `evaluation/utils.py` and `evaluation/metrics.py`: validation, token counting, and the conversion of predictions to delta mass notation. Run it from the repository root, with the evaluation dependencies from `evaluation.def` installed, plus `pytest`:

```bash
python -m pytest tests/test_evaluation_utils.py
```

pyteomics resolves Unimod accessions on first use, which needs network access.
