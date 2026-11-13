# Testing the run scripts

This directory contains test suites for the run scripts: `run.sh`, `run_split.sh`, `run_array.sh` and `run_test.sh`.

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
bats tests/test_run.bats

# Run specific test
bats tests/test_run_integration.bats --filter "run_array.sh"

# Run with tap output
bats --tap tests/test_run.bats
```

## Test Structure

- **`test_run.bats`** - Unit tests for individual components and logic
- **`test_run_integration.bats`** - End-to-end integration tests (require mocking)
- **`mock_env/`** - Mocked `apptainer` (runs `make_predictions.sh` of the algorithm locally, exports its output)
- **`run_tests.sh`** - Convenient test runner that checks dependencies

## Test Coverage

### Unit Tests (test_run.bats)
- ✓ Script existence and executability
- ✓ Argument parsing (dataset directory, algorithm, flags)
- ✓ Algorithm validation (unknown algorithm, `base`)
- ✓ Version extraction from `versions.log`
- ✓ Output paths (`outputs/<algorithm>/<version>/<dataset>/`)

### Integration Tests (test_run_integration.bats)
- `run.sh`: full execution, output and time log, skip logic for existing outputs, recalculation with `-r`, `--no-eval`, quiet mode `-q`, overlay cleanup, evaluation of only the run algorithm version
- `run_split.sh`: running and merging parts, more parts than files, failed parts (no merge, rerun of missing parts), 
  augmentation and evaluation, algorithm validation
- `run_array.sh`: status per dataset and algorithm (new or reused output, algorithm failed, augmentation failed, 
  unknown algorithm), one evaluation per dataset, `-q`, default algorithm list
- `run_test.sh`: run on the sample dataset with `.env`, output format validation, cleanup
- Datasets are mounted in augmentation and evaluation containers

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
