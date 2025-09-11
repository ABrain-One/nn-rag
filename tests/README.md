# NN-RAG Test Suite

Comprehensive test suite for the Neural Network Retrieval-Augmented Generation (NN-RAG) system.

## Overview

This test suite provides comprehensive coverage of all aspects of the RAG system, including:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Scalability and resource usage testing
- **Error Scenario Tests**: Error handling and edge case testing
- **CLI Consistency Tests**: API and CLI behavior consistency testing

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── pytest.ini                 # Pytest settings
├── run_tests.py               # Test runner script
├── README.md                  # This file
└── rag/                       # RAG-specific tests
    ├── test_block_extractor.py    # BlockExtractor API tests
    ├── test_validation.py         # Block validation tests
    ├── test_cli_consistency.py    # CLI/API consistency tests
    ├── test_error_scenarios.py    # Error handling tests
    ├── test_integration.py        # Integration tests
    └── test_performance.py        # Performance tests
```

## Test Categories

### 1. Unit Tests (`test_block_extractor.py`)
Tests individual methods and components of the `BlockExtractor` class:

- **Initialization**: Constructor and configuration
- **Single Block Extraction**: `extract_single_block()` method
- **Multiple Block Extraction**: `extract_multiple_blocks()` method
- **Retry Failed Blocks**: `retry_failed_blocks()` method
- **Auto-Extract**: `auto_extract_all_blocks()` method
- **JSON Generation**: `_generate_block_names()` method
- **Parameter Validation**: Input validation and error handling

### 2. Validation Tests (`test_validation.py`)
Tests the block validation and file movement functionality:

- **Valid Block Validation**: Successful validation and file movement
- **Invalid Block Validation**: Syntax error handling
- **Missing File Handling**: Non-existent file error handling
- **Batch Validation**: Multiple block validation
- **Cleanup Operations**: Invalid block cleanup
- **File System Operations**: File movement and permissions

### 3. CLI Consistency Tests (`test_cli_consistency.py`)
Ensures CLI and API produce identical results:

- **Single Block**: CLI vs API consistency
- **Multiple Blocks**: CLI vs API consistency
- **Validation Control**: `--no-validate` vs `validate=False`
- **Cleanup Control**: `--cleanup-invalid` vs `cleanup_invalid=True`
- **Error Handling**: Consistent error responses
- **Output Format**: Identical result structures

### 4. Error Scenario Tests (`test_error_scenarios.py`)
Tests error handling and edge cases:

- **File System Errors**: Permission denied, disk full, file locked
- **Network Errors**: Connection failures, timeouts
- **Memory Errors**: Out of memory scenarios
- **Invalid Input**: Malformed data, invalid parameters
- **Resource Exhaustion**: Large files, many blocks
- **Concurrent Access**: Race conditions, file locking

### 5. Integration Tests (`test_integration.py`)
Tests complete end-to-end workflows:

- **Complete Auto-Extract**: JSON generation → indexing → extraction → validation
- **JSON Generation Workflow**: Missing JSON file handling
- **Validation Workflow**: File validation and movement
- **Error Recovery**: Partial failure handling
- **Cleanup Workflow**: Invalid block cleanup
- **Concurrent Operations**: Parallel processing
- **Large Scale**: Many blocks processing
- **System Resilience**: Mixed valid/invalid blocks

### 6. Performance Tests (`test_performance.py`)
Tests performance characteristics and scalability:

- **Single Block Performance**: Extraction and validation speed
- **Batch Performance**: Large batch processing
- **Memory Usage**: Memory consumption monitoring
- **Concurrent Performance**: Parallel processing efficiency
- **Large File Handling**: Large file processing
- **Scalability**: Performance with increasing block count
- **CPU Usage**: CPU utilization monitoring
- **Disk I/O**: File operation performance
- **Memory Leak Detection**: Long-running operation testing

## Running Tests

### Quick Start

```bash
# Run all tests
python tests/run_tests.py all

# Run only fast tests (exclude slow tests)
python tests/run_tests.py fast

# Run specific test categories
python tests/run_tests.py unit
python tests/run_tests.py integration
python tests/run_tests.py performance
```

### Using pytest directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/rag/test_block_extractor.py -v

# Run tests with specific markers
pytest tests/ -m "not slow" -v
pytest tests/ -m "performance" -v

# Run tests with coverage
pytest tests/ --cov=ab.rag --cov-report=html -v
```

### Test Categories

```bash
# Unit tests
python tests/run_tests.py unit

# Validation tests
python tests/run_tests.py validation

# Integration tests
python tests/run_tests.py integration

# Performance tests
python tests/run_tests.py performance

# Error scenario tests
python tests/run_tests.py error

# CLI consistency tests
python tests/run_tests.py cli

# All tests
python tests/run_tests.py all

# Fast tests only (exclude slow tests)
python tests/run_tests.py fast

# Tests with coverage
python tests/run_tests.py coverage

# Lint tests
python tests/run_tests.py lint
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)

- **Test Discovery**: Automatically finds test files
- **Output Options**: Verbose output, short tracebacks
- **Markers**: Categorize tests (unit, integration, performance, slow)
- **Warnings**: Filter out expected warnings
- **Timeout**: 300-second timeout for tests

### Test Markers

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.slow`: Slow tests (can be excluded)

### Fixtures

- `temp_dir`: Temporary directory for test files
- `extractor`: BlockExtractor instance for testing
- `validator`: BlockValidator instance for testing
- `sample_json_file`: Sample JSON file with test blocks
- `clean_environment`: Clean environment for each test

## Test Data

### Sample Block Data

The test suite includes sample block data for testing:

- **Valid Blocks**: Properly formatted Python classes
- **Invalid Blocks**: Syntax errors, missing imports
- **Large Blocks**: Performance testing
- **Unicode Blocks**: International character handling

### Mock Data

- **Repository Data**: Mock repository information
- **Extraction Results**: Mock extraction responses
- **Validation Results**: Mock validation responses

## Coverage

The test suite aims for comprehensive coverage:

- **Code Coverage**: All major code paths
- **Error Coverage**: All error conditions
- **Edge Case Coverage**: Boundary conditions
- **Integration Coverage**: End-to-end workflows
- **Performance Coverage**: Scalability testing

## Continuous Integration

### GitHub Actions (if applicable)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/run_tests.py all
```

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test function
pytest tests/rag/test_block_extractor.py::TestBlockExtractor::test_extract_single_block -v

# Run with debugging
pytest tests/rag/test_block_extractor.py -v -s --pdb

# Run with print statements
pytest tests/rag/test_block_extractor.py -v -s
```

### Test Output

- **Verbose Output**: `-v` flag shows individual test results
- **Short Tracebacks**: `--tb=short` for concise error messages
- **Durations**: Shows slowest 10 tests
- **Color Output**: Colored output for better readability

## Contributing

### Adding New Tests

1. **Follow Naming Convention**: `test_*.py` files, `test_*` functions
2. **Use Appropriate Markers**: Mark tests with `@pytest.mark.*`
3. **Use Fixtures**: Leverage existing fixtures for setup
4. **Mock External Dependencies**: Use `unittest.mock` for external calls
5. **Test Edge Cases**: Include boundary conditions and error scenarios

### Test Guidelines

- **Isolation**: Each test should be independent
- **Cleanup**: Clean up after each test
- **Mocking**: Mock external dependencies
- **Assertions**: Use specific assertions
- **Documentation**: Document complex test logic

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure project root is in Python path
2. **Permission Errors**: Check file permissions in temp directories
3. **Timeout Errors**: Increase timeout for slow tests
4. **Memory Errors**: Reduce test data size for memory-constrained environments

### Debug Commands

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Check installed packages
pip list

# Run with maximum verbosity
pytest tests/ -vvv

# Run specific test with debugging
pytest tests/rag/test_block_extractor.py::TestBlockExtractor::test_extract_single_block -vvv -s --pdb
```

## Performance Benchmarks

The performance tests establish baseline metrics:

- **Single Block Extraction**: < 1 second
- **Batch Extraction (100 blocks)**: < 5 seconds
- **Validation (50 blocks)**: < 10 seconds
- **Memory Usage**: < 500MB for 1000 blocks
- **CPU Usage**: < 90% peak usage
- **Disk I/O**: > 10 files/second

## Test Results

### Success Criteria

- All unit tests pass
- All integration tests pass
- Performance benchmarks met
- No memory leaks detected
- Error scenarios handled gracefully

### Reporting

- **Console Output**: Real-time test results
- **HTML Coverage**: Coverage report in `htmlcov/`
- **JUnit XML**: For CI/CD integration
- **Performance Metrics**: Timing and resource usage

This test suite ensures the reliability, performance, and correctness of the NN-RAG system across all use cases and scenarios.
