#!/usr/bin/env python3
"""
Test runner for the nn-rag test suite.
Provides convenient commands to run different types of tests.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False


def run_unit_tests():
    """Run unit tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_block_extractor.py", "-v"]
    return run_command(cmd, "Unit Tests")


def run_validation_tests():
    """Run validation tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_validation.py", "-v"]
    return run_command(cmd, "Validation Tests")


def run_integration_tests():
    """Run integration tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_integration.py", "-v"]
    return run_command(cmd, "Integration Tests")


def run_performance_tests():
    """Run performance tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_performance.py", "-v", "-m", "performance"]
    return run_command(cmd, "Performance Tests")


def run_error_scenario_tests():
    """Run error scenario tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_error_scenarios.py", "-v"]
    return run_command(cmd, "Error Scenario Tests")


def run_cli_consistency_tests():
    """Run CLI consistency tests only."""
    cmd = [sys.executable, "-m", "pytest", "tests/rag/test_cli_consistency.py", "-v"]
    return run_command(cmd, "CLI Consistency Tests")


def run_all_tests():
    """Run all tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    return run_command(cmd, "All Tests")


def run_fast_tests():
    """Run fast tests only (exclude slow tests)."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "-m", "not slow"]
    return run_command(cmd, "Fast Tests Only")


def run_with_coverage():
    """Run tests with coverage reporting."""
    cmd = [
        sys.executable, "-m", "pytest", "tests/", 
        "--cov=ab.rag", 
        "--cov-report=html", 
        "--cov-report=term-missing",
        "-v"
    ]
    return run_command(cmd, "Tests with Coverage")


def lint_tests():
    """Run linting on test files."""
    test_files = list(Path("tests").rglob("test_*.py"))
    cmd = [sys.executable, "-m", "flake8"] + [str(f) for f in test_files]
    return run_command(cmd, "Linting Tests")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run nn-rag tests")
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "validation", "integration", "performance", 
            "error", "cli", "all", "fast", "coverage", "lint"
        ],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    print(f"Running tests from: {project_root}")
    print(f"Python executable: {sys.executable}")
    
    # Map test types to functions
    test_functions = {
        "unit": run_unit_tests,
        "validation": run_validation_tests,
        "integration": run_integration_tests,
        "performance": run_performance_tests,
        "error": run_error_scenario_tests,
        "cli": run_cli_consistency_tests,
        "all": run_all_tests,
        "fast": run_fast_tests,
        "coverage": run_with_coverage,
        "lint": lint_tests
    }
    
    # Run the selected tests
    success = test_functions[args.test_type]()
    
    if success:
        print(f"\n{args.test_type.title()} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n{args.test_type.title()} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
