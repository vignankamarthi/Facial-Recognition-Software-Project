#!/usr/bin/env python3
"""
Test runner script for the Facial Recognition Software Project.

This script provides a convenient way to run tests with the correct Python path settings.

Usage:
    python run_tests.py                      # Run all tests
    python run_tests.py test_file.py         # Run a specific test file
    python run_tests.py tests/unit/utils/    # Run all tests in a directory
    
Examples:
    python run_tests.py tests/unit/utils/test_common_utils.py
"""

import os
import sys
import pytest

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Run the tests with the correct path settings."""
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    print("Running tests...")
    
    # Create argument list for pytest
    args = ["-v"]  # Add verbosity
    
    # Add any additional arguments from the command line
    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    
    # Run pytest with the arguments
    exit_code = pytest.main(args)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
