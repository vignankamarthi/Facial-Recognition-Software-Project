#!/usr/bin/env python3
"""
Test runner script for the Facial Recognition Software Project.

This script provides a convenient way to run tests with the correct Python path settings.
It automatically detects the environment and configures tests appropriately for hardware independence.

Usage:
    python run_tests.py                      # Run all tests
    python run_tests.py test_file.py         # Run a specific test file
    python run_tests.py tests/unit/utils/    # Run all tests in a directory
    
Options:
    --headless              Force headless mode (no window display)
    --with-webcam           Force webcam availability
    --no-webcam             Force webcam unavailability
    --ci                    Simulate CI environment
    
Examples:
    python run_tests.py tests/unit/utils/test_common_utils.py
    python run_tests.py --headless tests/functional/
"""

import os
import sys
import pytest
import argparse

# Ensure the project root is in the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    """Run the tests with the correct path settings."""
    # Create an argument parser for our custom options
    parser = argparse.ArgumentParser(description="Run tests for Facial Recognition Software Project")
    parser.add_argument("--headless", action="store_true", help="Force headless mode (no window display)")
    parser.add_argument("--with-webcam", action="store_true", help="Force webcam availability")
    parser.add_argument("--no-webcam", action="store_true", help="Force webcam unavailability")
    parser.add_argument("--ci", action="store_true", help="Simulate CI environment")
    
    # Parse our arguments, but leave the rest for pytest
    our_args, pytest_args = parser.parse_known_args()
    
    # Set environment variables based on arguments
    if our_args.headless:
        os.environ["FORCE_HEADLESS"] = "true"
        print("Running in forced headless mode...")
        
    if our_args.with_webcam:
        os.environ["FORCE_WEBCAM_AVAILABLE"] = "true"
        print("Forcing webcam availability...")
        
    if our_args.no_webcam:
        os.environ["FORCE_WEBCAM_AVAILABLE"] = "false"
        print("Forcing webcam unavailability...")
        
    if our_args.ci:
        os.environ["GITHUB_ACTIONS"] = "true"
        print("Simulating CI environment...")
    
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    
    # Try to import and use our environment detection
    try:
        sys.path.insert(0, project_root)
        from src.utils.environment_utils import get_environment_info
        
        # Print detected environment
        env_info = get_environment_info()
        print("\nDetected environment:")
        print(f"  CI: {env_info['ci']}")
        print(f"  Headless: {env_info['headless']}")
        print(f"  Webcam available: {env_info['webcam_available']}")
        print(f"  Platform: {env_info['platform']}")
        print(f"  Python: {env_info['python_version']}")
        print(f"  OpenCV: {env_info['opencv_version']}")
    except ImportError:
        print("Could not import environment detection utilities.")
    
    print("\nRunning tests...")
    
    # Create argument list for pytest
    args = ["-v"]  # Add verbosity
    
    # Add any additional arguments from the command line
    args.extend(pytest_args)
    
    # Run pytest with the arguments
    exit_code = pytest.main(args)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
