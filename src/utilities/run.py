#!/usr/bin/env python3
"""
Run script that ensures all modules are properly set up before running the main application.

This script automatically checks for required dependencies and applies patches if needed.
It provides a more robust startup process for the facial recognition system.
"""

import os
import sys
import importlib
from .face_recognition_patch import verify_face_recognition
from .common_utils import run_command

def ensure_package_installed(package_name, pip_install_cmd=None):
    """
    Ensure a Python package is installed, installing it if needed.
    
    Args:
        package_name (str): Name of the package to check
        pip_install_cmd (str, optional): Custom pip install command
        
    Returns:
        bool: True if package is available after check/install
    """
    if pip_install_cmd is None:
        pip_install_cmd = f"pip install {package_name}"
        
    try:
        # Try to import the package
        importlib.import_module(package_name)
        print(f"{package_name} package is installed correctly.")
        return True
    except ImportError:
        print(f"{package_name} not found. Installing...")
        success = run_command(pip_install_cmd)
        
        if success:
            try:
                # Try to import again after install
                importlib.import_module(package_name)
                print(f"Successfully installed {package_name}.")
                return True
            except ImportError:
                print(f"Failed to import {package_name} after installation.")
        
        print(f"Failed to install {package_name}. Please install manually:")
        print(f"  {pip_install_cmd}")
        return False

def main():
    """Set up environment and run the main application."""
    print("=" * 60)
    print("FACIAL RECOGNITION SYSTEM - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check for required packages
    packages = [
        ("face_recognition_models", "pip install git+https://github.com/ageitgey/face_recognition_models"),
        ("face_recognition", None),
        ("opencv-python", None),
        ("numpy", None),
        ("matplotlib", None)
    ]
    
    # Track if any package failed to install
    install_failed = False
    
    for package, cmd in packages:
        if not ensure_package_installed(package, cmd):
            install_failed = True
    
    if install_failed:
        print("\nSome required packages could not be installed.")
        print("Please install them manually and try again.")
        sys.exit(1)
    
    # Verify and patch face_recognition if needed
    if not verify_face_recognition():
        print("Warning: Face recognition verification failed.")
        print("Some features may not work correctly.")
    
    # Now run the main application
    print("\nStarting the Facial Recognition System...\n")
    try:
        # Try to import main from the correct path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import main
    except ImportError:
        print("Error: Could not import main application module.")
        print("Make sure you are running this script from the project root directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
