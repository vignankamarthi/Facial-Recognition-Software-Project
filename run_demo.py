#!/usr/bin/env python3
"""
Launcher Script for Facial Recognition Demo

This script fixes the Python import path issue and launches the main application.
"""

import os
import sys
import subprocess

# Add the project directory to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

def main():
    """Launch the main application with proper path settings."""
    print("=" * 60)
    print("Facial Recognition Demo Launcher")
    print("=" * 60)
    
    # Check if src/main.py exists
    main_script = os.path.join(base_dir, "src", "main.py")
    if not os.path.exists(main_script):
        print(f"Error: Main script not found at {main_script}")
        return
    
    # Pass all command line arguments to the main script
    cmd = [sys.executable, main_script] + sys.argv[1:]
    
    try:
        # Run the script with the project directory in Python path
        env = os.environ.copy()
        env["PYTHONPATH"] = base_dir + os.pathsep + env.get("PYTHONPATH", "")
        
        # Print the command that's being executed
        print(f"Running: {' '.join(cmd)}")
        print("-" * 60)
        
        # Execute the command
        process = subprocess.Popen(cmd, env=env)
        process.wait()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"Error launching demo: {e}")

if __name__ == "__main__":
    main()
