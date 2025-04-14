#!/usr/bin/env python3
"""
Launcher Script for Facial Recognition Demo

This script fixes the Python import path issue and launches the main application.
"""

import os
import sys
import subprocess
import signal

# Add the project directory to Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

# Make sure all code uses the correct working directory
os.chdir(base_dir)

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
    
    process = None
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
        print("\nDemo interrupted by user. Terminating subprocess...")
        if process and process.poll() is None:
            # Try to terminate the child process gracefully
            if sys.platform != "win32":
                # On Unix-like systems, we can forward the SIGINT
                try:
                    process.send_signal(signal.SIGINT)
                    # Give it a moment to handle the signal
                    import time
                    time.sleep(0.5)
                except:
                    pass
            
            # If it's still running, force kill it
            if process.poll() is None:
                process.terminate()
                
            # Wait briefly for the process to exit
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If it doesn't terminate in time, kill it forcefully
                process.kill()
                
        print("Demo terminated.")
    except Exception as e:
        print(f"Error launching demo: {e}")
        if process and process.poll() is None:
            process.terminate()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Facial Recognition Demo Launcher.")
        sys.exit(0)
