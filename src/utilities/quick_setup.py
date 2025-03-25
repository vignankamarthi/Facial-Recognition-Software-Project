"""
Quick Setup Script

This script performs a comprehensive setup of the facial recognition project,
preparing it for immediate demonstration.
"""

import os
import sys
import shutil
import subprocess

def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def run_command(command):
    """Run a shell command and print its output."""
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line.strip())
        
        for line in process.stderr:
            print(f"ERROR: {line.strip()}")
            
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def create_init_files():
    """Create __init__.py files in all necessary directories."""
    print_section("Creating __init__.py files")
    
    directories = [
        "src",
        "src/facial_recognition_software",
        "src/utilities"
    ]
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Change to the project root directory to ensure relative paths work
    os.chdir(base_dir)
    
    for directory in directories:
        init_path = os.path.join(base_dir, directory, "__init__.py")
        if not os.path.exists(init_path):
            print(f"Creating: {init_path}")
            with open(init_path, 'w') as f:
                f.write(f"# This file makes the {os.path.basename(directory)} directory a Python package.\n")
        else:
            print(f"Already exists: {init_path}")

def create_directories():
    """Create all necessary data directories."""
    print_section("Creating data directories")
    
    directories = [
        "data",
        "data/sample_faces",
        "data/test_images",
        "data/test_images/known",
        "data/test_images/unknown",
        "data/test_datasets",
        "data/test_datasets/sample_dataset",
        "data/test_datasets/sample_dataset/group_a",
        "data/test_datasets/sample_dataset/group_b",
        "data/test_datasets/sample_dataset/group_c",
        "data/test_datasets/results",
        "data/datasets",
        "data/datasets/lfw",
        "data/results"
    ]
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            print(f"Creating: {dir_path}")
            os.makedirs(dir_path)
        else:
            print(f"Already exists: {dir_path}")

def fix_imports():
    """Fix import statements in all Python files."""
    print_section("Fixing import statements")
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fix_imports_script = os.path.join(base_dir, "src/utilities/fix_imports.py")
    
    if os.path.exists(fix_imports_script):
        run_command(f"python {fix_imports_script}")
    else:
        print(f"Error: Could not find fix_imports.py script at {fix_imports_script}")

def verify_dependencies():
    """Verify that all dependencies are installed."""
    print_section("Verifying dependencies")
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    requirements_file = os.path.join(base_dir, "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print(f"Error: Could not find requirements.txt at {requirements_file}")
        return
    
    print("Checking installed dependencies...")
    try:
        import face_recognition
        import cv2
        import numpy
        import matplotlib
        import sklearn
        print("All core dependencies are installed.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print(f"Please run: pip install -r {requirements_file}")

def clean_up_temporary_files():
    """Clean up any unnecessary temporary files."""
    print_section("Cleaning up temporary files")
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Patterns to delete
    patterns = [
        "**/*.py.bak",
        "**/*.pyc",
        "**/__pycache__"
    ]
    
    import glob
    for pattern in patterns:
        for path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
            if os.path.isfile(path):
                print(f"Removing file: {path}")
                os.remove(path)
            elif os.path.isdir(path):
                print(f"Removing directory: {path}")
                shutil.rmtree(path)

def run_quick_test():
    """Run a quick test to verify the system works."""
    print_section("Running quick test")
    
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main_script = os.path.join(base_dir, "src/main.py")
    
    if not os.path.exists(main_script):
        print(f"Error: Could not find main.py at {main_script}")
        return
    
    print("Testing import statements...")
    try:
        sys.path.insert(0, base_dir)
        from facial_recognition_software.face_detection import FaceDetector
        from facial_recognition_software.face_matching import FaceMatcher
        from facial_recognition_software.anonymization import FaceAnonymizer
        from facial_recognition_software.bias_testing import BiasAnalyzer
        from .image_processing import ImageProcessor
        print("All imports successful!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run the fix_imports.py script again.")

def main():
    """Main function to setup the project."""
    print_section("FACIAL RECOGNITION PROJECT QUICK SETUP")
    print("This script will prepare your project for demonstration.")
    
    # Create __init__.py files
    create_init_files()
    
    # Create data directories
    create_directories()
    
    # Fix import statements
    fix_imports()
    
    # Verify dependencies
    verify_dependencies()
    
    # Clean up temporary files
    clean_up_temporary_files()
    
    # Run quick test
    run_quick_test()
    
    print_section("SETUP COMPLETE")
    print("Your project is now ready for demonstration!")
    print("To run the main application, use:")
    print("  python src/main.py")
    print("\nDemo commands:")
    print("  python src/main.py --detect     # Face detection demo")
    print("  python src/main.py --match      # Face matching demo")
    print("  python src/main.py --anonymize  # Face anonymization demo")
    print("  python src/main.py --bias       # Bias testing demo")
    print("  python src/main.py --setup-dataset  # Setup LFW dataset")

if __name__ == "__main__":
    main()
