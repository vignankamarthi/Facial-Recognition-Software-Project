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
                # Add directory-specific imports to make the package more usable
                if directory == "src/facial_recognition_software":
                    f.write("# This file makes the facial_recognition_software directory a Python package.\n")
                    f.write("# Import the core classes for easier access\n")
                    f.write("from .face_detection import FaceDetector\n")
                    f.write("from .face_matching import FaceMatcher\n")
                    f.write("from .anonymization import FaceAnonymizer\n")
                    f.write("from .bias_testing import BiasAnalyzer\n")
                elif directory == "src/utilities":
                    f.write("# This file makes the utilities directory a Python package.\n")
                    f.write("# Import the core classes for easier access\n")
                    f.write("from .image_processing import ImageProcessor\n")
                else:
                    f.write(f"# This file makes the {os.path.basename(directory)} directory a Python package.\n")
        else:
            print(f"Already exists: {init_path}")

def create_directories():
    """Create all necessary data directories."""
    print_section("Creating data directories")

    directories = [
        "data",
        "data/known_faces",
        "data/test_images",
        "data/test_images/known",
        "data/test_images/unknown",
        "data/test_datasets",
        "data/test_datasets/demographic_split_set",
        "data/test_datasets/demographic_split_set/group_a",
        "data/test_datasets/demographic_split_set/group_b",
        "data/test_datasets/demographic_split_set/group_c",
        "data/test_datasets/results",
        "data/datasets",
        "data/datasets/lfw",
        "data/results",
    ]

    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Store the list of newly created directories for confirmation message
    created_dirs = []

    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            print(f"Creating: {dir_path}")
            os.makedirs(dir_path)
            created_dirs.append(directory)
        else:
            print(f"Already exists: {dir_path}")

    # Report what was actually created vs what already existed
    if created_dirs:
        print(f"\nCreated {len(created_dirs)} new directories:")
        for dir_name in created_dirs:
            print(f"  - {dir_name}")
    else:
        print("\nAll directories already exist - no new directories created")

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
    
    # List of dependencies to check
    dependencies = {
        "face_recognition": "face recognition library",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "matplotlib": "Matplotlib",
        "sklearn": "scikit-learn"
    }
    
    missing_deps = []
    
    # Check each dependency
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is not installed")
            missing_deps.append(module_name)
    
    # Handle missing dependencies
    if missing_deps:
        print("\nSome dependencies are missing. You can install them with:")
        print(f"pip install -r {requirements_file}")
        
        # Ask if the user wants to install dependencies now
        try:
            choice = input("\nDo you want to install missing dependencies now? (y/n): ")
            if choice.lower() == 'y':
                print("Installing missing dependencies...")
                run_command(f"pip install -r {requirements_file}")
                print("Dependency installation complete.")
            else:
                print("Skipping dependency installation.")
        except Exception:
            print("Skipping automatic dependency installation.")
    else:
        print("All core dependencies are installed.")

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
    deleted_count = 0
    
    for pattern in patterns:
        for path in glob.glob(os.path.join(base_dir, pattern), recursive=True):
            if os.path.isfile(path):
                print(f"Removing file: {path}")
                os.remove(path)
                deleted_count += 1
            elif os.path.isdir(path):
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
                deleted_count += 1
    
    if deleted_count == 0:
        print("No temporary files found to clean up.")
    else:
        print(f"Cleaned up {deleted_count} temporary files/directories.")

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
        # Add project directory to path to ensure imports work
        sys.path.insert(0, base_dir)
        
        # Create a new test environment to avoid import side effects
        import importlib
        
        modules_to_test = [
            "facial_recognition_software.face_detection",
            "facial_recognition_software.face_matching",
            "facial_recognition_software.anonymization",
            "facial_recognition_software.bias_testing"
        ]
        
        for module_name in modules_to_test:
            print(f"Importing {module_name}...")
            importlib.import_module(module_name)
        
        # Special case for image_processing due to relative import
        try:
            from utilities.image_processing import ImageProcessor
            print("Importing utilities.image_processing...OK")
        except ImportError as e:
            print(f"Error importing image_processing: {e}")
            
        print("All imports successful!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please run the fix_imports.py script again.")
        return False
    
    return True

def check_if_setup_already_done():
    """Check if setup has already been completed to avoid duplicate setup."""
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Check for key indicators that setup is already done
    indicators = [
        os.path.join(base_dir, "src", "__init__.py"),
        os.path.join(base_dir, "src", "facial_recognition_software", "__init__.py"),
        os.path.join(base_dir, "data", "known_faces"),
        os.path.join(base_dir, "data", "test_images"),
    ]

    exist_count = sum(1 for path in indicators if os.path.exists(path))

    # If most indicators exist, setup is likely already done
    if exist_count >= 3:
        return True

    return False

def main():
    """Main function to setup the project."""
    print_section("FACIAL RECOGNITION PROJECT QUICK SETUP")
    print("This script will prepare your project for demonstration.")
    
    # Check if setup appears to be already completed
    if check_if_setup_already_done():
        print("\nIt appears that project setup has already been completed.")
        try:
            choice = input("Do you want to run setup again? (y/n): ")
            if choice.lower() != 'y':
                print("Setup cancelled by user. Exiting.")
                return
            print("\nRunning setup again...")
        except Exception:
            # If we can't get user input, proceed with setup
            print("Running setup...")
    
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
    setup_success = run_quick_test()
    
    print_section("SETUP COMPLETE")
    if setup_success:
        print("Your project is now ready for demonstration!")
    else:
        print("Setup completed with some issues. Please check the errors above.")
    
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
