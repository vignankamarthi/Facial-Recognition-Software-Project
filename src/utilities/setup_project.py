"""
Project Setup Script

This script sets up the proper directory structure for the facial recognition project
and ensures that all paths referenced in the project files are correct.
"""

import os
import shutil
import sys

# Project base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Project base directory: {base_dir}")

def create_project_directories():
    """Create all necessary directories for the project."""
    directories = [
        # Main data directories
        os.path.join(base_dir, "data"),
        
        # Face reference directories
        os.path.join(base_dir, "data/sample_faces"),
        
        # Test image directories
        os.path.join(base_dir, "data/test_images"),
        os.path.join(base_dir, "data/test_images/known"),
        os.path.join(base_dir, "data/test_images/unknown"),
        
        # Dataset directories
        os.path.join(base_dir, "data/datasets"),
        os.path.join(base_dir, "data/datasets/lfw"),
        
        # Bias testing directories
        os.path.join(base_dir, "data/test_datasets"),
        os.path.join(base_dir, "data/test_datasets/sample_dataset"),
        os.path.join(base_dir, "data/test_datasets/sample_dataset/group_a"),
        os.path.join(base_dir, "data/test_datasets/sample_dataset/group_b"),
        os.path.join(base_dir, "data/test_datasets/sample_dataset/group_c"),
        os.path.join(base_dir, "data/test_datasets/results"),
        
        # Results directories
        os.path.join(base_dir, "data/results"),
        
        # Docs directory
        os.path.join(base_dir, "docs"),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
        else:
            print(f"Directory already exists: {directory}")

def fix_lfw_dataset_structure():
    """
    Check if LFW dataset exists outside the project directory and move it
    to the correct location if necessary.
    """
    # Possible locations for LFW dataset based on the screenshot
    possible_locations = [
        "/Users/vkamarthi24/Desktop/Personal Projects/data/datasets/lfw",
    ]
    
    target_lfw_dir = os.path.join(base_dir, "data/datasets/lfw")
    
    # Check each possible location
    for location in possible_locations:
        if os.path.exists(location) and location != target_lfw_dir:
            print(f"Found LFW dataset at: {location}")
            
            # Check if the target directory already has content
            if os.path.exists(target_lfw_dir) and any(os.listdir(target_lfw_dir)):
                print(f"Target directory already has content: {target_lfw_dir}")
                user_input = input("Do you want to replace it? (y/n): ")
                if user_input.lower() != 'y':
                    print("Skipping dataset move.")
                    continue
            
            # Move the dataset to the correct location
            print(f"Moving LFW dataset to: {target_lfw_dir}")
            try:
                # If the target directory exists, remove it first
                if os.path.exists(target_lfw_dir):
                    shutil.rmtree(target_lfw_dir)
                
                # Move the directory
                shutil.move(location, target_lfw_dir)
                print("Dataset moved successfully!")
                break
            except Exception as e:
                print(f"Error moving dataset: {e}")
                print("Please manually move the dataset to the correct location.")

def cleanup_sample_directories():
    """Remove any temporary or sample directories that are not needed."""
    # Directories to remove (adjust as needed)
    dirs_to_remove = [
        os.path.join(base_dir, "../data/datasets/lfw/lfw_sample"),
    ]
    
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            print(f"Removing directory: {dir_path}")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")

def cleanup_debug_files():
    """Remove any debug or temporary files."""
    # Files to remove (adjust as needed)
    files_to_remove = [
        os.path.join(base_dir, "src/download_debug.py"),
    ]
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing {file_path}: {e}")

def main():
    """Main function to set up the project."""
    print("Setting up the facial recognition project...")
    
    # Create all necessary directories
    create_project_directories()
    
    # Fix LFW dataset structure if needed
    fix_lfw_dataset_structure()
    
    # Clean up sample directories
    cleanup_sample_directories()
    
    # Clean up debug files
    cleanup_debug_files()
    
    print("\nProject setup completed successfully!")
    print("\nREMINDER: Please use the following paths in your code:")
    print(f"- Base project directory: {base_dir}")
    print(f"- Sample faces directory: {os.path.join(base_dir, 'data/sample_faces')}")
    print(f"- Test images directory: {os.path.join(base_dir, 'data/test_images')}")
    print(f"- LFW dataset directory: {os.path.join(base_dir, 'data/datasets/lfw')}")
    print(f"- Bias testing directory: {os.path.join(base_dir, 'data/test_datasets/sample_dataset')}")

if __name__ == "__main__":
    main()
