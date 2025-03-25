"""
Update Image Paths

This script updates the paths in the image_processing.py file to ensure that downloaded
faces are routed to the correct directories during demos.
"""

import os
import re
import sys

def update_file_paths(file_path, base_dir):
    """
    Update file paths in the specified file to ensure correct routing of downloaded faces.
    
    Args:
        file_path (str): Path to the file to update
        base_dir (str): Base directory of the project
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    try:
        # Read the file
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Define the correct paths
        correct_paths = {
            "../data/sample_faces": os.path.join(base_dir, "data/sample_faces"),
            "../data/test_images": os.path.join(base_dir, "data/test_images"),
            "../data/test_datasets": os.path.join(base_dir, "data/test_datasets"),
            "../data/datasets/lfw": os.path.join(base_dir, "data/datasets/lfw")
        }
        
        # Create a backup of the original file
        backup_path = file_path + '.bak'
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Created backup: {backup_path}")
        
        # Update the paths
        updated_content = content
        for old_path, new_path in correct_paths.items():
            # Make sure to use the paths as they appear in the code
            updated_content = updated_content.replace(f'"{old_path}"', f'"{new_path}"')
            updated_content = updated_content.replace(f"'{old_path}'", f"'{new_path}'")
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated paths in {file_path}")
        return True
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update paths in all relevant files."""
    # Get the project base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Project base directory: {base_dir}")
    
    # List of files to update
    files_to_update = [
        os.path.join(base_dir, "src/image_processing.py"),
        os.path.join(base_dir, "src/face_matching.py"),
        os.path.join(base_dir, "src/bias_testing.py")
    ]
    
    # Update paths in each file
    for file_path in files_to_update:
        if os.path.exists(file_path):
            print(f"Checking paths in {file_path}...")
            update_file_paths(file_path, base_dir)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print("\nPath updates completed!")
    print("\nREMINDER: You might need to adjust the paths in the Python files manually")
    print("if they use different path formats or if the automatic update missed anything.")

if __name__ == "__main__":
    main()
