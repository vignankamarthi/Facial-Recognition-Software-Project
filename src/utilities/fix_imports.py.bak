"""
Fix Imports Script

This script fixes import statements in all Python files to work with the new directory structure.
"""

import os
import re


def fix_imports_in_file(file_path):
    """
    Fix imports in a given Python file.
    
    Args:
        file_path (str): Path to the Python file
    
    Returns:
        bool: True if changes were made, False otherwise
    """
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make a backup
    backup_path = f"{file_path}.bak"
    with open(backup_path, 'w') as f:
        f.write(content)
    
    # Define replacements for imports
    replacements = [
        (r'from anonymization import', 'from src.facial_recognition_software.anonymization import'),
        (r'from face_detection import', 'from src.facial_recognition_software.face_detection import'),
        (r'from face_matching import', 'from src.facial_recognition_software.face_matching import'),
        (r'from bias_testing import', 'from src.facial_recognition_software.bias_testing import'),
        (r'from image_processing import', 'from src.utilities.image_processing import')
    ]
    
    # Apply replacements
    modified = False
    new_content = content
    for pattern, replacement in replacements:
        new_content, count = re.subn(pattern, replacement, new_content)
        if count > 0:
            modified = True
    
    # Save changes if modified
    if modified:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"Fixed imports in: {file_path}")
    else:
        # Remove backup if no changes
        os.remove(backup_path)
    
    return modified


def fix_imports_in_directory(directory_path):
    """
    Fix imports in all Python files in a directory (recursively).
    
    Args:
        directory_path (str): Path to the directory
    
    Returns:
        int: Number of files modified
    """
    modified_count = 0
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_imports_in_file(file_path):
                    modified_count += 1
    
    return modified_count


def main():
    # Base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Fixing imports in: {base_dir}")
    
    # Fix imports in all Python files
    modified_count = fix_imports_in_directory(base_dir)
    
    print(f"Fixed imports in {modified_count} files.")


if __name__ == "__main__":
    main()
