#!/usr/bin/env python3
"""
Cleanup Script

This script cleans up unnecessary files in the project directory:
- Removes Python cache files (__pycache__, .pyc, .pyo)
- Clears temporary generated datasets
- Removes backup files (.bak, .backup)
- Cleans up logs and error reports
- Optionally resets the face datasets to their initial state

Use this script before committing changes or when experiencing unexpected behavior.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def delete_cache_files(project_dir, dry_run=False):
    """Delete Python cache files and directories."""
    print_section("Removing Python Cache Files")
    
    # Cache patterns to delete
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/.pytest_cache",
        "**/.coverage",
        "**/htmlcov",
        "**/.tox"
    ]
    
    deleted_count = 0
    
    try:
        for pattern in cache_patterns:
            for path in Path(project_dir).glob(pattern):
                if path.is_dir():
                    if not dry_run:
                        try:
                            shutil.rmtree(path)
                            print(f"Deleted directory: {path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
                    else:
                        print(f"Would delete directory: {path}")
                        deleted_count += 1
                else:
                    if not dry_run:
                        try:
                            os.remove(path)
                            print(f"Deleted file: {path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
                    else:
                        print(f"Would delete file: {path}")
                        deleted_count += 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Stopping cache file deletion...")
        return deleted_count
    
    print(f"\nTotal Python cache items {'that would be ' if dry_run else ''}deleted: {deleted_count}")
    return deleted_count


def delete_backup_files(project_dir, dry_run=False):
    """Delete backup files created during development."""
    print_section("Removing Backup Files")
    
    # Backup patterns to delete
    backup_patterns = [
        "**/*.bak",
        "**/*.backup",
        "**/*.swp",
        "**/*.swo",
        "**/*~",
        "**/*.tmp"
    ]
    
    deleted_count = 0
    
    try:
        for pattern in backup_patterns:
            for path in Path(project_dir).glob(pattern):
                if not dry_run:
                    try:
                        os.remove(path)
                        print(f"Deleted backup file: {path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")
                else:
                    print(f"Would delete backup file: {path}")
                    deleted_count += 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Stopping backup file deletion...")
        return deleted_count
    
    print(f"\nTotal backup files {'that would be ' if dry_run else ''}deleted: {deleted_count}")
    return deleted_count


def cleanup_datasets(project_dir, reset_all=False, dry_run=False):
    """Clean up dataset files and directories."""
    print_section("Cleaning Datasets")

    # Get the data directory
    data_dir = os.path.join(project_dir, "data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return 0

    deleted_count = 0

    # Always clean these temporary/generated files
    temp_patterns = [
        "**/*.tgz",
        "**/*.tar.gz",
        "**/*.zip",
        "**/results/*.png",
        "**/results/*.jpg",
        "**/results/*.jpeg",
        "**/utkface/**/*.zip"
    ]

    try:
        # Delete temporary files
        for pattern in temp_patterns:
            for path in Path(data_dir).glob(pattern):
                if not dry_run:
                    try:
                        if path.is_dir():
                            shutil.rmtree(path)
                            print(f"Deleted directory: {path}")
                        else:
                            os.remove(path)
                            print(f"Deleted file: {path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {path}: {e}")
                else:
                    print(f"Would delete: {path}")
                    deleted_count += 1

        # Optional: Reset all datasets
        if reset_all:
            dataset_dirs = ["datasets", "known_faces", "test_datasets", "test_images"]

            for dir_name in dataset_dirs:
                dir_path = os.path.join(data_dir, dir_name)
                if os.path.exists(dir_path):
                    if not dry_run:
                        try:
                            # Delete content but preserve directory structure
                            for item in os.listdir(dir_path):
                                item_path = os.path.join(dir_path, item)
                                if os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                                else:
                                    os.remove(item_path)
                            print(f"Reset dataset directory: {dir_path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error resetting {dir_path}: {e}")
                    else:
                        print(f"Would reset dataset directory: {dir_path}")
                        deleted_count += 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Stopping dataset cleanup...")
        return deleted_count

    print(f"\nTotal dataset items {'that would be ' if dry_run else ''}cleaned: {deleted_count}")
    return deleted_count


def delete_logs(project_dir, dry_run=False):
    """Delete log files."""
    print_section("Removing Log Files")
    
    # Log patterns to delete
    log_patterns = [
        "**/*.log",
        "**/logs/*",
        "**/error_reports/*"
    ]
    
    deleted_count = 0
    
    try:
        for pattern in log_patterns:
            for path in Path(project_dir).glob(pattern):
                if path.is_dir() and not path.match("**/logs") and not path.match("**/error_reports"):
                    # Skip deleting the log directory itself, just its contents
                    continue
                    
                if path.is_dir():
                    if not dry_run:
                        try:
                            shutil.rmtree(path)
                            print(f"Deleted log directory: {path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
                    else:
                        print(f"Would delete log directory: {path}")
                        deleted_count += 1
                else:
                    if not dry_run:
                        try:
                            os.remove(path)
                            print(f"Deleted log file: {path}")
                            deleted_count += 1
                        except Exception as e:
                            print(f"Error deleting {path}: {e}")
                    else:
                        print(f"Would delete log file: {path}")
                        deleted_count += 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user. Stopping log file deletion...")
        return deleted_count
    
    print(f"\nTotal log items {'that would be ' if dry_run else ''}deleted: {deleted_count}")
    return deleted_count


def main():
    """Main function to clean up the project."""
    parser = argparse.ArgumentParser(description="Clean up unnecessary files in the project directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--reset-datasets", action="store_true", help="Reset all datasets to initial state")
    parser.add_argument("--project-dir", type=str, help="Project directory (default: autodetect)")
    
    args = parser.parse_args()
    
    # Get project directory
    if args.project_dir:
        project_dir = args.project_dir
    else:
        # Try to detect project directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))
    
    if not os.path.exists(project_dir):
        print(f"Error: Project directory not found: {project_dir}")
        sys.exit(1)
    
    print(f"Cleaning up project directory: {project_dir}")
    print(f"Dry run: {'Yes' if args.dry_run else 'No'}")
    
    try:
        # Run cleanup operations
        total_deleted = 0
        total_deleted += delete_cache_files(project_dir, args.dry_run)
        total_deleted += delete_backup_files(project_dir, args.dry_run)
        total_deleted += cleanup_datasets(project_dir, args.reset_datasets, args.dry_run)
        total_deleted += delete_logs(project_dir, args.dry_run)
        
        print_section("Cleanup Summary")
        if args.dry_run:
            print(f"Total items that would be cleaned up: {total_deleted}")
            print("No files were actually deleted (dry run mode)")
        else:
            print(f"Total items cleaned up: {total_deleted}")
        
        print("\nCleanup complete!")
        
        if not args.dry_run and total_deleted > 0:
            print("\nNote: If you're using Git, you might want to run:")
            print("  git status")
            print("to see the changes made by this cleanup.")
            
    except KeyboardInterrupt:
        print("\n\nCleanup interrupted by user.")
        print_section("Cleanup Terminated")
        print("The cleanup process was stopped before completion.")
        print("Some files may have been removed while others remain.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nAn error occurred during cleanup: {e}")
        print_section("Cleanup Error")
        print("The cleanup process encountered an error and did not complete.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting gracefully.")
        sys.exit(0)
