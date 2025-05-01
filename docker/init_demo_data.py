#!/usr/bin/env python
"""
Initializes directory structure for the Facial Recognition Software Project Docker container.

This script:
1. Creates required data directories if they don't exist
2. Sets up the basic structure needed for the application
3. Adds a log message guiding users to download the actual UTKFace dataset

# TODO: Implement a message in the UI directing users to follow README instructions
# for downloading the UTKFace dataset for proper demonstration
"""

import os
import sys
import logging

# Add the app directory to the path to use the project modules
sys.path.insert(0, "/app")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("init_demo_data")

# Define paths
DATA_DIR = "/app/data"
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
TEST_DATASETS_DIR = os.path.join(DATA_DIR, "test_datasets")
DEMOGRAPHIC_SPLIT_DIR = os.path.join(TEST_DATASETS_DIR, "demographic_split_set")


def ensure_dir_exists(directory):
    """Ensure directory exists, create if it doesn't."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    except PermissionError:
        logger.error(f"Permission denied when creating directory: {directory}")
        logger.info("Try running the container with appropriate volume permissions")
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")


def setup_directory_structure():
    """Set up the complete directory structure needed for the application."""
    # Create main data directories
    ensure_dir_exists(DATA_DIR)
    ensure_dir_exists(KNOWN_FACES_DIR)
    ensure_dir_exists(TEST_IMAGES_DIR)
    ensure_dir_exists(TEST_DATASETS_DIR)

    # Create known/unknown test image directories
    ensure_dir_exists(os.path.join(TEST_IMAGES_DIR, "known"))
    ensure_dir_exists(os.path.join(TEST_IMAGES_DIR, "unknown"))

    # Create demographic split directories
    ethnicities = ["white", "black", "asian", "indian", "others"]
    for ethnicity in ethnicities:
        ethnicity_dir = os.path.join(DEMOGRAPHIC_SPLIT_DIR, ethnicity)
        ensure_dir_exists(ethnicity_dir)

    # Create results directories
    ensure_dir_exists(os.path.join(DATA_DIR, "results"))
    ensure_dir_exists(os.path.join(TEST_DATASETS_DIR, "results"))

    # Additional dataset directories
    ensure_dir_exists(os.path.join(DATA_DIR, "datasets", "utkface", "utkface_aligned"))
    ensure_dir_exists(
        os.path.join(DATA_DIR, "datasets", "utkface", "demographic_split")
    )






def main():
    """Initialize directory structure for the application."""
    logger.info("Setting up directory structure for the application...")

    # Create all necessary directories
    setup_directory_structure()
    
    # Log instructions for downloading the actual dataset
    logger.info("Directory structure created!")
    logger.info(
        "IMPORTANT: For proper demonstration, please follow the instructions in README.md"
    )
    logger.info("to download the UTKFace dataset and prepare it for bias testing.")

    # TODO: Add UI message directing users to download the proper dataset

    logger.info("Initialization complete!")


if __name__ == "__main__":
    main()
