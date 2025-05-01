#!/usr/bin/env python
"""
Initializes demo data for the Facial Recognition Software Project Docker container.

This script:
1. Creates sample data directories if they don't exist
2. Provides sample test faces for demonstration
3. Sets up minimal datasets for bias testing demonstration
4. Ensures all necessary files for a demo are available
"""

import os
import sys
import shutil
import numpy as np
import cv2
import logging

# Add the app directory to the path to use the project modules
sys.path.insert(0, '/app')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('init_demo_data')

# Define paths
DATA_DIR = '/app/data'
KNOWN_FACES_DIR = os.path.join(DATA_DIR, 'known_faces')
TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_images')
TEST_DATASETS_DIR = os.path.join(DATA_DIR, 'test_datasets')
DEMOGRAPHIC_SPLIT_DIR = os.path.join(TEST_DATASETS_DIR, 'demographic_split_set')

def ensure_dir_exists(directory):
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def create_test_face(path, has_face=True):
    """Create a test image, optionally with a face-like shape."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # If specified, draw a face-like shape
    if has_face:
        # Draw face (circle)
        cv2.circle(img, (200, 150), 100, (255, 255, 255), -1)
        # Draw eyes
        cv2.circle(img, (150, 120), 20, (0, 0, 0), -1)  # Left eye
        cv2.circle(img, (250, 120), 20, (0, 0, 0), -1)  # Right eye
        # Draw mouth
        cv2.ellipse(img, (200, 180), (50, 20), 0, 0, 180, (0, 0, 0), -1)
    
    # Save the image
    cv2.imwrite(path, img)
    logger.info(f"Created test image: {path}")

def create_bias_test_data():
    """Create sample data for bias testing."""
    # Ensure demographic directories exist
    ethnicities = ['white', 'black', 'asian', 'indian', 'others']
    for ethnicity in ethnicities:
        ethnicity_dir = os.path.join(DEMOGRAPHIC_SPLIT_DIR, ethnicity)
        ensure_dir_exists(ethnicity_dir)
        
        # Create 3 sample face images for each ethnicity
        for i in range(3):
            face_path = os.path.join(ethnicity_dir, f'sample_face_{i}.jpg')
            create_test_face(face_path)

def create_known_faces():
    """Create sample known faces for matching."""
    ensure_dir_exists(KNOWN_FACES_DIR)
    
    # Create 3 reference faces
    names = ['john_smith', 'jane_doe', 'test_person']
    for name in names:
        face_path = os.path.join(KNOWN_FACES_DIR, f'{name}.jpg')
        create_test_face(face_path)

def create_test_images():
    """Create sample test images for detection and matching."""
    # Create directories for known and unknown test images
    known_dir = os.path.join(TEST_IMAGES_DIR, 'known')
    unknown_dir = os.path.join(TEST_IMAGES_DIR, 'unknown')
    ensure_dir_exists(known_dir)
    ensure_dir_exists(unknown_dir)
    
    # Create test images with faces
    create_test_face(os.path.join(TEST_IMAGES_DIR, 'test_face.jpg'))
    create_test_face(os.path.join(known_dir, 'known_face.jpg'))
    create_test_face(os.path.join(unknown_dir, 'unknown_face.jpg'))
    
    # Create a blank image without a face
    create_test_face(os.path.join(TEST_IMAGES_DIR, 'blank_test.jpg'), has_face=False)

def setup_result_directories():
    """Set up directories for storing results."""
    # Create results directories
    ensure_dir_exists(os.path.join(DATA_DIR, 'results'))
    ensure_dir_exists(os.path.join(TEST_DATASETS_DIR, 'results'))

def main():
    """Initialize all demo data."""
    logger.info("Starting demo data initialization...")
    
    # Create necessary directories
    ensure_dir_exists(DATA_DIR)
    ensure_dir_exists(KNOWN_FACES_DIR)
    ensure_dir_exists(TEST_IMAGES_DIR)
    ensure_dir_exists(TEST_DATASETS_DIR)
    ensure_dir_exists(DEMOGRAPHIC_SPLIT_DIR)
    
    # Create sample data
    create_known_faces()
    create_test_images()
    create_bias_test_data()
    setup_result_directories()
    
    logger.info("Demo data initialization complete!")

if __name__ == "__main__":
    main()
