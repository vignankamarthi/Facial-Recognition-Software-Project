"""
Configuration Module

This module centralizes all configuration settings for the facial recognition project.
It provides a single source of truth for paths, thresholds, and other parameters.
"""

import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Data subdirectories
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
TEST_DATASETS_DIR = os.path.join(DATA_DIR, "test_datasets")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# UTKFace dataset paths
UTKFACE_DIR = os.path.join(DATASETS_DIR, "utkface")
UTKFACE_ALIGNED_DIR = os.path.join(UTKFACE_DIR, "utkface_aligned")
UTKFACE_DEMOGRAPHIC_SPLIT_DIR = os.path.join(UTKFACE_DIR, "demographic_split")

# Demographic bias testing paths
DEMOGRAPHIC_SPLIT_SET_DIR = os.path.join(TEST_DATASETS_DIR, "demographic_split_set")
DEMOGRAPHIC_RESULTS_DIR = os.path.join(TEST_DATASETS_DIR, "results")

# Image processing parameters
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Face detection parameters
FACE_DETECTION_CONFIDENCE = 0.5  # Confidence threshold for face detection

# Face matching parameters
FACE_MATCHING_THRESHOLD = 0.6  # Threshold for face matching (higher = stricter)

# Anonymization parameters
DEFAULT_ANONYMIZATION_METHOD = "blur"
DEFAULT_ANONYMIZATION_INTENSITY = 90  # Range 1-100

# Demographic bias testing parameters
DEMOGRAPHIC_GROUPS = ["white", "black", "asian", "indian", "others"]
ETHNICITY_CODES = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
GENDER_CODES = {0: "Male", 1: "Female"}

# GUI parameters
WINDOW_NAME = "Video"
TEXT_COLOR = (255, 255, 255)
SUCCESS_COLOR = (0, 255, 0)  # Green
WARNING_COLOR = (0, 255, 255)  # Yellow
ERROR_COLOR = (0, 0, 255)  # Red

# OpenCV parameters
WAIT_KEY_DELAY = 100  # Milliseconds to wait for key press

# Avoid reference to cv2 at module level to prevent import issues
# These will be set when cv2 is actually imported
FONT = None  # Will be cv2.FONT_HERSHEY_SIMPLEX when available
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Function to initialize OpenCV-dependent constants
def initialize_opencv_constants():
    """Initialize constants that depend on OpenCV. Call this after cv2 is imported."""
    global FONT
    import cv2
    FONT = cv2.FONT_HERSHEY_SIMPLEX

# Function to ensure a directory exists
def ensure_dir_exists(directory_path):
    """Ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

# Ensure essential directories exist
for directory in [
    DATA_DIR, KNOWN_FACES_DIR, TEST_IMAGES_DIR, TEST_DATASETS_DIR,
    DATASETS_DIR, RESULTS_DIR, DEMOGRAPHIC_RESULTS_DIR
]:
    ensure_dir_exists(directory)
