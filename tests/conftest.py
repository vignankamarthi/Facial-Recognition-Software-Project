"""
Global test configuration and fixtures for the Facial Recognition Software Project.

This file contains fixtures and configuration shared across all test categories
(unit, integration, and functional tests).
"""

import os
import sys
import pytest
import numpy as np
import cv2

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Test data directory management
@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for the test data directory."""
    data_dir = os.path.join(project_root, "tests", "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

# TODO: Pull image from somewhere instead of drawing one if one doesn't exist
# Sample real face image fixture
@pytest.fixture(scope="session")
def sample_image(test_data_dir):
    """Use a real face image for testing.
    
    Note: The test will look for a real face image in test_data_dir/real_face.jpg.
    If the image doesn't exist, the test will be skipped with a message
    suggesting to provide a real face image.
    """
    image_path = os.path.join(test_data_dir, "real_face.jpg")
    
    # Check if the real face image exists
    if not os.path.exists(image_path):
        pytest.skip("Real face test image not found. Please add a real face image at: " + image_path)
    
    return image_path

# Mock face detection result fixture 
@pytest.fixture
def mock_face_locations():
    """Return mock face detection results."""
    return [(50, 350, 250, 50)]  # (top, right, bottom, left)

# Cleanup after tests
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up any resources after each test."""
    yield
    # Clean up OpenCV windows that might be left open
    cv2.destroyAllWindows()
