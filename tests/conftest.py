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
from unittest.mock import MagicMock, patch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# Test data directory management
@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for the test data directory."""
    data_dir = os.path.join(project_root, "tests", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create required subdirectories for tests
    os.makedirs(os.path.join(data_dir, "demographic_split_set", "white"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "demographic_split_set", "black"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "demographic_split_set", "asian"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "demographic_split_set", "indian"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "demographic_split_set", "others"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "test_images"), exist_ok=True)
    
    # Create test image if it doesn't exist
    real_face_path = os.path.join(data_dir, "real_face.jpg")
    if not os.path.exists(real_face_path):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)  # Simple face circle
        cv2.imwrite(real_face_path, img)
    
    return data_dir


def is_ci_environment():
    """Check if we're running in a CI environment."""
    return os.environ.get('CI', 'False').lower() in ('true', '1', 't')


@pytest.fixture
def mock_video_capture():
    """Create a mock for cv2.VideoCapture that behaves like a working camera."""
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    
    # Create a simple test frame (black image with a white circle)
    test_frame = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(test_frame, (200, 150), 100, (255, 255, 255), -1)
    
    # Mock the read method to return the test frame
    mock_capture.read.return_value = (True, test_frame)
    mock_capture.release.return_value = None
    
    # Create the patch
    with patch('cv2.VideoCapture', return_value=mock_capture):
        yield mock_capture


# Pull image from somewhere instead of drawing one if one doesn't exist
# Sample real face image fixture
@pytest.fixture(scope="session")
def sample_image(test_data_dir):
    """Use a real face image for testing.

    Note: The test will look for a real face image in tests/data/real_face.jpg.
    If the image doesn't exist, the test will fail with a clear message.
    """
    image_path = os.path.join(test_data_dir, "real_face.jpg")

    # Check if the real face image exists
    if not os.path.exists(image_path):
        pytest.fail(
            "REQUIRED TEST IMAGE MISSING: Real face test image not found. Please add a real face image at: "
            + image_path
            + "\nTests cannot run without required test images."
        )

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
