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

# Project root was already added above

# Import environment utilities
from src.utils.environment_utils import is_ci_environment, is_headless_environment, is_webcam_available

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
    
    # Create test image if it doesn't exist or if in CI environment (always recreate in CI)
    real_face_path = os.path.join(data_dir, "real_face.jpg")
    if not os.path.exists(real_face_path) or is_ci_environment():
        # Create a more recognizable face-like image for unit tests
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img.fill(220)  # Light gray background
        
        # Draw face shape (ellipse for more realistic face shape)
        cv2.ellipse(img, (150, 150), (120, 150), 0, 0, 360, (240, 200, 200), -1)  # Face shape
        
        # Add more pronounced features with good contrast
        cv2.circle(img, (110, 120), 30, (40, 40, 180), -1)  # Left eye
        cv2.circle(img, (190, 120), 30, (40, 40, 180), -1)  # Right eye
        cv2.ellipse(img, (150, 190), (60, 30), 0, 0, 180, (40, 40, 180), -1)  # Mouth
        cv2.ellipse(img, (150, 130), (80, 40), 0, 180, 360, (40, 40, 180), 5)  # Eyebrows
        
        # Add a nose for better face detection
        cv2.ellipse(img, (150, 160), (25, 35), 0, 0, 360, (180, 120, 120), -1)  # Nose
        
        print(f"Creating main test face image at {real_face_path}")
        cv2.imwrite(real_face_path, img)
    
    return data_dir


# Using is_ci_environment from environment_utils module


@pytest.fixture
def mock_video_capture():
    """Create a mock for cv2.VideoCapture that behaves like a working camera.
    
    This fixture automatically applies in headless environments or when no webcam is available.
    In environments with a working webcam, it will only apply if explicitly requested.
    """
    # Create a more realistic test frame with a recognizable face
    test_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    test_frame.fill(200)  # Light gray background
    
    # Face centered in the frame
    center_x, center_y = 300, 200
    
    # Draw face shape
    cv2.ellipse(test_frame, (center_x, center_y), (120, 160), 0, 0, 360, (255, 220, 200), -1)  # Face shape
    
    # Add recognizable features
    cv2.circle(test_frame, (center_x-40, center_y-30), 30, (60, 60, 170), -1)  # Left eye
    cv2.circle(test_frame, (center_x+40, center_y-30), 30, (60, 60, 170), -1)  # Right eye
    cv2.ellipse(test_frame, (center_x, center_y+50), (60, 25), 0, 0, 180, (60, 60, 170), -1)  # Mouth
    cv2.ellipse(test_frame, (center_x, center_y-40), (90, 40), 0, 180, 360, (60, 60, 170), 5)  # Eyebrows
    
    # Add nose
    cv2.ellipse(test_frame, (center_x, center_y+10), (25, 40), 0, 0, 360, (200, 120, 120), -1)
    
    # Configure mock
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.read.return_value = (True, test_frame)
    mock_capture.release.return_value = None
    
    # In headless environments or when webcam is not available, always mock
    if is_headless_environment() or not is_webcam_available():
        with patch('cv2.VideoCapture', return_value=mock_capture):
            yield mock_capture
    else:
        # In environments with webcam, only mock if tests explicitly use this fixture
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
    # Skip in headless environments where no windows can be created
    if not is_headless_environment():
        cv2.destroyAllWindows()


@pytest.fixture
def headless_aware_test():
    """Fixture that provides information about the test environment.
    
    This fixture helps tests adapt to different environments (headless, CI, webcam availability).
    It can be used to conditionally skip tests or adjust expectations.
    """
    env_info = {
        'headless': is_headless_environment(),
        'ci': is_ci_environment(),
        'webcam_available': is_webcam_available()
    }
    
    # Print environment info for better test debugging
    print(f"\nTest environment: {env_info}")
    
    return env_info
