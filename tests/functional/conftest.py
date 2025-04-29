"""
Functional test configuration and fixtures for the Facial Recognition Software Project.

This file contains fixtures specific to functional tests, which test
end-to-end workflows and user interactions.
"""

import os
import pytest
import numpy as np
import cv2
import tempfile
import shutil
from unittest.mock import patch

# Temporary working directory fixture
@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory for functional tests."""
    temp_dir = tempfile.mkdtemp(prefix="facial_recognition_test_")
    
    # Create subdirectories
    os.makedirs(os.path.join(temp_dir, "known_faces"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, "test_images"), exist_ok=True)
    
    # Set up a few test images
    for i in range(3):
        # Create test image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 255), -1)
        # Draw a face
        cv2.circle(img, (200, 150), 100, (255, 255, 255), -1)
        # Save it
        cv2.imwrite(os.path.join(temp_dir, "test_images", f"test_{i}.jpg"), img)
        
        # Create a known face
        cv2.imwrite(os.path.join(temp_dir, "known_faces", f"person_{i}.jpg"), img)
    
    # Return the temporary directory
    yield temp_dir
    
    # Clean up
    shutil.rmtree(temp_dir)

# Mock subprocess for testing run_demo.py
@pytest.fixture
def mock_subprocess():
    """Mock the subprocess module for testing the launcher script."""
    with patch("subprocess.Popen") as mock_popen, \
         patch("subprocess.check_call") as mock_check_call:
        
        # Configure the mock
        mock_process = mock_popen.return_value
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_check_call.return_value = 0
        
        yield {
            "popen": mock_popen,
            "check_call": mock_check_call,
            "process": mock_process
        }

# Mock for testing UI components
@pytest.fixture
def mock_streamlit():
    """Mock the streamlit module for testing the UI components."""
    with patch("streamlit.title") as mock_title, \
         patch("streamlit.header") as mock_header, \
         patch("streamlit.sidebar") as mock_sidebar, \
         patch("streamlit.image") as mock_image, \
         patch("streamlit.file_uploader") as mock_file_uploader:
        
        yield {
            "title": mock_title,
            "header": mock_header,
            "sidebar": mock_sidebar,
            "image": mock_image,
            "file_uploader": mock_file_uploader
        }
