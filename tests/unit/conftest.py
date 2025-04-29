"""
Unit test configuration and fixtures for the Facial Recognition Software Project.

This file contains fixtures specific to unit tests, including mocks for
external dependencies.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Mock the face_recognition library
@pytest.fixture
def mock_face_recognition():
    """Mock the face_recognition library."""
    with patch("face_recognition.face_locations") as mock_locations, \
         patch("face_recognition.face_encodings") as mock_encodings, \
         patch("face_recognition.compare_faces") as mock_compare, \
         patch("face_recognition.face_distance") as mock_distance, \
         patch("face_recognition.load_image_file") as mock_load:
        
        # Configure the mocks
        mock_locations.return_value = [(50, 350, 250, 50)]  # (top, right, bottom, left)
        mock_encodings.return_value = [np.random.rand(128)]  # Random 128D encoding
        mock_compare.return_value = [True]  # Match found
        mock_distance.return_value = np.array([0.4])  # 60% similarity
        mock_load.side_effect = lambda x: np.zeros((300, 400, 3), dtype=np.uint8)
        
        yield {
            "face_locations": mock_locations,
            "face_encodings": mock_encodings,
            "compare_faces": mock_compare, 
            "face_distance": mock_distance,
            "load_image_file": mock_load
        }

# Mock the OpenCV (cv2) functions
@pytest.fixture
def mock_cv2():
    """Mock OpenCV (cv2) functions."""
    with patch("cv2.imread") as mock_imread, \
         patch("cv2.imwrite") as mock_imwrite, \
         patch("cv2.rectangle") as mock_rectangle, \
         patch("cv2.putText") as mock_puttext, \
         patch("cv2.VideoCapture") as mock_videocapture:
        
        # Configure the mocks
        mock_imread.return_value = np.zeros((300, 400, 3), dtype=np.uint8)
        mock_imwrite.return_value = True
        mock_rectangle.return_value = None
        mock_puttext.return_value = None
        
        # Video capture mock
        mockCapture = MagicMock()
        mockCapture.isOpened.return_value = True
        mockCapture.read.return_value = (True, np.zeros((300, 400, 3), dtype=np.uint8))
        mockCapture.release.return_value = None
        mock_videocapture.return_value = mockCapture
        
        yield {
            "imread": mock_imread,
            "imwrite": mock_imwrite,
            "rectangle": mock_rectangle,
            "putText": mock_puttext,
            "VideoCapture": mock_videocapture
        }
