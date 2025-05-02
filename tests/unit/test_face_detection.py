"""
Unit tests for the face detection module.
"""

import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.backend.face_detection import FaceDetector, DetectionError


class TestFaceDetector:
    """Tests for the FaceDetector class."""

    def test_initialization(self):
        """Test that the detector initializes correctly."""
        detector = FaceDetector()
        assert detector.face_locations == []
        assert detector.face_encodings == []

    def test_detect_faces_invalid_input(self):
        """Test that detect_faces handles invalid inputs correctly."""
        detector = FaceDetector()

        # Test with None input
        with pytest.raises(ValueError, match="Cannot detect faces in None image"):
            detector.detect_faces(None)

        # Test with empty frame
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Empty image provided for face detection"):
            detector.detect_faces(empty_frame)

    @patch("src.backend.face_detection.face_recognition.face_locations")
    @patch("src.backend.face_detection.face_recognition.face_encodings")
    def test_detect_faces(self, mock_encodings, mock_locations, sample_image):
        """Test face detection on a sample image."""
        # Configure the mocks
        expected_locations = [(50, 350, 250, 50)]
        expected_encodings = [np.ones(128)]  # Mock encoding
        mock_locations.return_value = expected_locations
        mock_encodings.return_value = expected_encodings

        # Create detector and load the test image
        detector = FaceDetector()
        test_image = cv2.imread(sample_image)

        # Run the detection
        face_locations, face_encodings = detector.detect_faces(test_image)

        # Verify results
        assert face_locations == expected_locations
        assert face_encodings == expected_encodings

        # Verify the mocks were called correctly
        mock_locations.assert_called_once()
        mock_encodings.assert_called_once()

    def test_draw_face_boxes(self, sample_image):
        """Test drawing boxes around detected faces."""
        # Create detector and load the test image
        detector = FaceDetector()
        test_image = cv2.imread(sample_image)

        # Define some test face locations
        face_locations = [(50, 350, 250, 50), (100, 200, 150, 120)]

        # Define a test color and thickness
        test_color = (0, 255, 0)  # Green
        test_thickness = 2

        # Test with default color and thickness
        with patch("cv2.rectangle") as mock_rectangle:
            result_image = detector.draw_face_boxes(test_image, face_locations)

            # Verify the result is a new image (copy)
            assert result_image is not test_image

            # Verify rectangle was called for each face location
            assert mock_rectangle.call_count == len(face_locations)
            for i, (top, right, bottom, left) in enumerate(face_locations):
                mock_rectangle.assert_any_call(
                    result_image,
                    (left, top),
                    (right, bottom),
                    test_color,
                    test_thickness,
                )

        # Test with invalid input (None frame)
        with pytest.raises(ValueError, match="Cannot draw boxes on None image"):
            detector.draw_face_boxes(None, face_locations)

        # Test with non-list face_locations
        with patch("cv2.rectangle") as mock_rectangle:
            non_list_locations = tuple(face_locations)  # Convert to tuple
            result_image = detector.draw_face_boxes(test_image, non_list_locations)

            # Should still work by converting to list
            assert mock_rectangle.call_count == len(face_locations)





    def test_detect_faces_exception_handling(self):
        """Test that detect_faces properly handles exceptions."""
        detector = FaceDetector()

        # Create a valid image but make face_recognition.face_locations throw an exception
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch(
            "src.backend.face_detection.face_recognition.face_locations"
        ) as mock_locations, patch(
            "src.backend.face_detection.log_exception"
        ) as mock_log_exception:

            # Make the mock raise an exception
            mock_locations.side_effect = Exception("Test exception")

            # Verify the exception is caught and transformed into a DetectionError
            with pytest.raises(DetectionError, match="Face detection failed"):
                detector.detect_faces(test_image)

            # Verify error was logged
            mock_log_exception.assert_called_once()


