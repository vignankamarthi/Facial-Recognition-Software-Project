"""
Unit tests for the face anonymization module.
"""

import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from src.backend.anonymization import FaceAnonymizer


class TestFaceAnonymizer:
    """Tests for the FaceAnonymizer class."""

    def test_initialization(self):
        """Test that the anonymizer initializes correctly."""
        # Test default initialization
        anonymizer = FaceAnonymizer()
        assert anonymizer.method == "blur"  # Default method
        assert anonymizer.intensity == 90  # Default intensity

        # Test custom initialization
        anonymizer = FaceAnonymizer(method="pixelate", intensity=50)
        assert anonymizer.method == "pixelate"
        assert anonymizer.intensity == 50

    def test_anonymize_face_blur(self, sample_image):
        """Test blur anonymization on a single face."""
        # Create an anonymizer with blur method
        anonymizer = FaceAnonymizer(
            method="blur", intensity=21
        )  # Use odd intensity for kernel size

        # Load sample image
        image = cv2.imread(sample_image)

        # Define face location
        face_location = (50, 200, 150, 100)  # (top, right, bottom, left)

        # Mock the GaussianBlur function to verify it's called with expected parameters
        with patch("cv2.GaussianBlur") as mock_blur, patch(
            "cv2.rectangle"
        ) as mock_rectangle, patch("cv2.putText") as mock_text:

            # Configure the blur mock
            blurred_face = np.zeros(
                (100, 100, 3), dtype=np.uint8
            )  # Mock blurred result
            mock_blur.return_value = blurred_face

            # Apply anonymization
            result = anonymizer.anonymize_face(image, face_location)

            # Check the result is a new image (copy)
            assert result is not image

            # Verify GaussianBlur was called
            mock_blur.assert_called_once()
            args, kwargs = mock_blur.call_args
            # Check kernel size matches intensity
            assert args[1] == (21, 21)  # Kernel size should be (intensity, intensity)

            # Verify rectangle was called to draw anonymization indicator
            mock_rectangle.assert_called()

            # Verify text was added
            mock_text.assert_called()

    def test_anonymize_face_pixelate(self, sample_image):
        """Test pixelation anonymization on a single face."""
        # Create an anonymizer with pixelate method
        anonymizer = FaceAnonymizer(method="pixelate", intensity=50)

        # Load sample image
        image = cv2.imread(sample_image)

        # Define face location
        face_location = (50, 200, 150, 100)  # (top, right, bottom, left)

        # Mock the resize function to verify it's called with expected parameters
        with patch("cv2.resize") as mock_resize, patch(
            "cv2.rectangle"
        ) as mock_rectangle, patch("cv2.putText") as mock_text:

            # Configure the resize mock
            downscaled = np.zeros((20, 20, 3), dtype=np.uint8)  # Mock downscaled result
            upscaled = np.zeros((100, 100, 3), dtype=np.uint8)  # Mock upscaled result
            mock_resize.side_effect = [downscaled, upscaled]

            # Apply anonymization
            result = anonymizer.anonymize_face(image, face_location, method="pixelate")

            # Check the result is a new image (copy)
            assert result is not image

            # Verify resize was called twice (downscale and upscale)
            assert mock_resize.call_count == 2

            # Second call should use INTER_NEAREST interpolation for pixelation
            _, kwargs = mock_resize.call_args_list[1]
            assert kwargs.get("interpolation") == cv2.INTER_NEAREST

            # Verify rectangle was called to draw anonymization indicator
            mock_rectangle.assert_called()

            # Verify text was added
            mock_text.assert_called()

    def test_anonymize_face_mask(self, sample_image):
        """Test mask anonymization on a single face."""
        # Create an anonymizer with mask method
        anonymizer = FaceAnonymizer(method="mask", intensity=50)

        # Load sample image
        image = cv2.imread(sample_image)

        # Define face location
        face_location = (50, 200, 150, 100)  # (top, right, bottom, left)

        # Mock the required functions
        with patch("cv2.rectangle") as mock_rectangle, patch(
            "cv2.circle"
        ) as mock_circle, patch("cv2.ellipse") as mock_ellipse, patch(
            "cv2.putText"
        ) as mock_text:

            # Apply anonymization
            result = anonymizer.anonymize_face(image, face_location, method="mask")

            # Check the result is a new image (copy)
            assert result is not image

            # Verify rectangle was called to draw the mask and anonymization indicator
            assert mock_rectangle.call_count >= 2

            # Verify circles were drawn (eyes)
            assert mock_circle.call_count >= 3  # Face + 2 eyes

            # Verify ellipse was called (mouth)
            mock_ellipse.assert_called_once()

            # Verify text was added
            mock_text.assert_called()

    def test_anonymize_frame(self, sample_image, mock_face_locations):
        """Test anonymizing all faces in a frame."""
        # Create an anonymizer
        anonymizer = FaceAnonymizer(method="blur", intensity=30)

        # Load sample image
        image = cv2.imread(sample_image)

        # Multiple face locations
        face_locations = [(50, 200, 150, 100), (200, 350, 300, 250)]

        # Mock anonymize_face to track calls
        with patch.object(anonymizer, "anonymize_face") as mock_anonymize_face, patch(
            "cv2.rectangle"
        ) as mock_rectangle, patch("cv2.addWeighted") as mock_addWeighted, patch(
            "cv2.putText"
        ) as mock_text:

            # Configure the mock to return a copy of the image
            mock_anonymize_face.side_effect = lambda frame, loc: frame.copy()

            # Anonymize all faces
            result = anonymizer.anonymize_frame(image, face_locations)

            # Check the result is a new image
            assert result is not image

            # Verify anonymize_face was called for each face location
            assert mock_anonymize_face.call_count == len(face_locations)
            for i, face_location in enumerate(face_locations):
                mock_anonymize_face.assert_any_call(
                    mock_anonymize_face.return_value, face_location
                )

            # Verify semi-transparent background was added
            mock_rectangle.assert_called()
            mock_addWeighted.assert_called_once()

            # Verify method indicator was added
            mock_text.assert_called_once()

    def test_set_method(self):
        """Test changing the anonymization method."""
        anonymizer = FaceAnonymizer(method="blur")

        # Test valid method changes
        with patch("builtins.print") as mock_print:
            # Change to pixelate
            anonymizer.set_method("pixelate")
            assert anonymizer.method == "pixelate"
            mock_print.assert_called_with("Anonymization method set to: pixelate")

            # Change to mask
            anonymizer.set_method("mask")
            assert anonymizer.method == "mask"
            mock_print.assert_called_with("Anonymization method set to: mask")

            # Change back to blur
            anonymizer.set_method("blur")
            assert anonymizer.method == "blur"
            mock_print.assert_called_with("Anonymization method set to: blur")

        # Test invalid method
        with patch("builtins.print") as mock_print:
            original_method = anonymizer.method
            anonymizer.set_method("invalid_method")

            # Method should not change
            assert anonymizer.method == original_method

            # Error message should be printed
            mock_print.assert_called_once()
            args, _ = mock_print.call_args
            assert "Invalid method" in args[0]

    def test_set_intensity(self):
        """Test changing the anonymization intensity."""
        anonymizer = FaceAnonymizer(intensity=50)

        # Test valid intensity changes
        with patch("builtins.print") as mock_print:
            # Change to minimum
            anonymizer.set_intensity(1)
            assert anonymizer.intensity == 1
            mock_print.assert_called_with("Anonymization intensity set to: 1")

            # Change to maximum
            anonymizer.set_intensity(100)
            assert anonymizer.intensity == 100
            mock_print.assert_called_with("Anonymization intensity set to: 100")

            # Change to middle value
            anonymizer.set_intensity(50)
            assert anonymizer.intensity == 50
            mock_print.assert_called_with("Anonymization intensity set to: 50")

        # Test invalid intensities
        with patch("builtins.print") as mock_print:
            # Below minimum
            original_intensity = anonymizer.intensity
            anonymizer.set_intensity(0)

            # Intensity should not change
            assert anonymizer.intensity == original_intensity

            # Error message should be printed
            mock_print.assert_called_with("Intensity must be between 1 and 100")

            # Above maximum
            anonymizer.set_intensity(101)

            # Intensity should not change
            assert anonymizer.intensity == original_intensity

            # Error message should be printed
            mock_print.assert_called_with("Intensity must be between 1 and 100")

    def test_demonstrate_methods(self, sample_image, mock_face_locations):
        """Test the method demonstration function."""
        # Create an anonymizer
        anonymizer = FaceAnonymizer()

        # Load sample image
        image = cv2.imread(sample_image)

        # Single face location
        face_location = (50, 200, 150, 100)  # (top, right, bottom, left)

        # Mock anonymize_face to track calls for different methods
        with patch.object(anonymizer, "anonymize_face") as mock_anonymize_face:
            # Configure the mock to return a copy of the image
            mock_anonymize_face.side_effect = (
                lambda frame, loc, method=None: frame.copy()
            )

            # Demonstrate methods
            result = anonymizer.demonstrate_methods(image, face_location)

            # Verify the result is a dictionary
            assert isinstance(result, dict)

            # Verify all methods are included
            assert "original" in result
            assert "blur" in result
            assert "pixelate" in result
            assert "mask" in result

            # Verify anonymize_face was called with each method
            assert mock_anonymize_face.call_count == 3  # blur, pixelate, mask
            mock_anonymize_face.assert_any_call(image, face_location, "blur")
            mock_anonymize_face.assert_any_call(image, face_location, "pixelate")
            mock_anonymize_face.assert_any_call(image, face_location, "mask")

            # Verify original is a copy (not the same as input)
            assert result["original"] is not image

    def test_run_anonymization_demo(self):
        """Test the standalone anonymization demo function."""
        # Test with mocked dependencies to avoid actual webcam usage
        with patch(
            "src.backend.anonymization.FaceAnonymizer"
        ) as mock_anonymizer_class, patch(
            "src.backend.anonymization._FaceDetector", create=True
        ) as mock_detector_class, patch(
            "cv2.VideoCapture"
        ) as mock_video_capture, patch(
            "src.backend.anonymization.create_resizable_window"
        ) as mock_create_window, patch(
            "cv2.imshow"
        ) as mock_imshow, patch(
            "cv2.waitKey"
        ) as mock_waitkey, patch(
            "src.backend.anonymization.safely_close_windows"
        ) as mock_close_windows:

            # Configure mocks
            mock_detector = MagicMock()
            mock_detector.detect_faces.return_value = (
                [(50, 200, 150, 100)],
                [np.ones(128)],
            )
            mock_detector_class.return_value = mock_detector

            mock_anonymizer = MagicMock()
            mock_anonymizer_class.return_value = mock_anonymizer

            # Video capture returns a valid frame then simulates pressing 'q'
            mock_video_capture.return_value.isOpened.return_value = True
            mock_video_capture.return_value.read.return_value = (
                True,
                np.zeros((300, 400, 3), dtype=np.uint8),
            )
            mock_waitkey.return_value = ord("q")  # Simulate pressing 'q' to quit

            # Run the demo
            from src.backend.anonymization import run_anonymization_demo

            run_anonymization_demo()

            # Verify the appropriate functions were called
            mock_detector_class.assert_called_once()
            mock_anonymizer_class.assert_called_once()
            mock_detector.detect_faces.assert_called()
            mock_anonymizer.anonymize_frame.assert_called()
            mock_close_windows.assert_called_once()

    def test_anonymize_face_with_even_intensity(self):
        """Test that blur method works with even intensity values (kernel size must be odd)."""
        # Create an anonymizer with even intensity
        anonymizer = FaceAnonymizer(method="blur", intensity=30)  # Even intensity

        # Create a test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)

        # Define face location
        face_location = (50, 150, 150, 50)  # (top, right, bottom, left)

        # Mock GaussianBlur
        with patch("cv2.GaussianBlur") as mock_blur:
            # Apply anonymization
            anonymizer.anonymize_face(test_image, face_location)

            # Verify GaussianBlur was called with odd kernel size (intensity + 1)
            args, _ = mock_blur.call_args
            assert args[1] == (
                31,
                31,
            )  # Should be (intensity+1, intensity+1) since 30 is even
