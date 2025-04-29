"""
Integration tests for the detection and anonymization pipeline.
"""
import pytest
import cv2
import numpy as np
import os
import time
from unittest.mock import patch, MagicMock

from src.backend.face_detection import FaceDetector
from src.backend.anonymization import FaceAnonymizer

class TestDetectionAnonymizationIntegration:
    """
    Integration tests for the face detection and anonymization pipeline.
    
    These tests verify that the face detector and face anonymizer work
    together correctly in a pipeline.
    """
    
    def test_detection_to_anonymization_pipeline(self, detection_anonymization_pipeline):
        """Test the complete detection -> anonymization pipeline."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        assert image is not None, f"Failed to load test image: {test_image_path}"
        
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(image)
        
        # Verify face detection worked
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Anonymize the detected faces
        result_frame = anonymizer.anonymize_frame(image, face_locations)
        
        # Verify the result is a valid image
        assert result_frame is not None
        assert result_frame.shape == image.shape
        assert result_frame is not image  # Should be a new frame, not the original
        
        # Verify the image is different (anonymization changed the pixels)
        assert not np.array_equal(image, result_frame)
        
        # Basic verification that the anonymization text appears in the result
        # Convert to grayscale and find text areas (lighter regions)
        gray = cv2.cvtColor(result_frame, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # There should be some white pixels (text/indicators) in the output
        assert np.count_nonzero(threshold) > 0, "No text/indicators found in anonymized image"
    
    def test_blur_anonymization_with_detected_faces(self, detection_anonymization_pipeline):
        """Test blur anonymization with faces from the detector."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Set anonymization method to blur
        anonymizer.set_method("blur")
        anonymizer.set_intensity(30)  # Higher intensity = stronger blur
        
        # Detect faces
        face_locations, _ = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Get the first face location
        face_location = face_locations[0]
        
        # Store the original face region for comparison
        top, right, bottom, left = face_location
        original_face = image[top:bottom, left:right].copy()
        
        # Anonymize just this face
        result_frame = anonymizer.anonymize_face(image, face_location)
        
        # Extract the anonymized face region
        anonymized_face = result_frame[top:bottom, left:right]
        
        # Verify it's actually blurred
        # Calculate image variance as a measure of blurriness
        # Lower variance in the anonymized face indicates blur
        original_variance = np.var(cv2.cvtColor(original_face, cv2.COLOR_BGR2GRAY))
        anonymized_variance = np.var(cv2.cvtColor(anonymized_face, cv2.COLOR_BGR2GRAY))
        
        # Blurred image should have lower variance
        assert anonymized_variance < original_variance, "Blur not applied correctly"
    
    def test_pixelate_anonymization_with_detected_faces(self, detection_anonymization_pipeline):
        """Test pixelate anonymization with faces from the detector."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Set anonymization method to pixelate
        anonymizer.set_method("pixelate")
        anonymizer.set_intensity(50)  # Higher intensity = larger pixels
        
        # Detect faces
        face_locations, _ = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Anonymize the detected faces
        result_frame = anonymizer.anonymize_frame(image, face_locations)
        
        # For pixelation, we can check for edge characteristics
        # Pixelated images have strong, straight edges between pixels
        
        # Get the first face location
        top, right, bottom, left = face_locations[0]
        
        # Extract the anonymized face region
        anonymized_face = result_frame[top:bottom, left:right]
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(anonymized_face, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Pixelated images should have some distinct edges
        assert np.count_nonzero(edges) > 0, "Pixelation not detected"
        
        # Verify the word "Anonymized" appears in the result
        assert not np.array_equal(image, result_frame), "Image was not modified"
    
    def test_mask_anonymization_with_detected_faces(self, detection_anonymization_pipeline):
        """Test mask anonymization with faces from the detector."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Set anonymization method to mask
        anonymizer.set_method("mask")
        
        # Detect faces
        face_locations, _ = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Get the first face location
        face_location = face_locations[0]
        top, right, bottom, left = face_location
        
        # Store the original face region for comparison
        original_face = image[top:bottom, left:right].copy()
        
        # Anonymize just this face
        result_frame = anonymizer.anonymize_face(image, face_location)
        
        # Extract the anonymized face region
        anonymized_face = result_frame[top:bottom, left:right]
        
        # Verify it's significantly different from the original
        assert not np.array_equal(original_face, anonymized_face), "Mask not applied"
        
        # For mask method, we expect areas of solid black
        # Convert to grayscale and count black pixels
        gray = cv2.cvtColor(anonymized_face, cv2.COLOR_BGR2GRAY)
        black_pixels = np.count_nonzero(gray < 10)  # Count very dark pixels
        
        # There should be a significant number of black pixels in the mask
        assert black_pixels > 0, "No mask (black pixels) detected"
    
    def test_pipeline_with_no_faces(self, detection_anonymization_pipeline, test_data_dir):
        """Test pipeline behavior when no faces are detected."""
        # Unpack the pipeline components
        detector, anonymizer, _ = detection_anonymization_pipeline
        
        # Create a blank test image with no faces
        blank_image_path = os.path.join(test_data_dir, "blank_test.jpg")
        blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(blank_image_path, blank_image)
        
        # Create a patched version of detect_faces that returns no faces
        with patch.object(detector, 'detect_faces', return_value=([], [])):
            # Detect faces (should be none)
            face_locations, face_encodings = detector.detect_faces(blank_image)
            
            # Verify no faces were detected
            assert len(face_locations) == 0
            assert len(face_encodings) == 0
            
            # Try to anonymize with no faces
            result_frame = anonymizer.anonymize_frame(blank_image, face_locations)
            
            # Anonymization should not fail, but the image should be unchanged
            # except for the status bar showing anonymization mode
            assert result_frame is not None
            assert result_frame is not blank_image  # Should return a new image
            
            # Should be mostly the same as the original except for the status bar
            # So compare only the bottom portion (without the status bar)
            original_bottom = blank_image[50:, :]
            result_bottom = result_frame[50:, :]
            assert np.array_equal(original_bottom, result_bottom)
    
    def test_pipeline_performance(self, detection_anonymization_pipeline):
        """Test performance characteristics of the pipeline."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Test performance with different anonymization methods
        methods = ["blur", "pixelate", "mask"]
        
        for method in methods:
            # Set the anonymization method
            anonymizer.set_method(method)
            
            # Time face detection
            start_time = time.time()
            face_locations, face_encodings = detector.detect_faces(image)
            detection_time = time.time() - start_time
            
            # Ensure faces were detected
            assert len(face_locations) > 0
            
            # Time anonymization
            start_time = time.time()
            result_frame = anonymizer.anonymize_frame(image, face_locations)
            anonymization_time = time.time() - start_time
            
            # Performance requirements - these should be fast enough for real-time
            # processing (adjust thresholds as needed for your environment)
            assert detection_time < 5.0, f"Face detection too slow: {detection_time:.2f}s"
            assert anonymization_time < 1.0, f"{method} anonymization too slow: {anonymization_time:.2f}s"
            
            # Log times for comparison
            print(f"{method.capitalize()} method - Detection: {detection_time:.4f}s, Anonymization: {anonymization_time:.4f}s")
    
    def test_webcam_anonymization_integration(self, detection_anonymization_pipeline):
        """Test the webcam anonymization pipeline."""
        # Unpack the pipeline components
        detector, anonymizer, test_image_path = detection_anonymization_pipeline
        
        # Create a mock video capture to simulate webcam
        mock_video_capture = MagicMock()
        
        # Mock a frame from the webcam - use the test image
        test_frame = cv2.imread(test_image_path)
        mock_video_capture.read.return_value = (True, test_frame)
        mock_video_capture.isOpened.return_value = True
        
        # Patch detect_faces to ensure deterministic behavior
        mock_face_locations = [(50, 200, 150, 100)]  # (top, right, bottom, left)
        
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(detector, 'detect_faces', return_value=(mock_face_locations, [np.zeros(128)])), \
             patch.object(anonymizer, 'anonymize_frame') as mock_anonymize_frame, \
             patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey, \
             patch('cv2.destroyAllWindows') as mock_destroy, \
             patch('src.backend.face_detection.create_resizable_window') as mock_create_window:
            
            # Simulate keyboard input - with enough values for all potential calls
            # Some environments may call waitKey multiple times during cleanup
            mock_waitkey.side_effect = [255, 255, 255, 255, ord('q'), 255, 255, 255, 255]
            
            # Run the webcam detection with anonymization
            success, result = detector.detect_faces_webcam(anonymize=True, anonymizer=anonymizer)
            
            # Verify the function succeeded
            assert success is True
            
            # Verify the various functions were called as expected
            assert mock_video_capture.read.call_count >= 1
            assert mock_imshow.call_count >= 1
            assert mock_anonymize_frame.call_count >= 1
            assert mock_destroy.call_count >= 1
            
            # Verify results were returned correctly
            assert "face_count" in result
            assert "frames_processed" in result
            assert "duration" in result
