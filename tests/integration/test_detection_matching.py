"""
Integration tests for the detection and matching pipeline.
"""
import pytest
import cv2
import numpy as np
import os
import time
import face_recognition
from unittest.mock import patch, MagicMock

from src.backend.face_detection import FaceDetector
from src.backend.face_matching import FaceMatcher

class TestDetectionMatchingIntegration:
    """
    Integration tests for the face detection and matching pipeline.
    
    These tests verify that the face detector and face matcher work
    together correctly in a pipeline.
    """
    
    def test_detection_to_matching_pipeline(self, detection_matching_pipeline):
        """Test the complete detection -> matching pipeline."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        assert image is not None, f"Failed to load test image: {test_image_path}"
        
        # Verify the matcher has loaded known faces
        assert len(matcher.known_face_encodings) > 0, "No known faces loaded"
        assert len(matcher.known_face_names) > 0, "No known face names loaded"
        
        # Detect faces in the test image
        face_locations, face_encodings = detector.detect_faces(image)
        
        # Verify face detection worked
        assert len(face_locations) > 0, "No faces detected in test image"
        assert len(face_encodings) > 0, "No face encodings generated"
        assert len(face_locations) == len(face_encodings), "Mismatch between locations and encodings"
        
        # Match the detected faces against known faces
        result_frame, face_names = matcher.identify_faces(image, face_locations, face_encodings)
        
        # Verify the result is a valid image
        assert result_frame is not None
        assert result_frame.shape == image.shape
        assert result_frame is not image  # Should be a new frame, not the original
        
        # Verify names were generated for each face
        assert len(face_names) == len(face_locations)
        assert all(isinstance(name, str) for name in face_names)
        
        # Check that names include either known names or "Unknown"
        valid_names = set(matcher.known_face_names + ["Unknown"])
        for name in face_names:
            # Check if name contains any of the known names
            # (it might include confidence score in parentheses)
            assert any(known_name in name for known_name in valid_names), f"Invalid name: {name}"
    
    def test_matching_with_detected_faces(self, detection_matching_pipeline):
        """Test matching with faces from the detector."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Get the original known face encodings
        original_encodings = matcher.known_face_encodings.copy()
        original_names = matcher.known_face_names.copy()
        
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Test with exact match scenario (detected face is in known faces)
        # Replace the known faces with the detected face to force a match
        matcher.known_face_encodings = face_encodings.copy()
        matcher.known_face_names = ["Test Person"] * len(face_encodings)
        
        # Match the faces
        result_frame, face_names = matcher.identify_faces(image, face_locations, face_encodings)
        
        # Verify each face was matched correctly
        for name in face_names:
            assert "Test Person" in name, f"Face should match exactly: {name}"
            assert "(" in name and ")" in name, "Name should include confidence score"
            
            # Parse the confidence score
            confidence_str = name.split("(")[1].split(")")[0]
            confidence = float(confidence_str)
            
            # Confidence should be very high for exact match
            assert confidence > 0.9, f"Confidence too low for exact match: {confidence}"
        
        # Restore original known faces
        matcher.known_face_encodings = original_encodings
        matcher.known_face_names = original_names
    
    def test_pipeline_with_no_faces(self, detection_matching_pipeline, test_data_dir):
        """Test pipeline behavior when no faces are detected."""
        # Unpack the pipeline components
        detector, matcher, _ = detection_matching_pipeline
        
        # Create a blank test image with no faces
        blank_image_path = os.path.join(test_data_dir, "blank_test.jpg")
        blank_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.imwrite(blank_image_path, blank_image)
        
        # Detect faces (should find none)
        with patch.object(face_recognition, 'face_locations', return_value=[]):
            face_locations, face_encodings = detector.detect_faces(blank_image)
            
            # Verify no faces were detected
            assert len(face_locations) == 0
            assert len(face_encodings) == 0
            
            # Try to match with no faces
            result_frame, face_names = matcher.identify_faces(blank_image, face_locations, face_encodings)
            
            # Verify the result is a valid image
            assert result_frame is not None
            assert result_frame.shape == blank_image.shape
            
            # Verify no names were returned
            assert len(face_names) == 0
    
    def test_pipeline_with_unknown_faces(self, detection_matching_pipeline, test_data_dir):
        """Test pipeline with faces that don't match known references."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Empty the known faces list to ensure no matches
        matcher.known_face_encodings = []
        matcher.known_face_names = []
        
        # Match the faces
        result_frame, face_names = matcher.identify_faces(image, face_locations, face_encodings)
        
        # Verify all faces are marked as unknown
        assert len(face_names) == len(face_locations)
        assert all(name == "Unknown" for name in face_names), "Faces should all be unknown"
        
        # Now add a reference face that's very different from test face
        # to ensure it doesn't match
        opposite_encoding = np.ones_like(face_encodings[0]) * -1  # Opposite values
        matcher.known_face_encodings = [opposite_encoding]
        matcher.known_face_names = ["Not Matching Person"]
        
        # Patch compare_faces and face_distance to ensure deterministic behavior
        with patch('face_recognition.compare_faces', return_value=[False]), \
             patch('face_recognition.face_distance', return_value=np.array([0.9])):  # High distance = poor match
            
            # Match the faces
            result_frame, face_names = matcher.identify_faces(image, face_locations, face_encodings)
            
            # Verify all faces are still marked as unknown due to high threshold
            assert len(face_names) == len(face_locations)
            assert all(name == "Unknown" for name in face_names), "Faces should not match with high distance"
    
    def test_pipeline_performance(self, detection_matching_pipeline):
        """Test performance characteristics of the pipeline."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Time face detection
        start_time = time.time()
        face_locations, face_encodings = detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        # Ensure faces were detected
        assert len(face_locations) > 0
        
        # Time face matching
        start_time = time.time()
        result_frame, face_names = matcher.identify_faces(image, face_locations, face_encodings)
        matching_time = time.time() - start_time
        
        # Total pipeline time
        pipeline_time = detection_time + matching_time
        
        # Performance requirements - these should be fast enough for interactive use
        # Adjust thresholds as needed for your environment
        assert detection_time < 5.0, f"Face detection too slow: {detection_time:.2f}s"
        assert matching_time < 1.0, f"Face matching too slow: {matching_time:.2f}s"
        assert pipeline_time < 6.0, f"Total pipeline too slow: {pipeline_time:.2f}s"
        
        # Log performance metrics
        print(f"Detection time: {detection_time:.4f}s")
        print(f"Matching time: {matching_time:.4f}s")
        print(f"Total pipeline time: {pipeline_time:.4f}s")
    
    def test_threshold_effect_on_matching(self, detection_matching_pipeline):
        """Test how different matching thresholds affect face identification."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Load the test image
        image = cv2.imread(test_image_path)
        
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(image)
        assert len(face_locations) > 0, "No faces detected in test image"
        
        # Get a copy of the first face encoding (assuming at least one face detected)
        face_encoding = face_encodings[0].copy()
        
        # Create similar encodings with increasing distance
        similar_encodings = []
        names = []
        
        # Create 5 variations with increasing difference
        for i in range(5):
            # Create a variation by adding random noise
            noise_level = i * 0.1  # 0.0, 0.1, 0.2, 0.3, 0.4
            variation = face_encoding + np.random.randn(128) * noise_level
            
            # Normalize the encoding (face_recognition encodings are normalized)
            variation = variation / np.linalg.norm(variation)
            
            similar_encodings.append(variation)
            names.append(f"Person_{i}")
        
        # Replace matcher's known faces with our variations
        matcher.known_face_encodings = similar_encodings
        matcher.known_face_names = names
        
        # Test with three different threshold levels
        thresholds = [0.8, 0.6, 0.4]  # Strict to lenient
        matches_count = []
        
        for threshold in thresholds:
            # Patch the threshold
            with patch('src.backend.face_matching.FACE_MATCHING_THRESHOLD', threshold):
                # Match the faces
                _, face_names = matcher.identify_faces(image, face_locations[:1], face_encodings[:1])
                
                # Count matches (non-"Unknown" results)
                match_count = sum(1 for name in face_names if "Unknown" not in name)
                matches_count.append(match_count)
                
                print(f"Threshold {threshold}: {match_count} matches")
        
        # More lenient thresholds should result in more matches
        assert matches_count[2] >= matches_count[1] >= matches_count[0], "Lenient thresholds should match more faces"
    
    def test_webcam_matching_integration(self, detection_matching_pipeline):
        """Test the webcam matching interface."""
        # Unpack the pipeline components
        detector, matcher, test_image_path = detection_matching_pipeline
        
        # Create a mock video capture to simulate webcam
        mock_video_capture = MagicMock()
        
        # Mock a frame from the webcam - use the test image
        test_frame = cv2.imread(test_image_path)
        mock_video_capture.read.return_value = (True, test_frame)
        mock_video_capture.isOpened.return_value = True
        
        # Patch required functions
        with patch('cv2.VideoCapture', return_value=mock_video_capture), \
             patch.object(detector.__class__, 'detect_faces', return_value=([(50, 200, 150, 100)], [np.ones(128)])), \
             patch.object(matcher, 'identify_faces') as mock_identify, \
             patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey, \
             patch('cv2.destroyAllWindows') as mock_destroy, \
             patch('src.backend.face_matching.create_resizable_window') as mock_create_window:
            
            # Configure mock_identify to return a valid result
            mock_identify.return_value = (test_frame.copy(), ["Test Person (0.95)"])
            
            # Simulate keyboard input to exit after one frame
            mock_waitkey.return_value = ord('q')
            
            # Run the webcam matching
            matcher.match_faces_webcam()
            
            # Verify the various functions were called as expected
            mock_video_capture.read.assert_called()
            assert detector.__class__.detect_faces.call_count >= 1
            assert mock_identify.call_count >= 1
            assert mock_imshow.call_count >= 1
            
            # Import the utility function to check environment
            from src.utils.environment_utils import is_headless_environment
            
            # In a headless environment, destroy might not be called
            if not is_headless_environment():
                assert mock_destroy.call_count >= 1
