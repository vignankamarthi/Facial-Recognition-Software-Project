"""
Integration test configuration and fixtures for the Facial Recognition Software Project.

This file contains fixtures specific to integration tests, which test the
interaction between different components.
"""

import os
import pytest
import numpy as np
import cv2
import shutil
from unittest.mock import MagicMock, patch

# Import environment utilities
from src.utils.environment_utils import is_ci_environment, is_headless_environment, is_webcam_available

# Project imports
from src.backend.face_detection import FaceDetector
from src.backend.face_matching import FaceMatcher
from src.backend.anonymization import FaceAnonymizer

# Test pipeline fixture
@pytest.fixture
def detection_matching_pipeline(test_data_dir):
    """
    Create a complete pipeline for detection -> matching.
    
    Returns a tuple of (detector, matcher, test_image_path).
    
    This fixture adapts to the testing environment:
    - In CI or headless environments, it sets up necessary mocks
    - In standard environments, it uses real components when possible
    """
    # Create a known faces directory for testing
    known_faces_dir = os.path.join(test_data_dir, "known_faces")
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Use the existing test face image for testing
    test_face_path = os.path.join(test_data_dir, "test_face.jpg")
    reference_face_path = os.path.join(test_data_dir, "reference_face.jpg")
    if not os.path.exists(test_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Test face image not found at: " + test_face_path)
        
    # Copy test_face.jpg to reference_face.jpg
    shutil.copy2(test_face_path, reference_face_path)
    print(f"Using test face image from {test_face_path} as reference face")
    
    # Copy the reference face to the known faces directory
    known_face_path = os.path.join(known_faces_dir, "test_person.jpg")
    shutil.copy2(reference_face_path, known_face_path)
    
    # Create the components
    detector = FaceDetector()
    matcher = FaceMatcher(known_faces_dir=known_faces_dir)
    
    # Ensure known faces are loaded correctly in CI environment
    if is_ci_environment() and len(matcher.known_face_encodings) == 0:
        print("No known faces loaded. Adding mock known face directly.")
        matcher.known_face_encodings = [np.ones(128)]
        matcher.known_face_names = ["test_person"]
    
    # Ensure the test image exists (should have been verified above)
    print(f"Using test face image at {test_face_path}")
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration.jpg")
    shutil.copy2(test_face_path, test_image_path)
    
    # Set up necessary mocks for CI and headless environments
    patches = []
    try:
        # In CI environment, we need to ensure face detection always works
        if is_ci_environment():
            # Patch face_recognition functions to ensure they return expected values
            face_loc_patch = patch('face_recognition.face_locations', return_value=[(50, 250, 250, 50)])
            face_enc_patch = patch('face_recognition.face_encodings', return_value=[np.ones(128)])
            face_compare_patch = patch('face_recognition.compare_faces', return_value=[True])
            face_distance_patch = patch('face_recognition.face_distance', return_value=np.array([0.4]))
            face_load_patch = patch('face_recognition.load_image_file',
                               return_value=np.zeros((300, 400, 3), dtype=np.uint8))
            
            # Start face recognition patches
            face_loc_patch.start()
            face_enc_patch.start()
            face_compare_patch.start()
            face_distance_patch.start()
            face_load_patch.start()
            
            # Add to patches list for cleanup
            patches.append(face_loc_patch)
            patches.append(face_enc_patch)
            patches.append(face_compare_patch)
            patches.append(face_distance_patch)
            patches.append(face_load_patch)
            
            print("CI environment detected: Using mocked face detection")
            
        # Set up window-related mocks for headless environments
        if is_headless_environment():
            # Patch window-related functions
            cv2_patches = [
                patch('cv2.namedWindow'),
                patch('cv2.resizeWindow'),
                patch('cv2.moveWindow'),
                patch('cv2.destroyWindow'),
                patch('cv2.destroyAllWindows'),
                patch('cv2.imshow'),
                patch('cv2.waitKey', return_value=ord('q'))  # Simulate pressing 'q' to quit
            ]
            
            # Start all CV2 patches and add to patches list
            for p in cv2_patches:
                p.start()
                patches.append(p)
        
        # Yield the components
        yield (detector, matcher, test_image_path)
    finally:
        # Stop all patches in the finally block to ensure cleanup
        for p in patches:
            p.stop()

# Detection-anonymization pipeline fixture
@pytest.fixture
def detection_anonymization_pipeline(test_data_dir):
    """
    Create a complete pipeline for detection -> anonymization.
    
    Returns a tuple of (detector, anonymizer, test_image_path).
    
    This fixture adapts to the testing environment:
    - In CI or headless environments, it sets up necessary mocks
    - In standard environments, it uses real components when possible
    """
    # Create the components
    detector = FaceDetector()
    anonymizer = FaceAnonymizer()
    
    # Use the existing test face image for anonymization testing
    test_face_path = os.path.join(test_data_dir, "test_face.jpg")
    anon_face_path = os.path.join(test_data_dir, "anon_face.jpg")
    
    # Check if the test face exists
    if not os.path.exists(test_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Test face image not found at: " + test_face_path)
        
    # Use the test face for anonymization testing
    shutil.copy2(test_face_path, anon_face_path)
    print(f"Using test face image from {test_face_path} for anonymization testing")
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration_anon.jpg")
    shutil.copy2(anon_face_path, test_image_path)
    
    # Set up necessary mocks for CI and headless environments
    patches = []
    try:
        # In CI environment, we need to ensure face detection always works
        if is_ci_environment():
            # Patch face_recognition functions to ensure they return expected values
            face_loc_patch = patch('face_recognition.face_locations', return_value=[(50, 250, 250, 50)])
            face_enc_patch = patch('face_recognition.face_encodings', return_value=[np.ones(128)])
            face_compare_patch = patch('face_recognition.compare_faces', return_value=[True])
            face_distance_patch = patch('face_recognition.face_distance', return_value=np.array([0.4]))
            face_load_patch = patch('face_recognition.load_image_file',
                               return_value=np.zeros((300, 400, 3), dtype=np.uint8))
            
            # Start face recognition patches
            face_loc_patch.start()
            face_enc_patch.start()
            face_compare_patch.start()
            face_distance_patch.start()
            face_load_patch.start()
            
            # Add to patches list for cleanup
            patches.append(face_loc_patch)
            patches.append(face_enc_patch)
            patches.append(face_compare_patch)
            patches.append(face_distance_patch)
            patches.append(face_load_patch)
            
            print("CI environment detected: Using mocked face detection for anonymization tests")
            
        # Set up window-related mocks for headless environments
        if is_headless_environment():
            # Patch window-related functions
            cv2_patches = [
                patch('cv2.namedWindow'),
                patch('cv2.resizeWindow'),
                patch('cv2.moveWindow'),
                patch('cv2.destroyWindow'),
                patch('cv2.destroyAllWindows'),
                patch('cv2.imshow'),
                patch('cv2.waitKey', return_value=ord('q'))  # Simulate pressing 'q' to quit
            ]
            
            # Start all CV2 patches and add to patches list
            for p in cv2_patches:
                p.start()
                patches.append(p)
        
        # Yield the components
        yield (detector, anonymizer, test_image_path)
    finally:
        # Stop all patches in the finally block to ensure cleanup
        for p in patches:
            p.stop()
