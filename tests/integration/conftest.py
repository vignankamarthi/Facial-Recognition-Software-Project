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
    
    # Look for real face images in the test_data_dir
    reference_face_path = os.path.join(test_data_dir, "reference_face.jpg")
    # Generate a synthetic face if needed and in CI environment
    if not os.path.exists(reference_face_path) and is_ci_environment():
        # Create a simple face-like image for testing
        face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(face_img, (100, 100), 80, (255, 255, 255), -1)  # Face shape
        cv2.circle(face_img, (70, 80), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(face_img, (130, 80), 15, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(face_img, (100, 120), (30, 20), 0, 0, 180, (0, 0, 0), -1)  # Mouth
        os.makedirs(os.path.dirname(reference_face_path), exist_ok=True)
        cv2.imwrite(reference_face_path, face_img)
    elif not os.path.exists(reference_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Reference face image not found. Please add a reference face image at: " + reference_face_path + "\nThis image is required for integration tests.")
    
    # Copy the reference face to the known faces directory
    known_face_path = os.path.join(known_faces_dir, "test_person.jpg")
    shutil.copy2(reference_face_path, known_face_path)
    
    # Create the components
    detector = FaceDetector()
    matcher = FaceMatcher(known_faces_dir=known_faces_dir)
    
    # Use another variant of the reference face for testing
    test_face_path = os.path.join(test_data_dir, "test_face.jpg")
    # Generate a synthetic face if needed and in CI environment
    if not os.path.exists(test_face_path) and is_ci_environment():
        # Create a variant of the face for testing
        face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(face_img, (100, 100), 80, (255, 255, 255), -1)  # Face shape
        cv2.circle(face_img, (70, 80), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(face_img, (130, 80), 15, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(face_img, (100, 130), (40, 20), 0, 0, 180, (0, 0, 0), -1)  # Different mouth
        os.makedirs(os.path.dirname(test_face_path), exist_ok=True)
        cv2.imwrite(test_face_path, face_img)
    elif not os.path.exists(test_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Test face image not found. Please add a test face image at: " + test_face_path + "\nThis image is required for integration tests.")
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration.jpg")
    shutil.copy2(test_face_path, test_image_path)
    
    # Set up necessary mocks for headless environments
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
        
        # Start all patches
        for p in cv2_patches:
            p.start()
            
        # Yield the components
        yield (detector, matcher, test_image_path)
        
        # Stop all patches
        for p in cv2_patches:
            p.stop()
    else:
        # In normal environments, just return the components
        yield (detector, matcher, test_image_path)

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
    
    # Use a real face image for anonymization testing
    anon_face_path = os.path.join(test_data_dir, "anon_face.jpg")
    # Generate a synthetic face if needed and in CI environment
    if not os.path.exists(anon_face_path) and is_ci_environment():
        # Create a simple face-like image for testing
        face_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(face_img, (100, 100), 80, (255, 255, 255), -1)  # Face shape
        cv2.circle(face_img, (70, 80), 15, (0, 0, 0), -1)  # Left eye
        cv2.circle(face_img, (130, 80), 15, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(face_img, (100, 120), (30, 20), 0, 0, 180, (0, 0, 0), -1)  # Mouth
        os.makedirs(os.path.dirname(anon_face_path), exist_ok=True)
        cv2.imwrite(anon_face_path, face_img)
    elif not os.path.exists(anon_face_path):
        # Fall back to using the same test face if available
        test_face_path = os.path.join(test_data_dir, "test_face.jpg")
        if not os.path.exists(test_face_path) and is_ci_environment():
            # Create a simple face-like image for testing
            face_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.circle(face_img, (100, 100), 80, (255, 255, 255), -1)  # Face shape
            cv2.circle(face_img, (70, 80), 15, (0, 0, 0), -1)  # Left eye
            cv2.circle(face_img, (130, 80), 15, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(face_img, (100, 120), (30, 20), 0, 0, 180, (0, 0, 0), -1)  # Mouth
            os.makedirs(os.path.dirname(test_face_path), exist_ok=True)
            cv2.imwrite(test_face_path, face_img)
        elif not os.path.exists(test_face_path):
            pytest.fail("REQUIRED TEST IMAGES MISSING: Neither anon_face.jpg nor test_face.jpg found in test data directory.\nAt least one of these images is required for anonymization tests.")
        anon_face_path = test_face_path
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration_anon.jpg")
    shutil.copy2(anon_face_path, test_image_path)
    
    # Set up necessary mocks for headless environments
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
        
        # Start all patches
        for p in cv2_patches:
            p.start()
            
        # Yield the components
        yield (detector, anonymizer, test_image_path)
        
        # Stop all patches
        for p in cv2_patches:
            p.stop()
    else:
        # In normal environments, just return the components
        yield (detector, anonymizer, test_image_path)
