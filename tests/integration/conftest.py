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
from unittest.mock import MagicMock

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
    """
    # Create a known faces directory for testing
    known_faces_dir = os.path.join(test_data_dir, "known_faces")
    os.makedirs(known_faces_dir, exist_ok=True)
    
    # Look for real face images in the test_data_dir
    reference_face_path = os.path.join(test_data_dir, "reference_face.jpg")
    if not os.path.exists(reference_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Reference face image not found. Please add a reference face image at: " + reference_face_path + "\nThis image is required for integration tests.")
    
    # Copy the reference face to the known faces directory
    known_face_path = os.path.join(known_faces_dir, "test_person.jpg")
    shutil.copy2(reference_face_path, known_face_path)
    
    # Create the components
    detector = FaceDetector()
    matcher = FaceMatcher(known_faces_dir=known_faces_dir)
    
    # Use another variant of the reference face for testing
    test_face_path = os.path.join(test_data_dir, "test_face.jpg")
    if not os.path.exists(test_face_path):
        pytest.fail("REQUIRED TEST IMAGE MISSING: Test face image not found. Please add a test face image at: " + test_face_path + "\nThis image is required for integration tests.")
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration.jpg")
    shutil.copy2(test_face_path, test_image_path)
    
    return (detector, matcher, test_image_path)

# Detection-anonymization pipeline fixture
@pytest.fixture
def detection_anonymization_pipeline(test_data_dir):
    """
    Create a complete pipeline for detection -> anonymization.
    
    Returns a tuple of (detector, anonymizer, test_image_path).
    """
    # Create the components
    detector = FaceDetector()
    anonymizer = FaceAnonymizer()
    
    # Use a real face image for anonymization testing
    anon_face_path = os.path.join(test_data_dir, "anon_face.jpg")
    if not os.path.exists(anon_face_path):
        # Fall back to using the same test face if available
        test_face_path = os.path.join(test_data_dir, "test_face.jpg")
        if not os.path.exists(test_face_path):
            pytest.fail("REQUIRED TEST IMAGES MISSING: Neither anon_face.jpg nor test_face.jpg found in test data directory.\nAt least one of these images is required for anonymization tests.")
        anon_face_path = test_face_path
        
    # Copy to the test location
    test_image_path = os.path.join(test_data_dir, "test_integration_anon.jpg")
    shutil.copy2(anon_face_path, test_image_path)
    
    return (detector, anonymizer, test_image_path)
