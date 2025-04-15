#!/usr/bin/env python3
"""
Test script to check if imports are working correctly.
This script attempts to import all the main modules to verify that 
circular dependencies are resolved.
"""

import os
import sys

# Add the project root to path to ensure imports work
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

print("Testing imports...")

# Try importing modules one by one
try:
    print("Importing utilities.common_utils...")
    from src.utilities.common_utils import safely_close_windows
    print("✓ Successfully imported utilities.common_utils")
    
    print("Importing utilities.config...")
    from src.utilities.config import WINDOW_NAME
    print("✓ Successfully imported utilities.config")
    
    print("Importing utilities.image_processing...")
    from src.utilities.image_processing import ImageProcessor
    print("✓ Successfully imported utilities.image_processing")
    
    print("Importing facial_recognition_software.face_detection...")
    from src.facial_recognition_software.face_detection import FaceDetector
    print("✓ Successfully imported facial_recognition_software.face_detection")
    
    print("Importing facial_recognition_software.face_matching...")
    from src.facial_recognition_software.face_matching import FaceMatcher
    print("✓ Successfully imported facial_recognition_software.face_matching")
    
    print("Importing facial_recognition_software.anonymization...")
    from src.facial_recognition_software.anonymization import FaceAnonymizer
    print("✓ Successfully imported facial_recognition_software.anonymization")
    
    print("\nAll imports successful!")
except ImportError as e:
    print(f"\nImport error: {e}")
    import traceback
    traceback.print_exc()
    print("\nFailed to import all modules.")
except Exception as e:
    print(f"\nUnexpected error: {e}")
    import traceback
    traceback.print_exc()
    print("\nFailed to import all modules.")
