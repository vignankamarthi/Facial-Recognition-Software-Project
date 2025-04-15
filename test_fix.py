#!/usr/bin/env python3
"""
Test script for fixing the circular import issues.
This script tests the lazy loading mechanism we implemented.
"""

import os
import sys
import traceback

# Add the project root to the Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "src"))

print("Testing imports after circular dependency fix...")

try:
    # Import directly from facial_recognition_software package
    print("\nTesting imports from facial_recognition_software:")
    import facial_recognition_software
    print("✓ Successfully imported facial_recognition_software package")
    
    # Import classes through the lazy loader
    from facial_recognition_software import FaceDetector
    print("✓ Successfully imported FaceDetector")
    
    from facial_recognition_software import FaceMatcher
    print("✓ Successfully imported FaceMatcher")
    
    from facial_recognition_software import FaceAnonymizer
    print("✓ Successfully imported FaceAnonymizer")
    
    from facial_recognition_software import BiasAnalyzer
    print("✓ Successfully imported BiasAnalyzer")
    
    # Import from utilities
    print("\nTesting imports from utilities:")
    import utilities
    print("✓ Successfully imported utilities package")
    
    from utilities import ImageProcessor
    print("✓ Successfully imported ImageProcessor")
    
    # Create instances to verify full initialization
    print("\nTesting class instantiation:")
    detector = FaceDetector()
    print("✓ Successfully instantiated FaceDetector")
    
    matcher = FaceMatcher()
    print("✓ Successfully instantiated FaceMatcher")
    
    anonymizer = FaceAnonymizer()
    print("✓ Successfully instantiated FaceAnonymizer")
    
    analyzer = BiasAnalyzer()
    print("✓ Successfully instantiated BiasAnalyzer")
    
    processor = ImageProcessor()
    print("✓ Successfully instantiated ImageProcessor")
    
    print("\nAll import tests passed successfully!")
    
except ImportError as e:
    print(f"\nImport Error: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\nUnexpected Error: {e}")
    traceback.print_exc()
    sys.exit(1)
