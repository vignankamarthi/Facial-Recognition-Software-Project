#!/usr/bin/env python3
"""
Run script that ensures all modules are properly set up before running the main application.

This script was added to solve dependency issues with face_recognition and face_recognition_models.
It automatically checks for dependencies and applies patches if needed.
"""

import os
import sys
from .face_recognition_patch import patch_face_recognition

# Check for face_recognition_models
try:
    import face_recognition_models
    print("face_recognition_models package is installed correctly.")
except ImportError:
    print("Installing face_recognition_models...")
    os.system("pip install git+https://github.com/ageitgey/face_recognition_models")
    try:
        import face_recognition_models
        print("Successfully installed face_recognition_models.")
    except ImportError:
        print("Failed to install face_recognition_models. Please install manually.")
        print("pip install git+https://github.com/ageitgey/face_recognition_models")
        sys.exit(1)

# Check for face_recognition
try:
    import face_recognition
    print("face_recognition package is installed correctly.")
except ImportError:
    print("face_recognition not found. Installing...")
    os.system("pip install face_recognition")
    try:
        import face_recognition
        print("Successfully installed face_recognition.")
    except ImportError:
        print("Failed to install face_recognition. Please install manually.")
        sys.exit(1)

# Monkey patch the face_recognition module if needed
try:
    # Test if the module works correctly
    face_recognition.load_image_file
    print("face_recognition functions are working correctly.")
except Exception as e:
    print(f"Error with face_recognition: {e}")
    print("Attempting to patch the face_recognition module...")
    
    # Apply the patch
    success = patch_face_recognition()
    
    if success:
        # Reload the module
        if 'face_recognition' in sys.modules:
            del sys.modules['face_recognition']
        import face_recognition
        print("Successfully reloaded face_recognition after patching.")

# Now run the main application
print("\nStarting the Facial Recognition System...\n")
import main
