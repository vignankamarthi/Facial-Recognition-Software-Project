"""
API Patch Module

This module provides a simplified interface for the face_recognition patching utility.
It automatically verifies the face_recognition module is working properly when imported
and applies patches if needed.

Usage:
    # Simply import this module at the start of any script that uses face_recognition
    import src.utils.api_patch
"""

from .face_recognition_patch import verify_face_recognition

# Verify face_recognition is working when this module is imported
success = verify_face_recognition()

if not success:
    print("WARNING: Could not verify face_recognition functionality")
    print("Some face detection features may not work correctly")
else:
    print("Face recognition API verified and ready")
