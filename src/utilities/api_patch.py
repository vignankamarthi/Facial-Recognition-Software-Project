"""
API Patch Module

This module is a wrapper for the face_recognition patching utility.
It modifies the face_recognition module to continue even if face_recognition_models 
can't be imported.

# This script relies on a centralized utility function to patch the face_recognition API.
# If you encounter issues with face detection, you might need to review this patch.
"""

from .face_recognition_patch import patch_face_recognition

# Apply the patch when this module is imported
success = patch_face_recognition()

if not success:
    print("WARNING: Could not apply face_recognition patch")
