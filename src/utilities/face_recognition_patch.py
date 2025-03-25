"""
Utility module for patching face_recognition imports.

This module provides a centralized solution for patching the face_recognition
module to continue functioning even when face_recognition_models can't be
properly imported.
"""

import os
import sys


def patch_face_recognition():
    """
    Apply monkey patch to face_recognition module to handle missing face_recognition_models.
    
    This function modifies the face_recognition API file to bypass the import check
    that would otherwise terminate the program if face_recognition_models is not found.
    
    Returns:
        bool: True if patching was successful, False otherwise
    """
    # Find the face_recognition module's api.py file
    face_rec_path = None
    for path in sys.path:
        api_path = os.path.join(path, "face_recognition", "api.py")
        if os.path.exists(api_path):
            face_rec_path = api_path
            break

    if not face_rec_path:
        print("Could not find face_recognition module to patch")
        return False

    # Read the original file
    with open(face_rec_path, "r") as f:
        content = f.read()
    
    # Check if the module is already patched to avoid duplicate patching
    if "Warning: face_recognition_models not found, trying to continue anyway" in content:
        print("face_recognition module is already patched")
        return True
    
    # Backup the original file only if not already backed up
    backup_path = face_rec_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)
        print(f"Backed up original to {backup_path}")
    else:
        print(f"Backup already exists at {backup_path}")
    
    # Modify the content to not exit on import error
    modified_content = content.replace(
        'try:\n    import face_recognition_models\nexcept Exception:\n    print("Please install `face_recognition_models` with this command before using `face_recognition`:\\n")\n    print("pip install git+https://github.com/ageitgey/face_recognition_models")\n    quit()',
        "try:\n    import face_recognition_models\nexcept Exception:\n    print(\"Warning: face_recognition_models not found, trying to continue anyway...\")\n    # Define paths manually\n    import os\n    class FaceRecognitionModels:\n        def __init__(self):\n            self.models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_recognition_models', 'models')\n        \n        def pose_predictor_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_68_face_landmarks.dat')\n            \n        def pose_predictor_five_point_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_5_face_landmarks.dat')\n            \n        def face_recognition_model_location(self):\n            return os.path.join(self.models_path, 'dlib_face_recognition_resnet_model_v1.dat')\n            \n        def cnn_face_detector_model_location(self):\n            return os.path.join(self.models_path, 'mmod_human_face_detector.dat')\n    \n    face_recognition_models = FaceRecognitionModels()",
    )
    
    # Only write if changes were made
    if content != modified_content:
        # Write the modified content
        with open(face_rec_path, "w") as f:
            f.write(modified_content)
        print(f"Patched {face_rec_path} to continue even if face_recognition_models is not found")
    else:
        print(f"No changes needed for {face_rec_path}")
    
    return True


if __name__ == "__main__":
    # Allow direct execution of this module for testing
    patch_face_recognition()
