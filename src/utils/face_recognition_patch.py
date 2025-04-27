"""
Utility module for patching face_recognition imports.

This module provides a centralized solution for patching the face_recognition
module to continue functioning even when face_recognition_models can't be
properly imported. This is a critical utility that ensures consistent behavior
across all modules that use face_recognition.
"""

import os
import sys
import importlib
from .logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Global flag to track if patch has been applied
_patch_applied = False


def patch_face_recognition(force=False):
    """
    Apply monkey patch to face_recognition module to handle missing face_recognition_models.
    
    This function modifies the face_recognition API file to bypass the import check
    that would otherwise terminate the program if face_recognition_models is not found.
    
    Parameters
    ----------
    force : bool, optional
        If True, apply the patch even if it appears to be already applied (default: False)
    
    Returns
    -------
    bool
        True if patching was successful, False otherwise
    """
    global _patch_applied
    
    # Skip if already patched and not forced
    if _patch_applied and not force:
        logger.info("face_recognition patch has already been applied in this session")
        return True
        
    # Find the face_recognition module's api.py file
    face_rec_path = None
    for path in sys.path:
        api_path = os.path.join(path, "face_recognition", "api.py")
        if os.path.exists(api_path):
            face_rec_path = api_path
            break
    
    # Alternative method to find the module path if the above fails
    if not face_rec_path:
        try:
            import face_recognition
            module_path = os.path.dirname(face_recognition.__file__)
            face_rec_path = os.path.join(module_path, "api.py")
        except ImportError:
            pass

    if not face_rec_path:
        logger.error("Could not find face_recognition module to patch")
        return False

    # Read the original file
    with open(face_rec_path, "r") as f:
        content = f.read()
    
    # Check if the module is already patched to avoid duplicate patching
    if "Warning: face_recognition_models not found, trying to continue anyway" in content:
        logger.info("face_recognition module is already patched")
        _patch_applied = True
        return True
    
    # Backup the original file only if not already backed up
    backup_path = face_rec_path + ".backup"
    if not os.path.exists(backup_path):
        with open(backup_path, "w") as f:
            f.write(content)
        logger.info(f"Backed up original to {backup_path}")
    else:
        logger.info(f"Backup already exists at {backup_path}")
    
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
        logger.info(f"Patched {face_rec_path} to continue even if face_recognition_models is not found")
        _patch_applied = True
    else:
        logger.info(f"No changes needed for {face_rec_path}")
        _patch_applied = True
    
    # Try to reload the module if it's already imported
    if 'face_recognition' in sys.modules:
        try:
            del sys.modules['face_recognition']
            import face_recognition
            logger.info("Successfully reloaded face_recognition module after patching")
        except Exception as e:
            logger.warning(f"Could not reload face_recognition module: {e}")
    
    return True


def verify_face_recognition():
    """
    Verify that face_recognition is working properly, patching if needed.
    
    Returns
    -------
    bool
        True if face_recognition is working, False otherwise
    """
    try:
        # Try to import and use a basic function to verify
        import face_recognition
        test_func = getattr(face_recognition, 'load_image_file', None)
        if test_func is None:
            logger.warning("face_recognition module is missing expected functions")
            return patch_face_recognition(force=True)
        logger.info("face_recognition module verified successfully")
        return True
    except Exception as e:
        logger.error(f"face_recognition verification failed: {e}")
        return patch_face_recognition(force=True)


if __name__ == "__main__":
    # Allow direct execution of this module for testing
    success = verify_face_recognition()
    print(f"Verification {'succeeded' if success else 'failed'}")
