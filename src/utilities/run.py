#!/usr/bin/env python3
"""
Run script that ensures all modules are properly set up before running the main application.

# TODO: PLEASE REVIEW THIS
# TODO: PLEASE REVIEW README FILE

# This script was added to solve dependency issues with face_recognition and face_recognition_models.
# It automatically patches the face_recognition module if needed and runs the main application.
# If you encounter issues, you may need to review how the patching is done.
"""

import os
import sys

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
    
    try:
        # Find the face_recognition module's api.py file
        face_rec_path = None
        for path in sys.path:
            api_path = os.path.join(path, 'face_recognition', 'api.py')
            if os.path.exists(api_path):
                face_rec_path = api_path
                break

        if face_rec_path:
            # Read the original file
            with open(face_rec_path, 'r') as f:
                content = f.read()
            
            # Backup the original file
            backup_path = face_rec_path + '.backup'
            if not os.path.exists(backup_path):
                with open(backup_path, 'w') as f:
                    f.write(content)
                print(f"Backed up original to {backup_path}")
            
            # Modify the content to not exit on import error
            modified_content = content.replace(
                "try:\n    import face_recognition_models\nexcept Exception:\n    print(\"Please install `face_recognition_models` with this command before using `face_recognition`:\\n\")\n    print(\"pip install git+https://github.com/ageitgey/face_recognition_models\")\n    quit()",
                "try:\n    import face_recognition_models\nexcept Exception:\n    print(\"Warning: face_recognition_models not found, trying to continue anyway...\")\n    # Define paths manually\n    import os\n    class FaceRecognitionModels:\n        def __init__(self):\n            self.models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_recognition_models', 'models')\n        \n        def pose_predictor_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_68_face_landmarks.dat')\n            \n        def pose_predictor_five_point_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_5_face_landmarks.dat')\n            \n        def face_recognition_model_location(self):\n            return os.path.join(self.models_path, 'dlib_face_recognition_resnet_model_v1.dat')\n            \n        def cnn_face_detector_model_location(self):\n            return os.path.join(self.models_path, 'mmod_human_face_detector.dat')\n    \n    face_recognition_models = FaceRecognitionModels()"
            )
            
            # Write the modified content
            with open(face_rec_path, 'w') as f:
                f.write(modified_content)
            
            print(f"Patched {face_rec_path} to continue even if face_recognition_models is not found")
            
            # Reload the module
            if 'face_recognition' in sys.modules:
                del sys.modules['face_recognition']
            import face_recognition
            print("Successfully reloaded face_recognition after patching.")
        else:
            print("Could not find face_recognition module to patch")
    except Exception as e:
        print(f"Failed to patch face_recognition: {e}")

# Now run the main application
print("\nStarting the Facial Recognition System...\n")
import main
