""" 
This is a patch module to fix the face_recognition import issue.
It modifies the face_recognition module to continue even if face_recognition_models can't be imported.

# TODO: PLEASE REVIEW THIS
# This script modifies the face_recognition API file to bypass the import check
# that would otherwise terminate the program if face_recognition_models is not found.
# If you encounter issues with face detection, you might need to review this patch.
"""

import sys
import os
import importlib.util

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
    
    # Modify the content to not exit on import error
    modified_content = content.replace(
        "try:\n    import face_recognition_models\nexcept Exception:\n    print(\"Please install `face_recognition_models` with this command before using `face_recognition`:\\n\")\n    print(\"pip install git+https://github.com/ageitgey/face_recognition_models\")\n    quit()",
        "try:\n    import face_recognition_models\nexcept Exception:\n    print(\"Warning: face_recognition_models not found, trying to continue anyway...\")\n    # Define paths manually\n    import os\n    class FaceRecognitionModels:\n        def __init__(self):\n            self.models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'face_recognition_models', 'models')\n        \n        def pose_predictor_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_68_face_landmarks.dat')\n            \n        def pose_predictor_five_point_model_location(self):\n            return os.path.join(self.models_path, 'shape_predictor_5_face_landmarks.dat')\n            \n        def face_recognition_model_location(self):\n            return os.path.join(self.models_path, 'dlib_face_recognition_resnet_model_v1.dat')\n            \n        def cnn_face_detector_model_location(self):\n            return os.path.join(self.models_path, 'mmod_human_face_detector.dat')\n    \n    face_recognition_models = FaceRecognitionModels()"
    )
    
    # Backup the original file
    backup_path = face_rec_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"Backed up original to {backup_path}")
    
    # Write the modified content
    with open(face_rec_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Patched {face_rec_path} to continue even if face_recognition_models is not found")
else:
    print("Could not find face_recognition module to patch")
