# This file makes the facial_recognition_software directory a Python package.
# Use lazy imports to avoid circular dependencies

# Define __all__ to control what's imported with "from facial_recognition_software import *"
__all__ = ['FaceDetector', 'FaceMatcher', 'FaceAnonymizer', 'BiasAnalyzer']

# Create proper class-like lazy loaders that return the actual class when called
class LazyClass:
    def __init__(self, module_path, class_name):
        self.module_path = module_path
        self.class_name = class_name
        self._class = None
        
    def __call__(self, *args, **kwargs):
        if self._class is None:
            module = __import__(self.module_path, fromlist=[self.class_name])
            self._class = getattr(module, self.class_name)
        return self._class(*args, **kwargs)

# Implement our classes with proper lazy loading
FaceDetector = LazyClass('facial_recognition_software.face_detection', 'FaceDetector')
FaceMatcher = LazyClass('facial_recognition_software.face_matching', 'FaceMatcher')
FaceAnonymizer = LazyClass('facial_recognition_software.anonymization', 'FaceAnonymizer')
BiasAnalyzer = LazyClass('facial_recognition_software.bias_testing', 'BiasAnalyzer')
