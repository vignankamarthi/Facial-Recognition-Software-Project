# This file makes the facial_recognition_software directory a Python package.
# Use lazy imports to avoid circular dependencies

# Define __all__ to control what's imported with "from facial_recognition_software import *"
__all__ = ['FaceDetector', 'FaceMatcher', 'FaceAnonymizer', 'BiasAnalyzer']

# Lazy import implementation
class LazyLoader(object):
    def __init__(self, local_name, parent_module_globals, name):
        self.local_name = local_name
        self.parent_module_globals = parent_module_globals
        self.name = name
        
    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)
        
    def _load(self):
        # Import the module and insert it into the parent's globals
        module = __import__(self.name, fromlist=[None])
        self.parent_module_globals[self.local_name] = module
        return module

# Set up lazy-loaded modules
import sys as _sys
_current_module = _sys.modules[__name__]

# These will be imported only when actually needed
FaceDetector = LazyLoader('FaceDetector', globals(), 'facial_recognition_software.face_detection')
FaceMatcher = LazyLoader('FaceMatcher', globals(), 'facial_recognition_software.face_matching')
FaceAnonymizer = LazyLoader('FaceAnonymizer', globals(), 'facial_recognition_software.anonymization')
BiasAnalyzer = LazyLoader('BiasAnalyzer', globals(), 'facial_recognition_software.bias_testing')
