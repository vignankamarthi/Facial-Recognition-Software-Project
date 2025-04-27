# This file makes the utilities directory a Python package.
# Use lazy imports to avoid circular dependencies

# Define __all__ to control what's imported with "from utilities import *"
__all__ = ['ImageProcessor']

# Use the same LazyClass approach as in facial_recognition_software/__init__.py
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

# Implement our classes with consistent lazy loading
ImageProcessor = LazyClass('utilities.image_processing', 'ImageProcessor')
