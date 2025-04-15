# This file makes the utilities directory a Python package.
# Use lazy imports to avoid circular dependencies

# Define __all__ to control what's imported with "from utilities import *"
__all__ = ['ImageProcessor']

# Create a simple lazy loader for ImageProcessor
class _ImageProcessorLoader:
    @staticmethod
    def __new__():
        from .image_processing import ImageProcessor
        return ImageProcessor

# Access ImageProcessor through the loader
def get_image_processor(*args, **kwargs):
    """Get an instance of ImageProcessor."""
    processor_class = _ImageProcessorLoader.__new__()
    return processor_class(*args, **kwargs)

# Make ImageProcessor accessible as if it were directly imported
class ImageProcessor:
    def __new__(cls, *args, **kwargs):
        processor_class = _ImageProcessorLoader.__new__()
        return processor_class(*args, **kwargs)
