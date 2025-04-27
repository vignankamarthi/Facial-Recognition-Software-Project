"""
Utility modules for the Facial Recognition Software Project.

This package contains utility functions and classes used across the project,
including configuration, logging, image processing, and common utilities.
"""

# Expose key utility classes at the package level
from .config import Config, get_config
from .logger import get_logger, log_exception, log_method_call
from .image_processing import ImageProcessor
from .common_utils import (
    safely_close_windows, 
    handle_opencv_error, 
    create_resizable_window,
    FaceRecognitionError
)
from .api_patch import patch_face_recognition
