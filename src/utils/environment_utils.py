"""
Environment detection utilities for testing and runtime adaptation.

This module provides functions to detect the execution environment
characteristics such as headless mode, webcam availability, and CI status.
"""

import os
import platform
import cv2
import logging

# Initialize logger
from src.utils.logger import get_logger
logger = get_logger(__name__)

def is_ci_environment():
    """
    Detect if running in a CI environment.
    
    Returns:
        bool: True if running in a CI environment, False otherwise
    """
    # Check for GitHub Actions CI environment variable
    return os.environ.get('GITHUB_ACTIONS', 'False').lower() in ('true', '1', 't', 'yes')

def is_headless_environment():
    """
    Detect if running in a headless environment (no display).
    
    Returns:
        bool: True if in a headless environment, False otherwise
    """
    # Check for DISPLAY environment variable on Unix-like systems
    if platform.system() != "Windows":
        if not os.environ.get('DISPLAY'):
            logger.debug("No DISPLAY environment variable found, assuming headless")
            return True
    
    # Check for forced headless mode
    if os.environ.get('FORCE_HEADLESS', 'False').lower() in ('true', '1', 't', 'yes'):
        logger.debug("FORCE_HEADLESS is set, assuming headless")
        return True
    
    # Check if we're in a CI environment (typically headless)
    if is_ci_environment():
        logger.debug("GitHub Actions environment detected, assuming headless")
        return True
    
    # Try to create a test window to verify display availability
    try:
        # Create a small off-screen window
        cv2.namedWindow('__test_window__', cv2.WINDOW_NORMAL)
        cv2.moveWindow('__test_window__', -1, -1)
        cv2.waitKey(1)
        cv2.destroyWindow('__test_window__')
        logger.debug("Test window created successfully, display is available")
        return False
    except Exception as e:
        logger.debug(f"Failed to create test window: {e}, assuming headless")
        return True

def is_webcam_available():
    """
    Check if a webcam is available on the system.
    
    Returns:
        bool: True if a webcam is available, False otherwise
    """
    # Check for forced webcam availability settings
    webcam_env = os.environ.get('FORCE_WEBCAM_AVAILABLE', '').lower()
    if webcam_env in ('true', '1', 't', 'yes'):
        logger.debug("FORCE_WEBCAM_AVAILABLE is set to True")
        return True
    if webcam_env in ('false', '0', 'f', 'no'):
        logger.debug("FORCE_WEBCAM_AVAILABLE is set to False")
        return False
    
    # Check if we're in a CI environment (typically no webcam)
    if is_ci_environment():
        logger.debug("GitHub Actions environment detected, assuming no webcam")
        return False
    
    # Actually try to open the camera
    try:
        cap = cv2.VideoCapture(0)
        success = cap.isOpened()
        if success:
            # Read a frame to verify camera is working
            ret, frame = cap.read()
            success = ret and frame is not None
        cap.release()
        logger.debug(f"Webcam detection result: {'available' if success else 'not available'}")
        return success
    except Exception as e:
        logger.debug(f"Error checking webcam availability: {e}")
        return False

def get_environment_info():
    """
    Get comprehensive information about the current environment.
    
    Returns:
        dict: Dictionary with environment information
    """
    return {
        'ci': is_ci_environment(),
        'headless': is_headless_environment(),
        'webcam_available': is_webcam_available(),
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'opencv_version': cv2.__version__
    }
