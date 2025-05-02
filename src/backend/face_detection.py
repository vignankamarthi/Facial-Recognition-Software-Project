"""
Face Detection Module

This module provides functionality for detecting faces in images and video streams.
It uses OpenCV and face_recognition libraries to identify facial features.

This module includes the FaceDetector class for locating faces in images and
drawing bounding boxes around them. It supports both real-time detection
via webcam and processing of static images.
"""

import cv2
import face_recognition
import numpy as np
import time
import sys
import os

# Import utilities with logging - use relative imports for sibling packages
from ..utils.common_utils import (
    safely_close_windows,
    handle_opencv_error,
    CameraError,
    DetectionError,
    format_error,
    create_resizable_window,
)
from ..utils.logger import get_logger, log_exception, log_method_call

# Initialize logger for this module
logger = get_logger(__name__)

# Import configuration
try:
    from ..utils.config import get_config
    # Get the config singleton instance
    config = get_config()
    # Initialize OpenCV constants after cv2 is imported if needed
    if hasattr(config.ui, 'initialize_opencv_constants'):
        config.ui.initialize_opencv_constants()
    # Get UI config values
    WINDOW_NAME = config.ui.window_name
    WAIT_KEY_DELAY = config.ui.wait_key_delay
except ImportError as e:
    # Fallback constants if config module is not available
    logger.warning(f"Could not import config module: {e}. Using fallback constants.")
    WINDOW_NAME = "Video"
    WAIT_KEY_DELAY = 100


class FaceDetector:
    """A class to handle face detection operations.
    
    This class provides methods to detect faces in images and video streams
    using the face_recognition library. It can identify face locations and
    generate face encodings for later use in face matching.
    
    Attributes
    ----------
    face_locations : list
        Last detected face locations as (top, right, bottom, left) tuples
    face_encodings : list
        Last generated face encodings for the detected faces
    """

    def __init__(self):
        """Initialize the face detector with empty face locations and encodings."""
        self.face_locations = []
        self.face_encodings = []
        logger.debug("FaceDetector initialized")

    @log_method_call(logger)
    def detect_faces(self, image):
        """
        Detect faces in a given frame.

        Parameters
        ----------
        image : numpy.ndarray
            Image to process

        Returns
        -------
        tuple
            (face_locations, face_encodings) where:
            - face_locations : list
              List of face location tuples (top, right, bottom, left)
            - face_encodings : list
              List of 128-dimensional face encodings
              
        Raises
        ------
        DetectionError
            If face detection fails due to invalid input
        ValueError
            If the input frame is None or invalid
        
        Examples
        --------
        >>> detector = FaceDetector()
        >>> image = cv2.imread("image.jpg")
        >>> face_locations, face_encodings = detector.detect_faces(image)
        >>> print(f"Found {len(face_locations)} faces")
        """
        # Validate input image
        if image is None:
            error_msg = "Cannot detect faces in None image"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if image.size == 0:
            error_msg = "Empty image provided for face detection"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Convert the image from BGR color (OpenCV's default format) to RGB (face_recognition)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find all the faces in the current frame
            self.face_locations = face_recognition.face_locations(rgb_image)
            self.face_encodings = face_recognition.face_encodings(
                rgb_image, self.face_locations
            )
            
            logger.debug(f"Detected {len(self.face_locations)} faces in image")
            return self.face_locations, self.face_encodings
            
        except Exception as e:
            error_msg = "Face detection failed"
            log_exception(logger, error_msg, e)
            raise DetectionError(error_msg, str(e), e)

    @log_method_call(logger)
    def draw_face_boxes(self, image, face_locations, color=(0, 255, 0), thickness=2):
        """
        Draw colored boxes around detected faces.

        Parameters
        ----------
        image : numpy.ndarray
            Image to draw on
        face_locations : list
            List of face location tuples (top, right, bottom, left)
        color : tuple, optional
            BGR color for the box (default: (0, 255, 0) - green)
        thickness : int, optional
            Line thickness (default: 2)

        Returns
        -------
        numpy.ndarray
            A new frame with boxes drawn around faces
            
        Raises
        ------
        ValueError
            If the input frame is None or invalid
            
        Examples
        --------
        >>> detector = FaceDetector()
        >>> image = cv2.imread("image.jpg")
        >>> face_locations, _ = detector.detect_faces(image)
        >>> result = detector.draw_face_boxes(image, face_locations)
        >>> cv2.imshow("Faces", result)
        """
        # Validate input image
        if image is None:
            error_msg = "Cannot draw boxes on None image"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Validate face locations
        if not isinstance(face_locations, list):
            logger.warning(f"face_locations is not a list: {type(face_locations)}")
            face_locations = list(face_locations) if face_locations else []
        
        # Create a copy of the image to avoid modifying the original
        display_image = image.copy()

        # Draw a box around each detected face
        for i, (top, right, bottom, left) in enumerate(face_locations):
            cv2.rectangle(display_image, (left, top), (right, bottom), color, thickness)
            logger.debug(f"Drew face box {i+1} at location ({left}, {top}, {right}, {bottom})")

        return display_image

    @handle_opencv_error
    @log_method_call(logger)
    def detect_faces_webcam(self, anonymize=False, anonymizer=None):
        """
        Deprecated: Direct webcam access is now handled through the Streamlit interface.
        
        This method is kept for backward compatibility with tests.

        Parameters
        ----------
        anonymize : bool, optional
            Whether to anonymize detected faces (default: False)
        anonymizer : FaceAnonymizer, optional
            Instance of FaceAnonymizer to use for anonymization.

        Returns
        -------
        tuple
            (success, result_dict) where:
            - success : bool
              True if operation completed normally, False if an error occurred
            - result_dict : dict
              Contains metadata about the operation:
              - "face_count" : int
                Total number of faces detected during the session
              - "frames_processed" : int
                Number of frames processed
              - "duration" : float
                Duration of the session in seconds
              - "note" : str
                Information about the deprecation
        
        Examples
        --------
        >>> detector = FaceDetector()
        >>> # Use Streamlit interface instead of this method
        """
        logger.warning("detect_faces_webcam is deprecated - use Streamlit interface instead")
        print("Direct webcam access is deprecated. Please use the Streamlit interface.")
        
        # Return a mock result for testing purposes
        return True, {
            'face_count': 0,
            'frames_processed': 0,
            'duration': 0,
            'note': 'Direct webcam access is deprecated. Please use the Streamlit interface.'
        }
            
    @log_method_call(logger)
    def process_image_file(self, image_path):
        """
        Process a single image file, detecting faces.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
            
        Returns
        -------
        tuple
            (success, result_dict) where:
            - success : bool
              True if processing was successful, False otherwise
            - result_dict : dict
              Dictionary containing metadata about the processing:
              - "face_count" : int
                Number of faces detected
              - "face_locations" : list
                List of face location tuples
              - "error" : str, optional
                Error message if an error occurred
        """
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Validate image path
            if not os.path.exists(image_path):
                error_msg = f"Image file not found: {image_path}"
                logger.error(error_msg)
                return False, {"error": error_msg}
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"Failed to load image: {image_path}"
                logger.error(error_msg)
                return False, {"error": error_msg}
            
            # Detect faces
            face_locations, _ = self.detect_faces(image)
            logger.info(f"Detected {len(face_locations)} faces in {image_path}")
            
            # Draw boxes around faces (for processing only, not display)
            display_image = self.draw_face_boxes(image, face_locations)
            
            # Return results (no GUI display)
            return True, {
                "face_count": len(face_locations),
                "face_locations": face_locations,
                "image_path": image_path
            }
            
        except Exception as e:
            error_msg = f"Error processing image: {e}"
            log_exception(logger, error_msg, e)
            return False, {"error": error_msg}


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    try:
        logger.info("Running face detection test")
        detector = FaceDetector()
        success, result = detector.detect_faces_webcam()
        
        if success:
            logger.info(f"Face detection test completed successfully with {result['face_count']} faces detected")
            print(f"Test completed successfully:")
            print(f"- Processed {result['frames_processed']} frames")
            print(f"- Detected {result['face_count']} faces total")
            print(f"- Session duration: {result['duration']:.2f} seconds")
        else:
            logger.error(f"Face detection test failed: {result.get('error', 'Unknown error')}")
            print(f"Test failed: {result.get('error', 'Unknown error')}")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        print("\nTest interrupted by user. Exiting.")
    except Exception as e:
        log_exception(logger, "Unexpected error during test", e)
        print(f"Unexpected error: {e}")
    finally:
        # Make sure to release any OpenCV resources
        safely_close_windows()
