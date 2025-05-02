"""
Face Matching Module

This module provides functionality for comparing detected faces against a database
of known faces for identification purposes.

The module includes the FaceMatcher class which loads reference face images and
compares detected faces against these known references to identify individuals.
It can be used with both static images and real-time video streams.

Functions and Classes
-------------------
FaceMatcher
    Main class for face matching operations
    
See Also
--------
face_detection : Module for detecting faces in images and video
anonymization : Module for applying privacy filters to faces

Examples
--------
>>> from facial_recognition_software.face_matching import FaceMatcher
>>> matcher = FaceMatcher()
>>> # Assume face_locations and face_encodings are from FaceDetector
>>> result_frame, face_names = matcher.identify_faces(frame, face_locations, face_encodings)
"""

import os
import cv2
import numpy as np
import face_recognition
import time
import sys

# Define local fallback constants that will be used if imports fail
_WINDOW_NAME = "Video"
_WAIT_KEY_DELAY = 100
_FACE_MATCHING_THRESHOLD = 0.6
_SUCCESS_COLOR = (0, 255, 0)  # Green
_ERROR_COLOR = (0, 0, 255)  # Red
_KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "known_faces")

# Import utilities safely with failover to local defaults
try:
    from ..utils.common_utils import (
        safely_close_windows, handle_opencv_error, CameraError, MatchingError,
        format_error, create_resizable_window, get_known_faces_dir
    )
    # Import configuration
    from ..utils.config import get_config
    # Get the config singleton instance
    config = get_config()
    # Initialize OpenCV constants after cv2 is imported if needed
    if hasattr(config.ui, 'initialize_opencv_constants'):
        config.ui.initialize_opencv_constants()
    # Get UI config values
    WINDOW_NAME = config.ui.window_name
    WAIT_KEY_DELAY = config.ui.wait_key_delay
    FACE_MATCHING_THRESHOLD = config.matching.threshold
    KNOWN_FACES_DIR = config.paths.known_faces_dir
    SUCCESS_COLOR = config.ui.success_color
    ERROR_COLOR = config.ui.error_color
except ImportError as e:
    # Provide dummy implementations if imports fail
    print(f"Warning: Could not import utilities. Using fallback implementations. Error: {e}")
    
    WINDOW_NAME = _WINDOW_NAME
    WAIT_KEY_DELAY = _WAIT_KEY_DELAY
    FACE_MATCHING_THRESHOLD = _FACE_MATCHING_THRESHOLD
    KNOWN_FACES_DIR = _KNOWN_FACES_DIR
    SUCCESS_COLOR = _SUCCESS_COLOR
    ERROR_COLOR = _ERROR_COLOR
    
    # Minimal fallback implementations to keep things working
    def safely_close_windows(window_name=None, video_capture=None):
        if video_capture is not None and video_capture.isOpened():
            video_capture.release()
        cv2.destroyAllWindows()
    
    def handle_opencv_error(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                cv2.destroyAllWindows()
                return None
        return wrapper
    
    class CameraError(Exception):
        """Fallback exception for camera errors."""
        pass
    
    class MatchingError(Exception):
        """Fallback exception for matching errors."""
        pass
    
    def format_error(error_type, message):
        return f"ERROR: {message}"
    
    def create_resizable_window(window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name
        
    def get_known_faces_dir():
        return KNOWN_FACES_DIR

# Variable to store the FaceDetector class once imported
_FaceDetector = None


class FaceMatcher:
    """
    A class to handle face matching operations.
    
    This class loads reference face images from a directory, extracts facial features,
    and compares detected faces against these known references to identify individuals.
    It supports both real-time webcam face recognition and static image processing.
    
    Parameters
    ----------
    known_faces_dir : str, optional
        Directory containing reference face images
        (default: value from config.KNOWN_FACES_DIR)
        
    Attributes
    ----------
    known_face_encodings : list
        List of 128-dimensional face encodings for known faces
    known_face_names : list
        List of names corresponding to the encodings
    known_faces_dir : str
        Path to the directory containing known face images
        
    Examples
    --------
    >>> # Initialize with default known faces directory
    >>> matcher = FaceMatcher()
    >>> # Initialize with custom directory
    >>> matcher = FaceMatcher('path/to/reference/faces')
    >>> # Use with detected faces
    >>> result_frame, names = matcher.identify_faces(frame, face_locations, face_encodings)
    """

    def __init__(self, known_faces_dir=None):
        """
        Initialize the face matcher with known faces.

        Parameters
        ----------
        known_faces_dir : str, optional
            Directory containing known face images
            (default: None, which uses the value from config.KNOWN_FACES_DIR)
            
        Notes
        -----
        Each image file in the known_faces_dir should be named with the person's name
        (e.g., john_smith.jpg), which will be used as the identity label.
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = known_faces_dir if known_faces_dir else KNOWN_FACES_DIR
        self.load_known_faces()

    def load_known_faces(self):
        """
        Load known faces from the specified directory.
        
        This method processes all image files in the known_faces_dir, extracts face
        encodings, and stores them with the corresponding person names extracted
        from the filenames.
        
        Returns
        -------
        int
            Number of faces successfully loaded
            
        Notes
        -----
        - Image filenames should be in the format: person_name.jpg
        - Underscores in filenames will be converted to spaces in the person's name
        - Only the first face found in each image will be used
        - Images without detectable faces will be skipped with a warning
        
        Examples
        --------
        >>> matcher = FaceMatcher('data/known_faces')
        >>> face_count = matcher.load_known_faces()
        >>> print(f"Loaded {face_count} known faces")
        """
        # Create known_faces_dir if it doesn't exist (for safety)
        if not os.path.exists(self.known_faces_dir):
            print(f"Creating directory since existing one wasn't found: {self.known_faces_dir}")
            os.makedirs(self.known_faces_dir)
            print(f"Please add reference face images to {self.known_faces_dir}")
            return

        # Check if directory is empty
        if not os.listdir(self.known_faces_dir):
            print(f"No face images found in {self.known_faces_dir}")
            print("Please add reference face images to compare against")
            return

        # Load each image file from the directory
        for filename in os.listdir(self.known_faces_dir):
            # Skip non-image files
            if not any(
                filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]
            ):
                continue

            # Extract name from filename (without extension)
            name = os.path.splitext(filename)[0].replace("_", " ")

            # Load the image
            image_path = os.path.join(self.known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)

            # Find face encodings
            face_encodings = face_recognition.face_encodings(image)

            # If no faces found in the image, skip it
            if not face_encodings:
                print(f"Warning: No face found in {filename}")
                continue

            # Use the first face found
            encoding = face_encodings[0]

            # Add to known faces
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

        print(f"Loaded {len(self.known_face_names)} known faces")

    def identify_faces(self, image, face_locations, face_encodings):
        """
        Compare detected faces against known faces to identify them.

        Parameters
        ----------
        image : numpy.ndarray
            Image containing the detected faces
        face_locations : list
            List of face location tuples (top, right, bottom, left)
        face_encodings : list
            List of 128-dimensional face encodings corresponding to face_locations

        Returns
        -------
        numpy.ndarray
            Frame with identified faces labeled and highlighted
        list
            List of identified names, including confidence scores for matched faces
            
        Notes
        -----
        - Matched faces are highlighted with green boxes
        - Unknown faces are highlighted with red boxes
        - Matched face labels include confidence scores (e.g., "John Smith (0.82)")
            
        Examples
        --------
        >>> # Using with face detector
        >>> face_locations, face_encodings = detector.detect_faces(frame)
        >>> result_frame, names = matcher.identify_faces(frame, face_locations, face_encodings)
        >>> print(f"Identified {len(names)} faces: {names}")
        """
        # Create a copy of the image
        display_image = image.copy()
        face_names = []

        # Check if we have known faces to compare against
        if not self.known_face_encodings:
            for top, right, bottom, left in face_locations:
                name = "Unknown"
                face_names.append(name)

                # Draw a box around the face
                cv2.rectangle(
                    display_image, (left, top), (right, bottom), (0, 0, 255), 2
                )

                # Draw a label with the name below the face
                cv2.rectangle(
                    display_image,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    display_image,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )
        else:
            for face_encoding, (top, right, bottom, left) in zip(
                face_encodings, face_locations
            ):
                # Compare the face with known faces
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, 0.8
                )
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]

                    if matches[best_match_index] and confidence > FACE_MATCHING_THRESHOLD:
                        name = self.known_face_names[best_match_index]
                        name = f"{name} ({confidence:.2f})"
                        box_color = SUCCESS_COLOR  # Green for recognized faces
                    else:
                        box_color = ERROR_COLOR  # Red for unknown faces
                else:
                    box_color = ERROR_COLOR

                face_names.append(name)

                # Draw a box around the face
                cv2.rectangle(display_image, (left, top), (right, bottom), box_color, 2)

                # Draw a label with the name below the face
                cv2.rectangle(
                    display_image,
                    (left, bottom - 35),
                    (right, bottom),
                    box_color,
                    cv2.FILLED,
                )
                cv2.putText(
                    display_image,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )

        return display_image, face_names

    @handle_opencv_error
    def match_faces_webcam(self):
        """
        Deprecated: Direct webcam access is now handled through the Streamlit interface.
        
        This method is kept for backward compatibility with tests.

        Returns
        -------
        tuple
            (success, result_dict) where:
            - success : bool
              True if operation completed normally
            - result_dict : dict
              Contains metadata about the operation
              
        Notes
        -----
        - Use the Streamlit interface instead of this method
        - The returned data is mocked for test compatibility
        
        Examples
        --------
        >>> # Use Streamlit interface instead of this method
        """
        print("Direct webcam access is deprecated. Please use the Streamlit interface.")
        
        # Return a mock result for testing purposes
        return True, {
            'face_count': 0,
            'frames_processed': 0,
            'duration': 0,
            'note': 'Direct webcam access is deprecated. Please use the Streamlit interface.'
        }


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    try:
        matcher = FaceMatcher()
        matcher.match_faces_webcam()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    finally:
        # Make sure to release any OpenCV resources
        safely_close_windows()
