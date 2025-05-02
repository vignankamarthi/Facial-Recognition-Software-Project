"""
Anonymization Module

This module provides functionality for anonymizing detected faces in images
and video streams to protect privacy.

The module includes the FaceAnonymizer class which implements various methods
for obscuring facial features while preserving the overall image context.
This is useful for privacy protection, ethical demonstrations, and educational
purposes about facial recognition technologies.

Functions and Classes
-------------------
FaceAnonymizer
    Main class for applying anonymization to faces
run_anonymization_demo
    Standalone demo showing anonymization capabilities

See Also
--------
face_detection : Module for detecting faces in images and video

Examples
--------
>>> from facial_recognition_software.anonymization import FaceAnonymizer
>>> anonymizer = FaceAnonymizer(method='blur', intensity=50)
>>> anonymized_frame = anonymizer.anonymize_frame(frame, face_locations)
"""

import cv2
import numpy as np
import time
import sys
import os

# Define local fallback constants that will be used if imports fail
_WINDOW_NAME = "Video"
_WAIT_KEY_DELAY = 100
_DEFAULT_ANONYMIZATION_METHOD = "blur"
_DEFAULT_ANONYMIZATION_INTENSITY = 90
_WARNING_COLOR = (0, 255, 255)  # Yellow

# Import utilities safely with failover to local defaults
try:
    from ..utils.common_utils import (
        safely_close_windows, handle_opencv_error, CameraError, AnonymizationError,
        format_error, create_resizable_window
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
    DEFAULT_ANONYMIZATION_METHOD = config.anonymization.default_method
    DEFAULT_ANONYMIZATION_INTENSITY = config.anonymization.default_intensity
    WARNING_COLOR = config.ui.warning_color
except ImportError as e:
    # Provide dummy implementations if imports fail
    print(f"Warning: Could not import utils. Using fallback implementations. Error: {e}")

    WINDOW_NAME = _WINDOW_NAME
    WAIT_KEY_DELAY = _WAIT_KEY_DELAY
    DEFAULT_ANONYMIZATION_METHOD = _DEFAULT_ANONYMIZATION_METHOD
    DEFAULT_ANONYMIZATION_INTENSITY = _DEFAULT_ANONYMIZATION_INTENSITY
    WARNING_COLOR = _WARNING_COLOR

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

    class AnonymizationError(Exception):
        """Fallback exception for anonymization errors."""
        pass

    def format_error(error_type, message):
        return f"ERROR: {message}"

    def create_resizable_window(window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name


class FaceAnonymizer:
    """
    A class to handle face anonymization operations.
    
    This class provides methods to anonymize faces in images and video streams
    using different techniques such as blurring, pixelation, and masking.
    It supports real-time anonymization for webcam feeds and static image
    processing.
    
    Parameters
    ----------
    method : str, optional
        Anonymization method to use ('blur', 'pixelate', or 'mask')
        (default: value from config.DEFAULT_ANONYMIZATION_METHOD = blur)
    intensity : int, optional
        Intensity of the anonymization effect, range 1-100
        (default: value from config.DEFAULT_ANONYMIZATION_INTENSITY = 90)
        
    Attributes
    ----------
    method : str
        Current anonymization method being used
    intensity : int
        Current intensity level of the anonymization effect
        
    Examples
    --------
    >>> anonymizer = FaceAnonymizer(method='blur', intensity=50)
    >>> anonymized_frame = anonymizer.anonymize_frame(frame, face_locations)
    >>> anonymizer.set_method('pixelate')  # Change method
    """

    def __init__(self, method=None, intensity=None):
        """
        Initialize the face anonymizer.

        Parameters
        ----------
        method : str, optional
            Anonymization method ('blur', 'pixelate', or 'mask')
            (default: value from config.DEFAULT_ANONYMIZATION_METHOD)
        intensity : int, optional
            Intensity of the anonymization effect, range 1-100
            (default: value from config.DEFAULT_ANONYMIZATION_INTENSITY)
        """
        self.method = method if method is not None else DEFAULT_ANONYMIZATION_METHOD
        self.intensity = intensity if intensity is not None else DEFAULT_ANONYMIZATION_INTENSITY

    def anonymize_face(self, image, face_location, method=None):
        """
        Apply anonymization to a single face.

        Parameters
        ----------
        image : numpy.ndarray
            Image containing the face to anonymize
        face_location : tuple
            Face location tuple (top, right, bottom, left)
        method : str, optional
            Override the default anonymization method
            (default: None, which uses self.method)

        Returns
        -------
        numpy.ndarray
            Frame with the specified face anonymized
            
        Examples
        --------
        >>> # Anonymize a single face in an image
        >>> frame = cv2.imread('image.jpg')
        >>> face_location = (50, 200, 150, 100)  # (top, right, bottom, left)
        >>> result = anonymizer.anonymize_face(frame, face_location)
        """
        # Make a copy of the image to avoid modifying the original
        result_image = image.copy()

        # Extract face location coordinates
        top, right, bottom, left = face_location
        face_img = image[top:bottom, left:right]

        # Use specified method or default
        current_method = method if method else self.method

        if current_method == "blur":
            # Apply Gaussian blur
            # Ensure kernel size is odd (required by GaussianBlur)
            kernel_size = (
                self.intensity if self.intensity % 2 == 1 else self.intensity + 1
            )
            blurred_face = cv2.GaussianBlur(face_img, (kernel_size, kernel_size), 0)
            result_image[top:bottom, left:right] = blurred_face

        elif current_method == "pixelate":
            # Pixelate by downscaling and upscaling
            height, width = face_img.shape[:2]

            # Calculate the scaling factor based on intensity
            scale_factor = max(1, self.intensity // 5)

            # Downscale
            temp = cv2.resize(
                face_img,
                (width // scale_factor, height // scale_factor),
                interpolation=cv2.INTER_LINEAR,
            )

            # Upscale with nearest neighbor to create pixelation effect
            pixelated_face = cv2.resize(
                temp, (width, height), interpolation=cv2.INTER_NEAREST
            )

            result_image[top:bottom, left:right] = pixelated_face

        elif current_method == "mask":
            # Create a solid color mask
            mask_color = (0, 0, 0)  # Black mask
            cv2.rectangle(result_image, (left, top), (right, bottom), mask_color, -1)

            # Optional: Draw a face icon or text
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            radius = min(right - left, bottom - top) // 4

            # Draw a simple face icon
            cv2.circle(result_image, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(
                result_image,
                (center_x - radius // 2, center_y - radius // 3),
                radius // 5,
                (0, 0, 0),
                -1,
            )  # Left eye
            cv2.circle(
                result_image,
                (center_x + radius // 2, center_y - radius // 3),
                radius // 5,
                (0, 0, 0),
                -1,
            )  # Right eye
            cv2.ellipse(
                result_image,
                (center_x, center_y + radius // 3),
                (radius // 2, radius // 3),
                0,
                0,
                180,
                (0, 0, 0),
                -1,
            )  # Mouth

        # Add a visual indicator that this face is anonymized
        cv2.rectangle(result_image, (left, top), (right, bottom), WARNING_COLOR, 2)
        cv2.putText(
            result_image,
            "Anonymized",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WARNING_COLOR,
            2,
        )

        return result_image

    def anonymize_frame(self, image, face_locations):
        """
        Apply anonymization to all faces in a frame.

        Parameters
        ----------
        image : numpy.ndarray
            Image containing faces to anonymize
        face_locations : list
            List of face location tuples (top, right, bottom, left)

        Returns
        -------
        numpy.ndarray
            Frame with all faces anonymized
            
        Notes
        -----
        This method also adds a semi-transparent status bar showing
        the current anonymization method.
            
        Examples
        --------
        >>> # Anonymize all faces in an image
        >>> image = cv2.imread('image.jpg')
        >>> # Assume face_locations has been obtained from a FaceDetector
        >>> result = anonymizer.anonymize_frame(image, face_locations)
        """
        result_image = image.copy()

        for face_location in face_locations:
            result_image = self.anonymize_face(result_image, face_location)

        # Add semi-transparent background for menu text
        overlay = result_image.copy()
        # Make the box for showing information
        cv2.rectangle(overlay, (5, 5), (400, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_image, 0.4, 0, result_image)
        
        # Add indicator for anonymization mode
        cv2.putText(
            result_image,
            f"Anonymization: {self.method.capitalize()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            WARNING_COLOR,
            2,
        )

        return result_image

    def set_method(self, method):
        """
        Change the anonymization method.

        Parameters
        ----------
        method : str
            New anonymization method ('blur', 'pixelate', or 'mask')
            
        Examples
        --------
        >>> # Switch to pixelation method
        >>> anonymizer.set_method('pixelate')
        Anonymization method set to: pixelate
        """
        valid_methods = ["blur", "pixelate", "mask"]
        if method in valid_methods:
            self.method = method
            print(f"Anonymization method set to: {method}")
        else:
            print(f"Invalid method. Choose from: {valid_methods}")

    def set_intensity(self, intensity):
        """
        Change the intensity of the anonymization effect.

        Parameters
        ----------
        intensity : int
            New intensity value, range 1-100
            
        Notes
        -----
        Higher intensity values result in:
        - Stronger blur for 'blur' method
        - Larger pixels for 'pixelate' method
        - No effect for 'mask' method (which is always 100% opaque)
            
        Examples
        --------
        >>> # Set to medium intensity
        >>> anonymizer.set_intensity(50)
        Anonymization intensity set to: 50
        """
        if 1 <= intensity <= 100:
            self.intensity = intensity
            print(f"Anonymization intensity set to: {intensity}")
        else:
            print("Intensity must be between 1 and 100")

    def demonstrate_methods(self, image, face_location):
        """
        Demonstrate different anonymization methods on a single face.

        Parameters
        ----------
        image : numpy.ndarray
            Image containing the face
        face_location : tuple
            Face location tuple (top, right, bottom, left)

        Returns
        -------
        dict
            Dictionary of frames with different anonymization methods:
            - 'original': Original unmodified frame
            - 'blur': Frame with blurred face
            - 'pixelate': Frame with pixelated face
            - 'mask': Frame with masked face
            
        Examples
        --------
        >>> # Compare all anonymization methods
        >>> image = cv2.imread('image.jpg')
        >>> face_location = (50, 200, 150, 100)  # (top, right, bottom, left)
        >>> results = anonymizer.demonstrate_methods(image, face_location)
        >>> # Display results
        >>> cv2.imshow('Original', results['original'])
        >>> cv2.imshow('Blur', results['blur'])
        """
        methods = {
            "original": image.copy(),
            "blur": self.anonymize_face(image, face_location, "blur"),
            "pixelate": self.anonymize_face(image, face_location, "pixelate"),
            "mask": self.anonymize_face(image, face_location, "mask"),
        }

        return methods


# Variable to store the FaceDetector class once imported
_FaceDetector = None

@handle_opencv_error
def run_anonymization_demo():
    """
    Run a standalone anonymization demo.
    
    This function initializes webcam capture, detects faces in real-time,
    applies anonymization, and displays the results. It serves as a
    complete demonstration of the anonymization functionality.
    
    Returns
    -------
    None
    
    Notes
    -----
    - Press 'q' or ESC to quit the demo
    - The demo uses default anonymization settings (blur method)
    - This function handles all necessary cleanup of resources
    
    Raises
    ------
    CameraError
        If the webcam cannot be opened or an error occurs during capture
    AnonymizationError
        If an error occurs during the anonymization process
    """
    # Lazily import FaceDetector to avoid circular imports
    global _FaceDetector
    if _FaceDetector is None:
        from .face_detection import FaceDetector
        _FaceDetector = FaceDetector
    
    detector = _FaceDetector()
    anonymizer = FaceAnonymizer()

    # Initialize webcam
    video_capture = None
    
    try:
        # Initialize webcam - try multiple indices for macOS compatibility
        video_capture = None
        for camera_index in [0, 1, -1]:
            video_capture = cv2.VideoCapture(camera_index)
            if video_capture.isOpened():
                print(f"Successfully opened webcam with index {camera_index}")
                break
            else:
                print(f"Failed to open webcam with index {camera_index}")

        if not video_capture or not video_capture.isOpened():
            error_msg = format_error("Camera", "Could not open webcam")
            print(error_msg)
            raise CameraError(error_msg)

        print("Press Ctrl+C to quit...")
        
        # Create a resizable window using utility function
        create_resizable_window(WINDOW_NAME)

        # Main processing loop
        while True:
            # Capture frame-by-frame
            ret, image = video_capture.read()

            if not ret:
                error_msg = format_error("Camera", "Failed to capture frame")
                print(error_msg)
                break

            # Detect faces in the image
            face_locations, _ = detector.detect_faces(image)

            # Anonymize the faces
            display_image = anonymizer.anonymize_frame(image, face_locations)

            # Display the resulting image
            cv2.imshow(WINDOW_NAME, display_image)
            
            # Short wait time
            key = cv2.waitKey(WAIT_KEY_DELAY) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:  # q, Q, or ESC
                print("Quitting anonymization...")
                break
            
    except KeyboardInterrupt:
        print("\nAnonymization interrupted by user.")
    except CameraError as e:
        print(f"Camera error: {e}")
    except AnonymizationError as e:
        print(f"Anonymization error: {e}")
    except Exception as e:
        print(f"Error in anonymization: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Use the centralized window closing utility
        safely_close_windows(WINDOW_NAME, video_capture)
        print("Returned to main menu.")


if __name__ == "__main__":
    # Run the demo function if this module is executed directly
    run_anonymization_demo()
