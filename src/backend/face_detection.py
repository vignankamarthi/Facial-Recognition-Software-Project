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

# Add parent directory to path to ensure imports work in all contexts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities with logging
from src.utils.common_utils import (
    safely_close_windows,
    handle_opencv_error,
    CameraError,
    DetectionError,
    format_error,
    create_resizable_window,
)
from src.utils.logger import get_logger, log_exception, log_method_call

# Initialize logger for this module
logger = get_logger(__name__)

# Import configuration
try:
    from src.utils.config import get_config
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
    def detect_faces(self, frame):
        """
        Detect faces in a given frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Image frame to process

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
        # Validate input frame
        if frame is None:
            error_msg = "Cannot detect faces in None frame"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        if frame.size == 0:
            error_msg = "Empty frame provided for face detection"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Convert the image from BGR color (OpenCV's default format) to RGB (face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all the faces in the current frame
            self.face_locations = face_recognition.face_locations(rgb_frame)
            self.face_encodings = face_recognition.face_encodings(
                rgb_frame, self.face_locations
            )
            
            logger.debug(f"Detected {len(self.face_locations)} faces in frame")
            return self.face_locations, self.face_encodings
            
        except Exception as e:
            error_msg = "Face detection failed"
            log_exception(logger, error_msg, e)
            raise DetectionError(error_msg, str(e), e)

    @log_method_call(logger)
    def draw_face_boxes(self, frame, face_locations, color=(0, 255, 0), thickness=2):
        """
        Draw colored boxes around detected faces.

        Parameters
        ----------
        frame : numpy.ndarray
            Image frame to draw on
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
        # Validate input frame
        if frame is None:
            error_msg = "Cannot draw boxes on None frame"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Validate face locations
        if not isinstance(face_locations, list):
            logger.warning(f"face_locations is not a list: {type(face_locations)}")
            face_locations = list(face_locations) if face_locations else []
        
        # Create a copy of the frame to avoid modifying the original
        display_frame = frame.copy()

        # Draw a box around each detected face
        for i, (top, right, bottom, left) in enumerate(face_locations):
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, thickness)
            logger.debug(f"Drew face box {i+1} at location ({left}, {top}, {right}, {bottom})")

        return display_frame

    @handle_opencv_error
    @log_method_call(logger)
    def detect_faces_webcam(self, anonymize=False, anonymizer=None):
        """
        Start a webcam feed and detect faces in real-time.
        
        This method captures frames from the webcam, detects faces in each frame,
        and displays the results in real-time. If anonymization is enabled,
        it applies the anonymization filter to detected faces.

        Parameters
        ----------
        anonymize : bool, optional
            Whether to anonymize detected faces (default: False)
        anonymizer : FaceAnonymizer, optional
            Instance of FaceAnonymizer to use for anonymization.
            If None but anonymize=True, a default one will be created.

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
              - "error" : str, optional
                Error message (if an error occurred)
        
        Examples
        --------
        >>> detector = FaceDetector()
        >>> detector.detect_faces_webcam()  # Interactive webcam demo
        >>> detector.detect_faces_webcam(anonymize=True)  # With anonymization
        """
        # Initialize webcam
        video_capture = None
        
        # Track session statistics
        start_time = time.time()
        frames_processed = 0
        total_faces_detected = 0
        result_dict = {}
        
        # If anonymize is True but no anonymizer provided, create one
        if anonymize and anonymizer is None:
            logger.info("Creating default anonymizer")
            from .anonymization import FaceAnonymizer
            anonymizer = FaceAnonymizer()
        
        try:
            # Initialize webcam
            logger.info("Opening webcam")
            # Try different camera indices as macOS might use different indices
            for camera_index in [0, 1, -1]:
                video_capture = cv2.VideoCapture(camera_index)
                if video_capture.isOpened():
                    logger.info(f"Successfully opened webcam with index {camera_index}")
                    break
                else:
                    logger.warning(f"Failed to open webcam with index {camera_index}")
            
            if not video_capture.isOpened():
                error_msg = format_error("Camera", "Could not open webcam")
                logger.error(error_msg)
                raise CameraError(error_msg)

            logger.info("Webcam opened successfully")
            logger.info("Press 'q' to quit face detection")
            print("Press 'q' to quit...")
            
            # Create a resizable window using utility function
            create_resizable_window(WINDOW_NAME)
            
            # Variables for key feedback display
            last_key = None
            key_press_time = time.time()
            show_key_message = False

            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()
                frames_processed += 1

                if not ret:
                    error_msg = format_error("Camera", "Failed to capture frame")
                    logger.error(error_msg)
                    break

                # Detect faces in the frame
                try:
                    face_locations, _ = self.detect_faces(frame)
                    total_faces_detected += len(face_locations)
                except Exception as e:
                    logger.error(f"Error detecting faces: {e}")
                    face_locations = []

                # Create a frame to display
                display_frame = frame.copy()

                if anonymize:
                    # If anonymization is enabled, use the provided or created anonymizer
                    try:
                        display_frame = anonymizer.anonymize_frame(frame, face_locations)
                    except Exception as e:
                        logger.error(f"Error anonymizing faces: {e}")
                        # Fallback to just drawing boxes
                        display_frame = self.draw_face_boxes(frame, face_locations)
                else:
                    # Otherwise, just draw boxes around the faces
                    display_frame = self.draw_face_boxes(frame, face_locations)

                    # Add semi-transparent background for text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (5, 5), (350, 40), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                    
                    # Display information about detected faces
                    text = f"Faces detected: {len(face_locations)}"
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                
                # Add controls reminder if not already shown by anonymizer
                if not anonymize:
                    cv2.putText(
                        display_frame,
                        "Press 'q' to quit",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                
                # Show key press feedback on screen
                if show_key_message and time.time() - key_press_time < 2.0:  # Show for 2 seconds
                    key_text = f"KEY PRESSED: {last_key}" if last_key else ""
                    cv2.rectangle(display_frame, (10, 120), (400, 160), (0, 0, 0), -1)  # Background
                    cv2.putText(
                        display_frame,
                        key_text,
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,  # Larger font
                        (0, 255, 255),  # Yellow
                        2,
                    )

                # Display the resulting frame
                cv2.imshow(WINDOW_NAME, display_frame)
                
                # Use a longer wait time and try to get key input
                key = cv2.waitKey(WAIT_KEY_DELAY) & 0xFF
                
                # Process key presses with debugging
                if key not in [255, 0]:  # Valid key pressed
                    # Print key info to console
                    if 32 <= key <= 126:  # Printable ASCII
                        key_char = chr(key)
                        logger.debug(f"Key pressed: {key} (ASCII: {key_char})")
                        last_key = key_char
                    else:
                        logger.debug(f"Key pressed: {key} (non-printable)")
                        last_key = f"Code: {key}"
                        
                    # Update key display timing
                    key_press_time = time.time()
                    show_key_message = True
                    
                    # Process specific keys
                    if anonymize:
                        if key == ord("b") or key == ord("B"):  # b or B
                            logger.info("Switching to blur mode")
                            print("Switching to blur mode")
                            anonymizer.set_method("blur")
                        elif key == ord("p") or key == ord("P"):  # p or P
                            logger.info("Switching to pixelate mode")
                            print("Switching to pixelate mode")
                            anonymizer.set_method("pixelate")
                        elif key == ord("m") or key == ord("M"):  # m or M
                            logger.info("Switching to mask mode")
                            print("Switching to mask mode")
                            anonymizer.set_method("mask")
                    
                    if key == ord("q") or key == ord("Q") or key == 27:  # q, Q, or ESC
                        logger.info("Quitting face detection")
                        print("Quitting face detection...")
                        break
                
        except KeyboardInterrupt:
            logger.info("Face detection interrupted by user")
            print("\nFace detection interrupted by user.")
            result_dict["error"] = "Interrupted by user"
        except DetectionError as e:
            logger.error(f"Detection error: {e}")
            print(f"Detection error: {e}")
            result_dict["error"] = f"Detection error: {str(e)}"
        except CameraError as e:
            logger.error(f"Camera error: {e}")
            print(f"Camera error: {e}")
            result_dict["error"] = f"Camera error: {str(e)}"
        except Exception as e:
            error_msg = f"Error in face detection: {e}"
            log_exception(logger, error_msg, e)
            print(error_msg)
            result_dict["error"] = error_msg
        finally:
            # Calculate session statistics
            end_time = time.time()
            duration = end_time - start_time
            
            # Update result dictionary
            result_dict.update({
                "face_count": total_faces_detected,
                "frames_processed": frames_processed,
                "duration": duration
            })
            
            logger.info(f"Face detection session ended: processed {frames_processed} frames " +
                       f"with {total_faces_detected} total faces in {duration:.2f} seconds")
            
            # Use the centralized window closing utility
            safely_close_windows(WINDOW_NAME, video_capture)
            logger.info("Returned to main menu")
            print("Returned to main menu.")
            
            success = "error" not in result_dict
            return success, result_dict
            
    @log_method_call(logger)
    def process_image(self, image_path):
        """
        Process a single image file, detecting and displaying faces.
        
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
            
            # Draw boxes around faces
            display_image = self.draw_face_boxes(image, face_locations)
            
            # Add title with information
            title = f"Detected {len(face_locations)} face(s)"
            
            # Add semi-transparent background for text
            h, w = display_image.shape[:2]
            overlay = display_image.copy()
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
            
            # Add text
            cv2.putText(
                display_image,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            
            # Show the image
            create_resizable_window("Face Detection")
            cv2.imshow("Face Detection", display_image)
            logger.info("Showing image with detected faces. Press any key to continue.")
            print("Showing image with detected faces. Press any key to continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Return results
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
