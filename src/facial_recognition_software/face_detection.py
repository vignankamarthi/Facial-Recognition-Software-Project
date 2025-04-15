"""
Face Detection Module

This module provides functionality for detecting faces in images and video streams.
It uses OpenCV and face_recognition libraries to identify facial features.
"""

import cv2
import face_recognition
import numpy as np
import time
import sys
import os

# Define local fallback constants that will be used if imports fail
_WINDOW_NAME = "Video"
_WAIT_KEY_DELAY = 100

# Add parent directory to path to ensure imports work in all contexts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities safely with failover to local defaults
try:
    from utilities.common_utils import (
        safely_close_windows, handle_opencv_error, CameraError, DetectionError,
        format_error, create_resizable_window
    )
    from utilities.config import WINDOW_NAME, WAIT_KEY_DELAY, initialize_opencv_constants
    # Initialize OpenCV constants after cv2 is imported
    initialize_opencv_constants()
except ImportError as e:
    # Provide dummy implementations if imports fail
    print(f"Warning: Could not import utilities. Using fallback implementations. Error: {e}")
    
    WINDOW_NAME = _WINDOW_NAME
    WAIT_KEY_DELAY = _WAIT_KEY_DELAY
    
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
    
    class DetectionError(Exception):
        """Fallback exception for detection errors."""
        pass
    
    def format_error(error_type, message):
        return f"ERROR: {message}"
    
    def create_resizable_window(window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name


class FaceDetector:
    """A class to handle face detection operations."""

    def __init__(self):
        """Initialize the face detector."""
        self.face_locations = []
        self.face_encodings = []

    def detect_faces(self, frame):
        """
        Detect faces in a given frame.

        Args:
            frame (numpy.ndarray): Image frame to process

        Returns:
            tuple: (face_locations, face_encodings)
        """
        # Convert the image from BGR color (OpenCV's default format) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces in the current frame
        self.face_locations = face_recognition.face_locations(rgb_frame)
        self.face_encodings = face_recognition.face_encodings(
            rgb_frame, self.face_locations
        )

        return self.face_locations, self.face_encodings

    def draw_face_boxes(self, frame, face_locations, color=(0, 255, 0), thickness=2):
        """
        Draw green boxes around detected faces.

        Args:
            frame (numpy.ndarray): Image frame to draw on
            face_locations (list): List of face location tuples (top, right, bottom, left)
            color (tuple): BGR color for the box (green is the default)
            thickness (int): Line thickness (default is 2)
    

        Returns:
            numpy.ndarray: Frame with boxes drawn
        """
        # Create a copy of the frame to avoid modifying the original
        display_frame = frame.copy()

        # Draw a box around each detected face
        for top, right, bottom, left in face_locations:
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, thickness)

        return display_frame

    @handle_opencv_error
    def detect_faces_webcam(self, anonymize=False, anonymizer=None):
        """
        Start a webcam feed and detect faces in real-time.

        Args:
            anonymize (bool): Whether to blur detected faces (default is False)
            anonymizer (FaceAnonymizer): Instance of FaceAnonymizer to use for anonymization

        Returns:
            None
        """
        # Initialize webcam
        video_capture = None
        
        # If anonymize is True but no anonymizer provided, create one
        if anonymize and anonymizer is None:
            from .anonymization import FaceAnonymizer
            anonymizer = FaceAnonymizer()
        
        try:
            # Initialize webcam
            video_capture = cv2.VideoCapture(0)

            if not video_capture.isOpened():
                error_msg = format_error("Camera", "Could not open webcam")
                print(error_msg)
                raise CameraError(error_msg)

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

                if not ret:
                    error_msg = format_error("Camera", "Failed to capture frame")
                    print(error_msg)
                    break

                # Detect faces in the frame
                face_locations, _ = self.detect_faces(frame)

                # Create a frame to display
                display_frame = frame.copy()

                if anonymize:
                    # If anonymization is enabled, use the provided or created anonymizer
                    display_frame = anonymizer.anonymize_frame(frame, face_locations)
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
                        print(f"Key pressed: {key} (ASCII: {key_char})")
                        last_key = key_char
                    else:
                        print(f"Key pressed: {key} (non-printable)")
                        last_key = f"Code: {key}"
                        
                    # Update key display timing
                    key_press_time = time.time()
                    show_key_message = True
                    
                    # Process specific keys
                    if anonymize:
                        if key == ord("b") or key == ord("B"):  # b or B
                            print("Switching to blur mode")
                            anonymizer.set_method("blur")
                        elif key == ord("p") or key == ord("P"):  # p or P
                            print("Switching to pixelate mode")
                            anonymizer.set_method("pixelate")
                        elif key == ord("m") or key == ord("M"):  # m or M
                            print("Switching to mask mode")
                            anonymizer.set_method("mask")
                    
                    if key == ord("q") or key == ord("Q") or key == 27:  # q, Q, or ESC
                        print("Quitting face detection...")
                        break
                
        except KeyboardInterrupt:
            print("\nFace detection interrupted by user.")
        except DetectionError as e:
            print(f"Detection error: {e}")
        except CameraError as e:
            print(f"Camera error: {e}")
        except Exception as e:
            print(f"Error in face detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Use the centralized window closing utility
            safely_close_windows(WINDOW_NAME, video_capture)
            print("Returned to main menu.")


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    try:
        detector = FaceDetector()
        detector.detect_faces_webcam()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    finally:
        # Make sure to release any OpenCV resources
        safely_close_windows()
