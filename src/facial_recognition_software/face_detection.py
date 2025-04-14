"""
Face Detection Module

This module provides functionality for detecting faces in images and video streams.
It uses OpenCV and face_recognition libraries to identify facial features.
"""

import cv2
import face_recognition
import numpy as np
import time  # Add time module for reliable key handling


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

    def detect_faces_webcam(self, anonymize=False):
        """
        Start a webcam feed and detect faces in real-time.

        Args:
            anonymize (bool): Whether to blur detected faces (default is False)
        If True, the detected faces will be anonymized using a blurring effect.

        Returns:
            None
        """
        # Initialize webcam
        video_capture = None
        
        try:
            # Initialize webcam
            video_capture = cv2.VideoCapture(0)

            if not video_capture.isOpened():
                print("Error: Could not open webcam.")
                return

            print("Press 'q' to quit...")
            
            # Create a named window and set it to normal (resizable) mode
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            
            # Set window as topmost to ensure it receives keyboard focus
            cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
            
            # Variables for key feedback display
            last_key = None
            key_press_time = time.time()
            show_key_message = False

            while True:
                # Capture frame-by-frame
                ret, frame = video_capture.read()

                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Detect faces in the frame
                face_locations, _ = self.detect_faces(frame)

                # Create a frame to display
                display_frame = frame.copy()

                if anonymize:
                    # If anonymization is enabled, blur the faces
                    from .anonymization import FaceAnonymizer

                    anonymizer = FaceAnonymizer()
                    display_frame = anonymizer.anonymize_frame(frame, face_locations)
                else:
                    # Otherwise, just draw boxes around the faces
                    display_frame = self.draw_face_boxes(frame, face_locations)

                # Display information about detected faces
                text = f"Faces detected: {len(face_locations)}"
                cv2.putText(
                    display_frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                
                # Add controls reminder
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
                cv2.imshow("Video", display_frame)

                # Use a longer wait time and try to get key input
                key = cv2.waitKey(100) & 0xFF  # Even longer wait (100ms)
                
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
                    if key == ord("q") or key == ord("Q") or key == 27:  # q, Q, or ESC
                        print("Quitting face detection...")
                        break
                
        except KeyboardInterrupt:
            print("\nFace detection interrupted by user.")
        except Exception as e:
            print(f"Error in face detection: {e}")
        finally:
            # Always ensure webcam is released and windows are closed
            print("Cleaning up resources...")
            if video_capture is not None and video_capture.isOpened():
                video_capture.release()
            
            print("Closing windows...")
            # Multiple attempts to close windows with forced focus and delays
            cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)  # Try to force focus
            cv2.waitKey(200)  # Longer wait
            
            # First try normal window closure
            cv2.destroyWindow("Video")  
            time.sleep(0.2)  # Sleep directly instead of waitKey
            
            # Second attempt with all windows
            cv2.destroyAllWindows()
            time.sleep(0.2)
            
            # Third attempt with a loop and delays
            for i in range(3):
                cv2.waitKey(200)  # Even longer wait
                cv2.destroyAllWindows()
                time.sleep(0.2)  # Direct sleep
                
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
        cv2.destroyAllWindows()
