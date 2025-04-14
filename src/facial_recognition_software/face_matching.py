"""
Face Matching Module

This module provides functionality for comparing detected faces against a database
of known faces for identification purposes.
"""

import os
import cv2
import numpy as np
import face_recognition
import time  # Add time module for reliable key handling
from .face_detection import FaceDetector


class FaceMatcher:
    """A class to handle face matching operations."""

    def __init__(self, known_faces_dir="./data/known_faces"):
        """
        Initialize the face matcher with known faces.

        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_faces_dir = known_faces_dir
        self.load_known_faces()

    def load_known_faces(self):
        """
        Load known faces from the specified directory.
        Each image file should be named with the person's name (e.g., john_smith.jpg).
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

    def identify_faces(self, frame, face_locations, face_encodings):
        """
        Compare detected faces against known faces to identify them.

        Args:
            frame (numpy.ndarray): Image frame
            face_locations (list): List of face location tuples
            face_encodings (list): List of face encodings

        Returns:
            numpy.ndarray: Frame with names drawn
            list: List of identified names
        """
        # Create a copy of the frame
        display_frame = frame.copy()
        face_names = []

        # Check if we have known faces to compare against
        if not self.known_face_encodings:
            for top, right, bottom, left in face_locations:
                name = "Unknown"
                face_names.append(name)

                # Draw a box around the face
                cv2.rectangle(
                    display_frame, (left, top), (right, bottom), (0, 0, 255), 2
                )

                # Draw a label with the name below the face
                cv2.rectangle(
                    display_frame,
                    (left, bottom - 35),
                    (right, bottom),
                    (0, 0, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    display_frame,
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
                    self.known_face_encodings, face_encoding, 0.6
                )
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    confidence = 1 - face_distances[best_match_index]

                    if matches[best_match_index] and confidence > 0.6:
                        name = self.known_face_names[best_match_index]
                        name = f"{name} ({confidence:.2f})"
                        box_color = (0, 255, 0)  # Green for recognized faces
                    else:
                        box_color = (0, 0, 255)  # Red for unknown faces
                else:
                    box_color = (0, 0, 255)

                face_names.append(name)

                # Draw a box around the face
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)

                # Draw a label with the name below the face
                cv2.rectangle(
                    display_frame,
                    (left, bottom - 35),
                    (right, bottom),
                    box_color,
                    cv2.FILLED,
                )
                cv2.putText(
                    display_frame,
                    name,
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.8,
                    (255, 255, 255),
                    1,
                )

        return display_frame, face_names

    def match_faces_webcam(self):
        """
        Start a webcam feed and perform real-time face recognition.

        Returns:
            None
        """
        # Initialize face detector
        detector = FaceDetector()

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
                face_locations, face_encodings = detector.detect_faces(frame)

                # Identify the faces
                display_frame, face_names = self.identify_faces(
                    frame, face_locations, face_encodings
                )

                # Add semi-transparent background for instructions
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (5, 5), (270, 95), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                
                # Display information with better visibility
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
                
                # Add controls reminder with better spacing
                cv2.putText(
                    display_frame,
                    "Press 'q' to quit",
                    (10, 70),
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
                        print("Quitting face matching...")
                        break
                
        except KeyboardInterrupt:
            print("\nFace matching interrupted by user.")
        except Exception as e:
            print(f"Error in face matching: {e}")
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
        matcher = FaceMatcher()
        matcher.match_faces_webcam()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting.")
    finally:
        # Make sure to release any OpenCV resources
        cv2.destroyAllWindows()
