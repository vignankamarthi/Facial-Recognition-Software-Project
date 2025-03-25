"""
Face Detection Module

This module provides functionality for detecting faces in images and video streams.
It uses OpenCV and face_recognition libraries to identify facial features.
"""

import cv2
import face_recognition
import numpy as np


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
        Draw boxes around detected faces.

        Args:
            frame (numpy.ndarray): Image frame to draw on
            face_locations (list): List of face location tuples (top, right, bottom, left)
            color (tuple): BGR color for the box
            thickness (int): Line thickness

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
            anonymize (bool): Whether to blur detected faces

        Returns:
            None
        """
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit...")

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
                from src.facial_recognition_software.anonymization import FaceAnonymizer

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

            # Display the resulting frame
            cv2.imshow("Video", display_frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the webcam and close all windows
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    detector = FaceDetector()
    detector.detect_faces_webcam()
