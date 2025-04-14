"""
Anonymization Module

This module provides functionality for anonymizing detected faces in images
and video streams to protect privacy.
"""

import cv2
import numpy as np
import time  # Add time module for more reliable key handling


class FaceAnonymizer:
    """A class to handle face anonymization operations."""

    def __init__(self, method="blur", intensity=90):
        """
        Initialize the face anonymizer.

        Args:
            method (str): Anonymization method ('blur', 'pixelate', or 'mask') with default 'blur'
            intensity (int): Intensity of the anonymization effect with range 1-100 with default 90
        """
        self.method = method
        self.intensity = intensity

    def anonymize_face(self, frame, face_location, method=None):
        """
        Apply anonymization to a single face.

        Args:
            frame (numpy.ndarray): Image frame
            face_location (tuple): Face location tuple (top, right, bottom, left)
            method (str, optional): Override the default anonymization method

        Returns:
            numpy.ndarray: Frame with anonymized face
        """
        # Make a copy of the frame to avoid modifying the original
        result_frame = frame.copy()

        # Extract face location coordinates
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]

        # Use specified method or default
        current_method = method if method else self.method

        if current_method == "blur":
            # Apply Gaussian blur
            # Ensure kernel size is odd (required by GaussianBlur)
            kernel_size = (
                self.intensity if self.intensity % 2 == 1 else self.intensity + 1
            )
            blurred_face = cv2.GaussianBlur(face_img, (kernel_size, kernel_size), 0)
            result_frame[top:bottom, left:right] = blurred_face

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

            result_frame[top:bottom, left:right] = pixelated_face

        elif current_method == "mask":
            # Create a solid color mask
            mask_color = (0, 0, 0)  # Black mask
            cv2.rectangle(result_frame, (left, top), (right, bottom), mask_color, -1)

            # Optional: Draw a face icon or text
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            radius = min(right - left, bottom - top) // 4

            # Draw a simple face icon
            cv2.circle(result_frame, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(
                result_frame,
                (center_x - radius // 2, center_y - radius // 3),
                radius // 5,
                (0, 0, 0),
                -1,
            )  # Left eye
            cv2.circle(
                result_frame,
                (center_x + radius // 2, center_y - radius // 3),
                radius // 5,
                (0, 0, 0),
                -1,
            )  # Right eye
            cv2.ellipse(
                result_frame,
                (center_x, center_y + radius // 3),
                (radius // 2, radius // 3),
                0,
                0,
                180,
                (0, 0, 0),
                -1,
            )  # Mouth

        # Add a visual indicator that this face is anonymized
        cv2.rectangle(result_frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(
            result_frame,
            "Anonymized",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

        return result_frame

    def anonymize_frame(self, frame, face_locations):
        """
        Apply anonymization to all faces in a frame.

        Args:
            frame (numpy.ndarray): Image frame
            face_locations (list): List of face location tuples

        Returns:
            numpy.ndarray: Frame with all faces anonymized
        """
        result_frame = frame.copy()

        for face_location in face_locations:
            result_frame = self.anonymize_face(result_frame, face_location)

        # Add semi-transparent background for menu text
        overlay = result_frame.copy()
        # Make the box for showing information
        cv2.rectangle(overlay, (5, 5), (400, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0, result_frame)
        
        # Add indicator for anonymization mode
        cv2.putText(
            result_frame,
            f"Anonymization: {self.method.capitalize()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return result_frame

    def set_method(self, method):
        """
        Change the anonymization method.

        Args:
            method (str): New anonymization method
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

        Args:
            intensity (int): New intensity value
        """
        if 1 <= intensity <= 100:
            self.intensity = intensity
            print(f"Anonymization intensity set to: {intensity}")
        else:
            print("Intensity must be between 1 and 100")

    def demonstrate_methods(self, frame, face_location):
        """
        Demonstrate different anonymization methods on a single face.

        Args:
            frame (numpy.ndarray): Image frame
            face_location (tuple): Face location tuple

        Returns:
            dict: Dictionary of frames with different anonymization methods
        """
        methods = {
            "original": frame.copy(),
            "blur": self.anonymize_face(frame, face_location, "blur"),
            "pixelate": self.anonymize_face(frame, face_location, "pixelate"),
            "mask": self.anonymize_face(frame, face_location, "mask"),
        }

        return methods


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    from .face_detection import FaceDetector

    detector = FaceDetector()
    anonymizer = FaceAnonymizer()

    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press Ctrl+C to quit...")
    
    # Create a named window and set it to normal (resizable) mode
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    
    # Set window as topmost to ensure it receives keyboard focus
    cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)

    # Main processing loop
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Detect faces in the frame
            face_locations, _ = detector.detect_faces(frame)

            # Anonymize the faces
            display_frame = anonymizer.anonymize_frame(frame, face_locations)

            # Display the resulting frame
            cv2.imshow("Video", display_frame)
            
            # Short wait time
            cv2.waitKey(10)
            
    except KeyboardInterrupt:
        print("\nAnonymization interrupted by user.")
    finally:
        # Release the webcam and close all windows
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
