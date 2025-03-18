"""
Anonymization Module

This module provides functionality for anonymizing detected faces in images 
and video streams to protect privacy.
"""

import cv2
import numpy as np


class FaceAnonymizer:
    """A class to handle face anonymization operations."""
    
    def __init__(self, method='blur', intensity=25):
        """
        Initialize the face anonymizer.
        
        Args:
            method (str): Anonymization method ('blur', 'pixelate', or 'mask')
            intensity (int): Intensity of the anonymization effect
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
        
        if current_method == 'blur':
            # Apply Gaussian blur
            blurred_face = cv2.GaussianBlur(face_img, (self.intensity, self.intensity), 0)
            result_frame[top:bottom, left:right] = blurred_face
            
        elif current_method == 'pixelate':
            # Pixelate by downscaling and upscaling
            height, width = face_img.shape[:2]
            
            # Calculate the scaling factor based on intensity
            scale_factor = max(1, self.intensity // 5)
            
            # Downscale
            temp = cv2.resize(face_img, (width // scale_factor, height // scale_factor),
                             interpolation=cv2.INTER_LINEAR)
            
            # Upscale with nearest neighbor to create pixelation effect
            pixelated_face = cv2.resize(temp, (width, height),
                                       interpolation=cv2.INTER_NEAREST)
            
            result_frame[top:bottom, left:right] = pixelated_face
            
        elif current_method == 'mask':
            # Create a solid color mask
            mask_color = (0, 0, 0)  # Black mask
            cv2.rectangle(result_frame, (left, top), (right, bottom), mask_color, -1)
            
            # Optional: Draw a face icon or text
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2
            radius = min(right - left, bottom - top) // 4
            
            # Draw a simple face icon
            cv2.circle(result_frame, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(result_frame, (center_x - radius//2, center_y - radius//3), 
                      radius//5, (0, 0, 0), -1)  # Left eye
            cv2.circle(result_frame, (center_x + radius//2, center_y - radius//3), 
                      radius//5, (0, 0, 0), -1)  # Right eye
            cv2.ellipse(result_frame, (center_x, center_y + radius//3), 
                       (radius//2, radius//3), 0, 0, 180, (0, 0, 0), -1)  # Mouth
            
        # Add a visual indicator that this face is anonymized
        cv2.rectangle(result_frame, (left, top), (right, bottom), (0, 255, 255), 2)
        cv2.putText(result_frame, "Anonymized", (left, top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
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
            
        # Add indicator for anonymization mode
        cv2.putText(result_frame, f"Anonymization: {self.method.capitalize()}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result_frame
    
    def set_method(self, method):
        """
        Change the anonymization method.
        
        Args:
            method (str): New anonymization method
        """
        valid_methods = ['blur', 'pixelate', 'mask']
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
            'original': frame.copy(),
            'blur': self.anonymize_face(frame, face_location, 'blur'),
            'pixelate': self.anonymize_face(frame, face_location, 'pixelate'),
            'mask': self.anonymize_face(frame, face_location, 'mask')
        }
        
        return methods


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    from face_detection import FaceDetector
    
    detector = FaceDetector()
    anonymizer = FaceAnonymizer()
    
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    print("Press 'b' for blur, 'p' for pixelate, 'm' for mask, 'q' to quit...")
    
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
        cv2.imshow('Video', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('b'):
            anonymizer.set_method('blur')
        elif key == ord('p'):
            anonymizer.set_method('pixelate')
        elif key == ord('m'):
            anonymizer.set_method('mask')
        elif key == ord('q'):
            break
    
    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
