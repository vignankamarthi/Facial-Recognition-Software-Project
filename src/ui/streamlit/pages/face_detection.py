"""
Face Detection page for the Streamlit interface.

This module provides the Streamlit UI for the face detection feature, allowing users
to upload images or use webcam feed for face detection.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import tempfile

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

# Import core functionality
from src.backend.face_detection import FaceDetector
from src.utils.config import get_config

# Get configuration
config = get_config()

def face_detection_page():
    """Render the face detection page."""
    
    st.title("Face Detection")
    st.write("Upload an image or use your webcam to detect faces.")
    
    # Create tabs for image upload vs webcam
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    # Initialize face detector
    detector = FaceDetector()
    
    # Image upload tab
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process uploaded image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR (OpenCV format)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
            # Detect faces
            face_locations, _ = detector.detect_faces(image_np)
            
            # Draw boxes on the image
            display_img = detector.draw_face_boxes(image_np, face_locations)
            
            # Convert back to RGB for display
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.image(display_img, caption=f"Detected {len(face_locations)} faces")
            
            # Show face details if faces were detected
            if len(face_locations) > 0:
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    st.write(f"Face {i+1} location: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
    
    # Webcam tab (placeholder)
    with tab2:
        st.write("Webcam integration is coming soon!")
        st.warning("This feature is under development in the Streamlit interface.")

if __name__ == "__main__":
    face_detection_page()
