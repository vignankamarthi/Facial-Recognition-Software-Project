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
import time

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

# Import core functionality
from src.backend.face_detection import FaceDetector
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import UI components
from src.ui.streamlit.components import (
    webcam_component,
    image_upload_component,
    detection_config_panel
)

# Setup logging
logger = get_logger(__name__)

# Get configuration
config = get_config()

def process_detection_frame(frame):
    """
    Process a frame for face detection.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame in BGR format
    
    Returns
    -------
    tuple
        (processed_frame, metadata_dict)
    """
    # Get detector instance
    detector = FaceDetector()
    
    # Get detection parameters from session state
    detection_params = st.session_state.config["detection"]
    
    # Start timing for performance measurement
    start_time = time.time()
    
    try:
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(frame)
        
        # Draw boxes on the frame
        processed_frame = detector.draw_face_boxes(frame, face_locations)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create metadata dictionary
        metadata = {
            "face_count": len(face_locations),
            "face_locations": face_locations,
            "processing_time": f"{processing_time*1000:.1f} ms",
            "fps_capacity": f"{1/processing_time:.1f}" if processing_time > 0 else "N/A"
        }
        
        return processed_frame, metadata
    
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return frame, {"error": str(e)}

def face_detection_page():
    """Render the face detection page."""
    
    st.markdown("# Face Detection")
    st.markdown("Detect faces in images or using your webcam feed.")
    
    # Detection settings
    with st.expander("Detection Settings", expanded=False):
        updated_config = detection_config_panel(
            st.session_state.config["detection"],
            key_prefix="fd_"
        )
        st.session_state.config["detection"] = updated_config
    
    # Create tabs for image upload vs webcam
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    # Image upload tab
    with tab1:
        st.markdown("### Upload an image to detect faces")
        st.markdown("Upload an image containing one or more faces to detect and analyze them.")
        
        # Use the image upload component
        image_upload_component(process_detection_frame, key_prefix="fd_")
    
    # Webcam tab
    with tab2:
        st.markdown("### Use webcam feed for real-time face detection")
        st.markdown("Use your camera to detect faces in real-time. Position yourself or others in front of the camera.")
        
        # Use the webcam component
        webcam_component(process_detection_frame, key_prefix="fd_")
    
    # Information panel
    with st.expander("About Face Detection", expanded=False):
        st.markdown("""
        ### How Face Detection Works
        
        Face detection uses computer vision algorithms to identify human faces in digital images. This application uses the `face_recognition` library which employs a model called Histogram of Oriented Gradients (HOG) to find face patterns.
        
        The process involves:
        
        1. **Image Processing**: Converting the image to grayscale and normalizing it
        2. **Feature Extraction**: Calculating HOG features that represent face shapes
        3. **Classification**: Using a trained classifier to determine if an area contains a face
        4. **Bounding Box Creation**: Drawing rectangles around detected face regions
        
        ### Settings Explained
        
        - **Detection Confidence**: Higher values make detection more strict, reducing false positives but potentially missing some faces.
        - **HOG Detector**: Faster but less accurate than CNN
        - **CNN Batch Size**: For CNN detector - larger values use more memory but can be faster
        
        ### Tips for Best Results
        
        - Ensure good lighting conditions for better detection
        - Face the camera directly for webcam usage
        - For group photos, make sure all faces are clearly visible
        - Higher resolution images generally provide better results
        """)

if __name__ == "__main__":
    face_detection_page()
