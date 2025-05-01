"""
Face Anonymization page for the Streamlit interface.

This module provides the Streamlit UI for the face anonymization feature, allowing users
to apply privacy-preserving filters to faces in images or webcam feed.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import time

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

# Import core functionality
from src.backend.face_detection import FaceDetector
from src.backend.anonymization import FaceAnonymizer
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import UI components
from src.ui.streamlit.components import (
    webcam_component,
    image_upload_component,
    anonymization_config_panel,
    before_after_comparison
)

# Setup logging
logger = get_logger(__name__)

# Get configuration
config = get_config()

def process_anonymization_frame(frame):
    """
    Process a frame for face anonymization.
    
    Parameters
    ----------
    frame : numpy.ndarray
        Input frame in BGR format
    
    Returns
    -------
    tuple
        (processed_frame, metadata_dict)
    """
    # Get detector and anonymizer instances
    detector = FaceDetector()
    
    # Get anonymization parameters from session state
    anon_params = st.session_state.config["anonymization"]
    
    # Create anonymizer with current settings
    anonymizer = FaceAnonymizer(
        method=anon_params["method"],
        intensity=anon_params["intensity"]
    )
    
    # Start timing for performance measurement
    start_time = time.time()
    
    try:
        # Make a copy of the original frame for later
        original_frame = frame.copy()
        
        # Detect faces
        face_locations, _ = detector.detect_faces(frame)
        
        # Anonymize faces if any detected
        if face_locations:
            processed_frame = anonymizer.anonymize_frame(frame, face_locations)
        else:
            # No faces detected, just use the input frame
            processed_frame = frame
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create metadata dictionary
        metadata = {
            "face_count": len(face_locations),
            "method": anon_params["method"].capitalize(), 
            "intensity": anon_params["intensity"],
            "processing_time": f"{processing_time*1000:.1f} ms"
        }
        
        # Store original frame for preview if needed
        if anon_params["preview"] and len(face_locations) > 0:
            st.session_state["original_frame"] = original_frame
            st.session_state["anonymized_frame"] = processed_frame
        
        return processed_frame, metadata
    
    except Exception as e:
        logger.error(f"Error in face anonymization: {e}")
        return frame, {"error": str(e)}

def face_anonymization_page():
    """Render the face anonymization page."""
    
    st.markdown("# Face Anonymization")
    st.markdown("Apply privacy-preserving filters to faces in images or webcam feed.")
    
    # Anonymization settings
    with st.expander("Anonymization Settings", expanded=True):
        updated_config = anonymization_config_panel(
            st.session_state.config["anonymization"], 
            key_prefix="fa_"
        )
        st.session_state.config["anonymization"] = updated_config
    
    # Create tabs for image upload vs webcam
    tab1, tab2 = st.tabs(["Upload Image", "Use Webcam"])
    
    # Image upload tab
    with tab1:
        st.markdown("### Upload an image to anonymize faces")
        st.markdown("Upload an image containing one or more faces to apply privacy filters.")
        
        # Use the image upload component
        image_upload_component(process_anonymization_frame, key_prefix="fa_")
        
        # If we have stored frames for preview, show them
        if "original_frame" in st.session_state and "anonymized_frame" in st.session_state:
            if st.session_state.config["anonymization"]["preview"]:
                # Show before/after comparison
                before_after_comparison(
                    st.session_state["original_frame"],
                    st.session_state["anonymized_frame"],
                    title=f"Before/After: {st.session_state.config['anonymization']['method'].capitalize()} Method",
                    key_prefix="fa_comparison_"
                )
    
    # Webcam tab
    with tab2:
        st.markdown("### Use webcam feed for real-time face anonymization")
        st.markdown("Use your camera to anonymize faces in real-time. Position yourself or others in front of the camera.")
        
        # Use the webcam component
        webcam_component(process_anonymization_frame, key_prefix="fa_webcam_")
    
    # Information panel
    with st.expander("About Face Anonymization", expanded=False):
        st.markdown("""
        ### How Face Anonymization Works
        
        Face anonymization applies privacy-preserving filters to detected faces to protect individuals' identities while preserving the overall context of the image. This application offers three anonymization methods:
        
        1. **Blur**: Applies a Gaussian blur filter to face regions
           - Higher intensity creates a stronger blur effect
           - Preserves general facial structure while hiding identifying details
           
        2. **Pixelate**: Creates a blocky, pixelated effect over faces
           - Higher intensity creates larger, more obscuring pixels
           - Creates a "censored" appearance commonly used in media
           
        3. **Mask**: Replaces faces with a solid overlay
           - Completely obscures the face with a black mask
           - Adds a simple face icon for visual context
        
        ### Privacy Considerations
        
        Face anonymization is important in several contexts:
        
        - **Privacy protection**: Respecting individuals' right to privacy
        - **Ethical data usage**: Demonstrating responsible use of facial images
        - **Consent management**: Addressing situations where consent for facial recognition is not given
        - **Sensitive contexts**: Protecting identities in sensitive scenarios
        
        ### Settings Explained
        
        - **Method**: The type of anonymization effect to apply
        - **Intensity**: Strength of the anonymization effect (1-100)
        - **Show Boxes**: Whether to display colored outlines around anonymized faces
        - **Show Labels**: Whether to show text labels indicating the anonymization method
        
        ### Tips for Best Results
        
        - Ensure good lighting for reliable face detection
        - For blur method, intensity 70+ typically ensures privacy
        - For pixelate method, intensity 50+ creates an effective anonymization
        - The mask method provides the most complete anonymization
        """)

if __name__ == "__main__":
    face_anonymization_page()
