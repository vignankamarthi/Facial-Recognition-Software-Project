"""
Face Matching page for the Streamlit interface.

This module provides the Streamlit UI for the face matching feature, allowing users
to match detected faces against known reference faces.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
from PIL import Image
import time

# Add the project root to the Python path for imports
sys.path.insert(
    0,
    os.path.abspath(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
    ),
)

# Import core functionality
from src.backend.face_detection import FaceDetector
from src.backend.face_matching import FaceMatcher
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import UI components
from src.ui.streamlit.components import (
    webcam_component,
    streamlit_webcam_component,
    image_upload_component,
    matching_config_panel,
    known_faces_grid,
    add_known_face,
    remove_known_faces,
)

# Setup logging
logger = get_logger(__name__)

# Get configuration
config = get_config()


def process_matching_frame(frame):
    """
    Process a frame for face matching.

    Parameters
    ----------
    frame : numpy.ndarray
        Input frame in BGR format

    Returns
    -------
    tuple
        (processed_frame, metadata_dict)
    """
    # Get detector and matcher instances
    detector = FaceDetector()
    matcher = FaceMatcher(st.session_state.paths["known_faces_dir"])

    # Get matching parameters from session state
    matching_params = st.session_state.config["matching"]

    # If the matcher's threshold is different from the config, update it
    if (
        hasattr(matcher, "set_threshold")
        and matching_params["threshold"] != config.matching.threshold
    ):
        matcher.set_threshold(matching_params["threshold"])

    # Start timing for performance measurement
    start_time = time.time()

    try:
        # Detect faces
        face_locations, face_encodings = detector.detect_faces(frame)

        # Match faces if any detected
        if face_locations:
            processed_frame, face_names = matcher.identify_faces(
                frame, face_locations, face_encodings
            )

            # Count known vs unknown faces
            known_count = sum(1 for name in face_names if name != "Unknown")
            unknown_count = len(face_names) - known_count
        else:
            # No faces detected, just use the input frame
            processed_frame = frame
            face_names = []
            known_count = 0
            unknown_count = 0

        # Calculate processing time
        processing_time = time.time() - start_time

        # Create metadata dictionary
        metadata = {
            "face_count": len(face_locations),
            "identified": known_count,
            "unknown": unknown_count,
            "names": (
                face_names if len(face_names) <= 10 else f"{len(face_names)} faces"
            ),
            "processing_time": f"{processing_time*1000:.1f} ms",
            "fps_capacity": (
                f"{1/processing_time:.1f}" if processing_time > 0 else "N/A"
            ),
        }

        return processed_frame, metadata

    except Exception as e:
        logger.error(f"Error in face matching: {e}")
        return frame, {"error": str(e)}


def face_matching_page():
    """Render the face matching page."""

    st.markdown("# Face Matching")
    st.markdown("Match detected faces against known reference faces.")

    # Check if known faces directory exists
    known_faces_dir = st.session_state.paths["known_faces_dir"]
    if not os.path.exists(known_faces_dir):
        st.warning(f"Known faces directory not found. Creating: {known_faces_dir}")
        os.makedirs(known_faces_dir, exist_ok=True)

    # Create tabs for different functionality
    main_tab, manage_tab = st.tabs(["Match Faces", "Manage Known Faces"])

    with main_tab:
        # Matching settings
        with st.expander("Matching Settings", expanded=False):
            updated_config = matching_config_panel(
                st.session_state.config["matching"], key_prefix="fm_"
            )
            st.session_state.config["matching"] = updated_config

        # Create tabs for image upload vs webcam
        img_tab, cam_tab = st.tabs(["Upload Image", "Use Webcam"])

        # Image upload tab
        with img_tab:
            st.markdown("### Upload an image to match faces")
            st.markdown(
                "Upload an image containing one or more faces to match against known references."
            )

            # Use the image upload component
            image_upload_component(process_matching_frame, key_prefix="fm_")

        # Webcam tab
        with cam_tab:
            st.markdown("### Use webcam feed for real-time face matching")
            st.markdown(
                "Use your camera to match faces in real-time against known references."
            )

            # Check if we're running in Docker/container environment
            use_streamlit_camera = os.environ.get("USE_STREAMLIT_CAMERA", "0") == "1"

            if use_streamlit_camera:
                # Use the Streamlit native webcam component for Docker/container environments
                st.info(
                    "Using Streamlit's built-in camera for compatibility with Docker."
                )
                streamlit_webcam_component(process_matching_frame, key_prefix="fm_")
            else:
                # Use the external window webcam component for direct installations
                webcam_component(process_matching_frame, key_prefix="fm_")

    with manage_tab:
        st.markdown("## Manage Known Faces")

        # Show current known faces
        st.markdown("### Reference Face Library")

        # Get current count of known faces
        known_faces_count = (
            len(
                [
                    f
                    for f in os.listdir(known_faces_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            if os.path.exists(known_faces_dir)
            else 0
        )

        if known_faces_count == 0:
            st.info("No known faces found. Add reference faces using the form below.")
        else:
            st.info(
                f"Found {known_faces_count} known faces. Select faces to manage them."
            )

        # Display known faces grid
        selected_faces = known_faces_grid(known_faces_dir, key_prefix="fm_")

        # Add/Remove faces
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Add New Face")
            add_known_face(known_faces_dir, key_prefix="fm_add_")

        with col2:
            st.markdown("### Remove Selected Faces")
            if selected_faces:
                st.info(f"Selected {len(selected_faces)} faces for removal.")
                remove_known_faces(
                    known_faces_dir, selected_faces, key_prefix="fm_remove_"
                )
            else:
                st.info("Select faces from the grid above to remove them.")

    # Information panel
    with st.expander("About Face Matching", expanded=False):
        st.markdown(
            """
        ### How Face Matching Works
        
        Face matching compares detected faces against a database of known faces to identify individuals. This process involves:
        
        1. **Face Detection**: Locating faces in the image (as in the Face Detection feature)
        2. **Feature Extraction**: Creating a 128-dimensional "face encoding" that numerically represents the face
        3. **Comparison**: Calculating the distance between the detected face encoding and all known reference faces
        4. **Identification**: Matching the face to the closest known reference if it's within the threshold
        
        ### Settings Explained
        
        - **Matching Threshold**: Controls how strict the matching is (depending on the compairison mode)
        - **Comparison Mode**: "Distance" measures difference (lower is better match), "Similarity" measures likeness (higher is a better match)
        - **Best Match Only**: Returns only the single best matching face rather than all matches
        
        ### Tips for Best Results
        
        - Use clear, well-lit reference photos with face centered
        - Multiple reference images per person improves accuracy
        - Higher-quality images provide better matching performance
        - Position faces similarly to reference images when matching
        """
        )


if __name__ == "__main__":
    face_matching_page()
