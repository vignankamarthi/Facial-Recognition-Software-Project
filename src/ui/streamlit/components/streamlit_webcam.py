"""
Streamlit Native Webcam Component

This module provides a webcam component that uses Streamlit's built-in camera_input
rather than launching external OpenCV windows. This is more compatible with
containerized environments like Docker.
"""

import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
from typing import Tuple, Dict, Any, Callable

def streamlit_webcam_component(
    callback_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    key_prefix: str = ""
) -> None:
    """
    Webcam component that uses Streamlit's native camera_input instead of external OpenCV windows.
    
    Parameters
    ----------
    callback_func : Callable
        Function to process each frame, should return (processed_frame, metadata_dict)
    key_prefix : str, optional
        Prefix for session state keys to avoid conflicts (default: "")
    """
    st.write("### Use your webcam")
    st.write("Allow camera access when prompted by your browser.")
    
    # Add instructions
    st.info("""
    **Instructions:**
    1. Click "Take Photo" to capture an image from your webcam
    2. The image will be processed automatically
    3. Click "Take Photo" again to capture a new image
    """)
    
    # Use Streamlit's camera_input
    img_file_buffer = st.camera_input("Take a photo", key=f"{key_prefix}camera")
    
    # Process the captured image
    if img_file_buffer is not None:
        # Create a placeholder for the processed image
        result_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Create a status message
        with st.spinner("Processing image..."):
            try:
                # Read the captured image
                bytes_data = img_file_buffer.getvalue()
                img = Image.open(img_file_buffer)
                img_array = np.array(img)
                
                # Check if we need to convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    cv_img = img_array
                
                # Process the image with the callback function
                start_time = time.time()
                processed_img, metadata = callback_func(cv_img)
                processing_time = time.time() - start_time
                
                # Convert back to RGB for display
                if len(processed_img.shape) == 3 and processed_img.shape[2] == 3:
                    display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                else:
                    display_img = processed_img
                
                # Add processing time to metadata
                if metadata is None:
                    metadata = {}
                metadata['processing_time'] = f"{processing_time:.3f} seconds"
                
                # Display the processed image
                result_placeholder.image(
                    display_img, 
                    caption="Processed image", 
                    use_column_width=True
                )
                
                # Display metadata
                if metadata:
                    # Format metadata as markdown
                    metadata_text = "\n".join([f"**{k}**: {v}" for k, v in metadata.items()])
                    info_placeholder.markdown(f"### Results\n{metadata_text}")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:
        # No image captured yet, display instructions
        st.write("Waiting for webcam capture...")

    # Add some space
    st.write("")
    
    # Optional: Add a button to clear the captured image
    if img_file_buffer is not None:
        if st.button("Clear", key=f"{key_prefix}clear"):
            # Clear the webcam input
            st.session_state[f"{key_prefix}camera"] = None
            st.rerun()
