"""
Webcam Component for Streamlit Interface

This module provides a reusable webcam component that can be used across different pages
of the Streamlit interface for capturing and processing video frames.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from PIL import Image
from typing import Tuple, Dict, Callable, Any, List, Optional

def webcam_component(
    callback_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    use_column: bool = True,
    fps_limit: int = 24,
    show_fps: bool = True,
    key_prefix: str = "",
) -> None:
    """
    Reusable webcam component that handles webcam capture and processing.
    
    Parameters
    ----------
    callback_func : Callable
        Function to process each frame, should return (processed_frame, metadata_dict)
    use_column : bool, optional
        Whether to use columns for layout (default: True)
    fps_limit : int, optional
        Maximum FPS to process (default: 24)
    show_fps : bool, optional
        Whether to show FPS counter (default: True)
    key_prefix : str, optional
        Prefix for session state keys to avoid conflicts (default: "")
    """
    # Set up session state for webcam
    if f"{key_prefix}webcam_running" not in st.session_state:
        st.session_state[f"{key_prefix}webcam_running"] = False
        st.session_state[f"{key_prefix}frame_count"] = 0
        st.session_state[f"{key_prefix}start_time"] = time.time()
        st.session_state[f"{key_prefix}fps"] = 0.0
        st.session_state[f"{key_prefix}last_frame_time"] = 0.0
        
    # Controls for webcam
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if not st.session_state[f"{key_prefix}webcam_running"]:
            if st.button("Start Webcam", key=f"{key_prefix}start_webcam"):
                st.session_state[f"{key_prefix}webcam_running"] = True
                st.session_state[f"{key_prefix}frame_count"] = 0
                st.session_state[f"{key_prefix}start_time"] = time.time()
                st.experimental_rerun()
        else:
            if st.button("Stop Webcam", key=f"{key_prefix}stop_webcam"):
                st.session_state[f"{key_prefix}webcam_running"] = False
                st.experimental_rerun()
    
    # Create placeholders for webcam feed and info
    if use_column:
        col_feed, col_info = st.columns([3, 1])
        frame_placeholder = col_feed.empty()
        info_placeholder = col_info.empty()
    else:
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
    
    # If webcam is running, capture and process frames
    if st.session_state[f"{key_prefix}webcam_running"]:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            st.session_state[f"{key_prefix}webcam_running"] = False
            return
        
        try:
            # Display status
            info_placeholder.info("Webcam is active. Press 'Stop Webcam' to end the session.")
            
            # Webcam loop with FPS limiting
            while st.session_state[f"{key_prefix}webcam_running"]:
                # Limit frame rate
                current_time = time.time()
                time_since_last_frame = current_time - st.session_state[f"{key_prefix}last_frame_time"]
                min_time_between_frames = 1.0 / fps_limit
                
                if time_since_last_frame < min_time_between_frames:
                    time.sleep(min_time_between_frames - time_since_last_frame)
                    continue
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    info_placeholder.error("Failed to capture frame from webcam.")
                    break
                
                # Update timing info
                st.session_state[f"{key_prefix}last_frame_time"] = time.time()
                st.session_state[f"{key_prefix}frame_count"] += 1
                
                # Calculate FPS
                elapsed_time = time.time() - st.session_state[f"{key_prefix}start_time"]
                if elapsed_time > 0:
                    st.session_state[f"{key_prefix}fps"] = st.session_state[f"{key_prefix}frame_count"] / elapsed_time
                
                # Process frame with callback function
                try:
                    processed_frame, metadata = callback_func(frame)
                    
                    # Convert to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Add FPS counter if enabled
                    if show_fps:
                        fps_text = f"FPS: {st.session_state[f'{key_prefix}fps']:.1f}"
                        cv2.putText(
                            rgb_frame,
                            fps_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )
                    
                    # Update placeholders
                    frame_placeholder.image(rgb_frame, channels='RGB', use_column_width=True)
                    
                    # Display metadata
                    if metadata:
                        info_text = ""
                        for key, value in metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                info_text += f"**{key}**: {value}\n\n"
                        
                        if info_text:
                            info_placeholder.markdown(info_text)
                    
                except Exception as e:
                    info_placeholder.error(f"Error processing frame: {str(e)}")
                    break
                
                # Check if webcam should still be running (may have been changed by button)
                if not st.session_state[f"{key_prefix}webcam_running"]:
                    break
                    
        finally:
            # Clean up
            cap.release()
            info_placeholder.info("Webcam session ended.")
    else:
        # Display instructions when webcam is not running
        frame_placeholder.info("Click 'Start Webcam' to begin.")
        info_placeholder.empty()

def image_upload_component(
    callback_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    allowed_types: List[str] = ["jpg", "jpeg", "png"],
    key_prefix: str = "",
) -> None:
    """
    Reusable image upload component that handles image upload and processing.
    
    Parameters
    ----------
    callback_func : Callable
        Function to process the image, should return (processed_image, metadata_dict)
    allowed_types : List[str], optional
        List of allowed file extensions (default: ["jpg", "jpeg", "png"])
    key_prefix : str, optional
        Prefix for session state keys to avoid conflicts (default: "")
    """
    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=allowed_types,
        key=f"{key_prefix}image_uploader"
    )
    
    if uploaded_file is not None:
        # Create columns for display
        col_img, col_info = st.columns([3, 1])
        
        try:
            # Read and process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR (OpenCV format) if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Process image with callback function
            processed_image, metadata = callback_func(image_np)
            
            # Convert back to RGB for display
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                display_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                display_img = processed_image
                
            # Display processed image
            col_img.image(
                display_img, 
                caption=f"Processed image: {uploaded_file.name}",
                use_column_width=True
            )
            
            # Display metadata
            if metadata:
                info_text = ""
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        info_text += f"**{key}**: {value}\n\n"
                    elif isinstance(value, list) and key == "face_locations" and len(value) > 0:
                        info_text += f"**{key}**: {len(value)} faces found\n\n"
                        
                        # Show first few faces' locations
                        max_faces_to_show = min(5, len(value))
                        for i in range(max_faces_to_show):
                            face = value[i]
                            if isinstance(face, tuple) and len(face) == 4:
                                top, right, bottom, left = face
                                info_text += f"Face {i+1}: (T={top}, R={right}, B={bottom}, L={left})\n\n"
                
                if info_text:
                    col_info.markdown(info_text)
            
            # Add download button for processed image
            if st.button("Save Processed Image", key=f"{key_prefix}save_button"):
                # Save image to temp file
                filename = f"processed_{uploaded_file.name}"
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{filename.split('.')[-1]}") as tmp:
                    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                        save_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    else:
                        save_img = processed_image
                    
                    pil_img = Image.fromarray(save_img)
                    pil_img.save(tmp.name)
                    
                # Offer download
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="Download Processed Image",
                        data=f.read(),
                        file_name=filename,
                        mime=f"image/{filename.split('.')[-1]}"
                    )
                    
                # Clean up temp file
                os.unlink(tmp.name)
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
