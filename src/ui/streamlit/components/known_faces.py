"""
Known Faces Management Component for Streamlit Interface

This module provides functionality for displaying, adding, and removing known faces
in the Streamlit interface.
"""

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import face_recognition
import tempfile
import shutil
import uuid
from typing import List, Dict, Any, Tuple

def format_person_name(filename: str) -> str:
    """
    Format person name from filename.
    
    Parameters
    ----------
    filename : str
        Filename of the known face image
        
    Returns
    -------
    str
        Formatted person name
    """
    # Remove extension
    name = os.path.splitext(filename)[0]
    
    # Replace underscores with spaces
    name = name.replace("_", " ")
    
    # Remove any numeric suffixes (e.g., "John_Smith_1" -> "John Smith")
    parts = name.split()
    if parts and parts[-1].isdigit():
        name = " ".join(parts[:-1])
    
    return name

def known_faces_grid(known_faces_dir: str, page_size: int = 12, key_prefix: str = "") -> List[str]:
    """
    Display a grid of known faces with pagination.
    
    Parameters
    ----------
    known_faces_dir : str
        Directory containing known face images
    page_size : int, optional
        Number of faces to display per page (default: 12)
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    List[str]
        List of selected face filenames
    """
    # Ensure directory exists
    if not os.path.exists(known_faces_dir):
        st.warning(f"Known faces directory not found. Creating: {known_faces_dir}")
        os.makedirs(known_faces_dir, exist_ok=True)
        return []
    
    # Get all image files
    image_files = [f for f in os.listdir(known_faces_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        st.info("No known faces found. Add reference faces using the form below.")
        return []
    
    # Initialize session state for pagination
    if f"{key_prefix}faces_page" not in st.session_state:
        st.session_state[f"{key_prefix}faces_page"] = 0
    
    # Initialize session state for selected faces
    if f"{key_prefix}selected_faces" not in st.session_state:
        st.session_state[f"{key_prefix}selected_faces"] = []
    
    # Calculate total pages
    total_pages = (len(image_files) + page_size - 1) // page_size
    
    # Page navigation
    col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
    
    with col1:
        if st.button("←", key=f"{key_prefix}prev_page", disabled=st.session_state[f"{key_prefix}faces_page"] <= 0):
            st.session_state[f"{key_prefix}faces_page"] -= 1
            st.experimental_rerun()
    
    with col2:
        st.write(f"Page {st.session_state[f'{key_prefix}faces_page'] + 1} of {max(1, total_pages)}")
    
    with col3:
        st.write(f"{len(image_files)} total known faces")
        
    with col4:
        if st.button("→", key=f"{key_prefix}next_page", disabled=st.session_state[f"{key_prefix}faces_page"] >= total_pages - 1):
            st.session_state[f"{key_prefix}faces_page"] += 1
            st.experimental_rerun()
    
    # Get faces for current page
    start_idx = st.session_state[f"{key_prefix}faces_page"] * page_size
    end_idx = min(start_idx + page_size, len(image_files))
    current_page_files = image_files[start_idx:end_idx]
    
    # Display grid
    cols_per_row = 4
    for i in range(0, len(current_page_files), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            
            if idx < len(current_page_files):
                file = current_page_files[idx]
                person_name = format_person_name(file)
                
                with cols[j]:
                    # Load and display image
                    img_path = os.path.join(known_faces_dir, file)
                    try:
                        image = Image.open(img_path)
                        st.image(image, caption=person_name, use_column_width=True)
                        
                        # Checkbox for selection
                        is_selected = st.checkbox(
                            "Select", 
                            value=file in st.session_state[f"{key_prefix}selected_faces"],
                            key=f"{key_prefix}select_{idx}_{file}"
                        )
                        
                        if is_selected and file not in st.session_state[f"{key_prefix}selected_faces"]:
                            st.session_state[f"{key_prefix}selected_faces"].append(file)
                        elif not is_selected and file in st.session_state[f"{key_prefix}selected_faces"]:
                            st.session_state[f"{key_prefix}selected_faces"].remove(file)
                        
                    except Exception as e:
                        st.error(f"Error loading {file}: {str(e)}")
    
    # Return list of selected faces
    return st.session_state[f"{key_prefix}selected_faces"]

def add_known_face(known_faces_dir: str, key_prefix: str = "") -> bool:
    """
    Add a new known face via file upload or webcam capture.
    
    Parameters
    ----------
    known_faces_dir : str
        Directory to save known face images
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    bool
        True if a face was successfully added, False otherwise
    """
    st.subheader("Add New Known Face")
    
    # Ensure directory exists
    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir, exist_ok=True)
    
    # Method selection
    add_method = st.radio(
        "Add Method",
        options=["Upload Image", "Capture from Webcam"],
        horizontal=True,
        key=f"{key_prefix}add_method_radio"
    )
    
    # Person name input
    person_name = st.text_input(
        "Person Name", 
        help="Enter the name of the person (e.g., 'John Smith')",
        key=f"{key_prefix}person_name_input"
    )
    
    # Initialize result flag
    success = False
    
    # Upload image
    if add_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}face_uploader"
        )
        
        if uploaded_file is not None:
            try:
                # Read image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Convert to BGR for OpenCV processing if needed
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    bgr_image = image_np
                
                # Display preview
                st.image(image, caption="Preview", use_column_width=True)
                
                # Detect faces
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    st.warning("No face detected in the uploaded image. Please try another image.")
                elif len(face_locations) > 1:
                    st.warning("Multiple faces detected. Please upload an image with only one face.")
                else:
                    # Process face
                    if st.button("Add as Known Face", key=f"{key_prefix}confirm_upload_button"):
                        if not person_name:
                            st.error("Please enter a person name.")
                        else:
                            # Format filename
                            safe_name = person_name.replace(" ", "_").lower()
                            
                            # Check if name already exists
                            existing_files = [f for f in os.listdir(known_faces_dir) 
                                             if f.lower().startswith(safe_name.lower())]
                            
                            if existing_files:
                                # Add numeric suffix
                                filename = f"{safe_name}_{len(existing_files)}.jpg"
                            else:
                                filename = f"{safe_name}.jpg"
                            
                            # Save image
                            output_path = os.path.join(known_faces_dir, filename)
                            
                            # Extract and save face region only
                            top, right, bottom, left = face_locations[0]
                            face_image = bgr_image[top:bottom, left:right]
                            cv2.imwrite(output_path, face_image)
                            
                            st.success(f"Added {person_name} to known faces!")
                            success = True
                            
                            # Clear file uploader
                            st.session_state[f"{key_prefix}face_uploader"] = None
                            
                            # Force page refresh to show new face
                            st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    # Capture from webcam
    elif add_method == "Capture from Webcam":
        st.info("Position the face in the center of the webcam and click 'Capture'.")
        
        # Initialize webcam state if not already done
        if f"{key_prefix}webcam_frame" not in st.session_state:
            st.session_state[f"{key_prefix}webcam_frame"] = None
        
        if f"{key_prefix}webcam_active" not in st.session_state:
            st.session_state[f"{key_prefix}webcam_active"] = False
        
        # Webcam controls
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state[f"{key_prefix}webcam_active"]:
                if st.button("Start Webcam", key=f"{key_prefix}start_webcam_button"):
                    st.session_state[f"{key_prefix}webcam_active"] = True
                    st.experimental_rerun()
            else:
                if st.button("Stop Webcam", key=f"{key_prefix}stop_webcam_button"):
                    st.session_state[f"{key_prefix}webcam_active"] = False
                    st.session_state[f"{key_prefix}webcam_frame"] = None
                    st.experimental_rerun()
        
        with col2:
            capture_button = st.button(
                "Capture Face", 
                key=f"{key_prefix}capture_button",
                disabled=not st.session_state[f"{key_prefix}webcam_active"]
            )
        
        # Webcam display placeholder
        webcam_placeholder = st.empty()
        
        if st.session_state[f"{key_prefix}webcam_active"]:
            # This is a simplified webcam implementation. In a real implementation,
            # you would use more sophisticated methods for continuous webcam feed.
            stframe = webcam_placeholder.camera_input("Webcam Feed", key=f"{key_prefix}webcam_feed")
            
            if stframe is not None:
                # Convert image to numpy array
                image = Image.open(stframe)
                image_np = np.array(image)
                
                # Store the frame
                st.session_state[f"{key_prefix}webcam_frame"] = image_np
        
        # Process captured frame
        if capture_button and st.session_state[f"{key_prefix}webcam_frame"] is not None:
            try:
                # Get captured frame
                image_np = st.session_state[f"{key_prefix}webcam_frame"]
                
                # Convert to BGR for OpenCV processing
                bgr_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                # Detect faces
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                
                if not face_locations:
                    st.warning("No face detected in the captured image. Please try again.")
                elif len(face_locations) > 1:
                    st.warning("Multiple faces detected. Please ensure only one face is in the frame.")
                else:
                    # Display the captured face
                    top, right, bottom, left = face_locations[0]
                    face_image = image_np[top:bottom, left:right]
                    st.image(face_image, caption="Captured Face", use_column_width=True)
                    
                    # Save button
                    if st.button("Save as Known Face", key=f"{key_prefix}save_capture_button"):
                        if not person_name:
                            st.error("Please enter a person name.")
                        else:
                            # Format filename
                            safe_name = person_name.replace(" ", "_").lower()
                            
                            # Check if name already exists
                            existing_files = [f for f in os.listdir(known_faces_dir) 
                                             if f.lower().startswith(safe_name.lower())]
                            
                            if existing_files:
                                # Add numeric suffix
                                filename = f"{safe_name}_{len(existing_files)}.jpg"
                            else:
                                filename = f"{safe_name}.jpg"
                            
                            # Save image
                            output_path = os.path.join(known_faces_dir, filename)
                            
                            # Save the face region only
                            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_path, face_image_bgr)
                            
                            st.success(f"Added {person_name} to known faces!")
                            success = True
                            
                            # Reset webcam frame
                            st.session_state[f"{key_prefix}webcam_frame"] = None
                            
                            # Force page refresh to show new face
                            st.experimental_rerun()
            
            except Exception as e:
                st.error(f"Error processing webcam image: {str(e)}")
    
    return success

def remove_known_faces(known_faces_dir: str, selected_faces: List[str], key_prefix: str = "") -> int:
    """
    Remove selected known faces.
    
    Parameters
    ----------
    known_faces_dir : str
        Directory containing known face images
    selected_faces : List[str]
        List of selected face filenames to remove
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    int
        Number of faces successfully removed
    """
    if not selected_faces:
        st.warning("No faces selected for removal.")
        return 0
    
    # Show confirmation
    st.subheader("Remove Selected Faces")
    
    st.warning(f"You are about to remove {len(selected_faces)} known faces.")
    
    # List faces to be removed
    face_list = ", ".join([format_person_name(f) for f in selected_faces])
    st.write(f"Selected faces: {face_list}")
    
    # Confirmation checkbox
    confirm = st.checkbox(
        "I understand this action cannot be undone",
        key=f"{key_prefix}confirm_removal_checkbox"
    )
    
    if not confirm:
        return 0
    
    # Remove button
    if st.button("Remove Selected Faces", key=f"{key_prefix}remove_button"):
        removed_count = 0
        
        try:
            # Create backup directory
            backup_dir = os.path.join(os.path.dirname(known_faces_dir), "known_faces_backup")
            os.makedirs(backup_dir, exist_ok=True)
            
            for file in selected_faces:
                file_path = os.path.join(known_faces_dir, file)
                
                if os.path.exists(file_path):
                    # Backup the file
                    backup_path = os.path.join(backup_dir, f"{uuid.uuid4()}_{file}")
                    shutil.copy2(file_path, backup_path)
                    
                    # Remove the file
                    os.remove(file_path)
                    removed_count += 1
            
            # Clear selected faces list
            st.session_state[f"{key_prefix}selected_faces"] = []
            
            st.success(f"Successfully removed {removed_count} faces. A backup was created in case you need to restore them.")
            
            # Force page refresh
            st.experimental_rerun()
        
        except Exception as e:
            st.error(f"Error removing faces: {str(e)}")
        
        return removed_count
    
    return 0
