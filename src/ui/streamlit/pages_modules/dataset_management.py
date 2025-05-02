"""
Dataset Management page for the Streamlit interface.

This module provides the Streamlit UI for managing datasets, including downloading
and preparing the UTKFace dataset for bias testing and other features.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import shutil
import subprocess
import threading
import tempfile
import glob

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

# Import core functionality
from src.utils.image_processing import ImageProcessor
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import UI components
from src.ui.streamlit.components import (
    dataset_management_config_panel,
    dataset_statistics_visualization,
    dataset_browser
)

# Setup logging
logger = get_logger(__name__)

# Get configuration
config = get_config()

# Initialize session state variables if they don't exist
def init_session_state():
    if "background_task" not in st.session_state:
        st.session_state["background_task"] = {
            "running": False,
            "progress": 0,
            "message": "",
            "complete": False,
            "success": False,
            "error": None
        }

# Always call this at the beginning of the script
init_session_state()

def run_background_task(task_func, *args, **kwargs):
    """
    Run a task in the background with progress tracking.
    
    Parameters
    ----------
    task_func : callable
        Function to run in the background
    *args, **kwargs
        Arguments to pass to the task function
    """
    # Make sure session state is initialized
    init_session_state()
    
    # Reset task state
    st.session_state["background_task"] = {
        "running": True,
        "progress": 0,
        "message": "Starting task...",
        "complete": False,
        "success": False,
        "error": None
    }
    
    # Define wrapper function for the thread
    def task_wrapper():
        try:
            # Make sure we can access session state
            if "background_task" not in st.session_state:
                # If we can't access session state, rebuild it
                init_session_state()
            
            # Run the task
            result = task_func(*args, **kwargs)
            
            # Update state on completion
            if "background_task" in st.session_state:
                st.session_state["background_task"]["complete"] = True
                st.session_state["background_task"]["running"] = False
                st.session_state["background_task"]["success"] = result
                st.session_state["background_task"]["progress"] = 100
                st.session_state["background_task"]["message"] = "Task completed successfully."
        
        except Exception as e:
            # Update state on error
            if "background_task" in st.session_state:
                st.session_state["background_task"]["complete"] = True
                st.session_state["background_task"]["running"] = False
                st.session_state["background_task"]["success"] = False
                st.session_state["background_task"]["error"] = str(e)
                st.session_state["background_task"]["message"] = f"Error: {str(e)}"
            logger.error(f"Background task error: {e}")
    
    # Start thread
    thread = threading.Thread(target=task_wrapper)
    thread.daemon = True
    thread.start()

def download_utkface_dataset(sample_size, ethnicity_selection, selected_ethnicities=None):
    """
    Task to download and extract the UTKFace dataset.
    
    Parameters
    ----------
    sample_size : int
        Number of images to include in the sample
    ethnicity_selection : str
        Type of ethnicity selection
    selected_ethnicities : List[str], optional
        Specific ethnicities to include if custom selection
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get the ImageProcessor instance
        processor = ImageProcessor()
        
        # Map ethnicity selection to specific ethnicities
        if ethnicity_selection == "All ethnicities":
            specific_ethnicities = None
        elif ethnicity_selection == "White and Black only":
            specific_ethnicities = [0, 1]  # White and Black
        elif ethnicity_selection == "White, Black, and Asian":
            specific_ethnicities = [0, 1, 2]  # White, Black, and Asian
        elif ethnicity_selection == "Custom selection" and selected_ethnicities:
            # Map from display names to codes
            ethnicity_map = {
                "White": 0,
                "Black": 1,
                "Asian": 2,
                "Indian": 3,
                "Others": 4
            }
            specific_ethnicities = [ethnicity_map[e] for e in selected_ethnicities if e in ethnicity_map]
        else:
            specific_ethnicities = None
        
        # Initialize session state if needed
        init_session_state()
        
        # Update progress safely with dict-style access
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 10
            st.session_state["background_task"]["message"] = "Starting UTKFace dataset download..."
        
        # Download and extract the dataset
        result = processor.download_and_extract_utkface_dataset(
            target_dir=st.session_state.paths["datasets_dir"],
            sample_size=sample_size,
            specific_ethnicities=specific_ethnicities
        )
        
        # Update progress
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 100
            st.session_state["background_task"]["message"] = "UTKFace dataset downloaded and extracted successfully."
        
        return result
    
    except Exception as e:
        logger.error(f"Error downloading UTKFace dataset: {e}")
        raise e

def setup_bias_testing(images_per_ethnicity):
    """
    Task to set up bias testing with UTKFace dataset.
    
    Parameters
    ----------
    images_per_ethnicity : int
        Number of images per ethnicity to include
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get the ImageProcessor instance
        processor = ImageProcessor()
        
        # Check if UTKFace dataset is available
        utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "demographic_split")
        
        if not os.path.exists(utkface_dir):
            raise ValueError("UTKFace dataset not found. Please download it first.")
        
        # Initialize session state if needed
        init_session_state()
        
        # Update progress safely
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 10
            st.session_state["background_task"]["message"] = "Setting up bias testing dataset..."
        
        # Prepare the dataset for bias testing
        result = processor.prepare_utkface_for_bias_testing(
            utkface_dir=utkface_dir,
            test_datasets_dir=st.session_state.paths["test_datasets_dir"],
            images_per_ethnicity=images_per_ethnicity
        )
        
        # Update progress
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 100
            st.session_state["background_task"]["message"] = "Bias testing dataset prepared successfully."
        
        return result
    
    except Exception as e:
        logger.error(f"Error setting up bias testing: {e}")
        raise e

def prepare_known_faces(num_people, ethnicity_balanced):
    """
    Task to prepare known faces from UTKFace dataset.
    
    Parameters
    ----------
    num_people : int
        Number of reference people to include
    ethnicity_balanced : bool
        Whether to balance ethnicity representation
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get the ImageProcessor instance
        processor = ImageProcessor()
        
        # Check if UTKFace dataset is available
        utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "utkface_aligned")
        
        if not os.path.exists(utkface_dir):
            raise ValueError("UTKFace aligned dataset not found. Please download it first.")
        
        # Initialize session state if needed
        init_session_state()
        
        # Update progress safely
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 10
            st.session_state["background_task"]["message"] = "Preparing known faces..."
        
        # Prepare known faces
        result = processor.prepare_known_faces_from_utkface(
            num_people=num_people,
            ethnicity_balanced=ethnicity_balanced,
            utkface_dir=utkface_dir,
            output_dir=st.session_state.paths["known_faces_dir"]
        )
        
        # Update progress
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 100
            st.session_state["background_task"]["message"] = "Known faces prepared successfully."
        
        return result
    
    except Exception as e:
        logger.error(f"Error preparing known faces: {e}")
        raise e

def prepare_test_dataset(num_known, num_unknown):
    """
    Task to prepare test dataset from UTKFace dataset.
    
    Parameters
    ----------
    num_known : int
        Number of known people to include
    num_unknown : int
        Number of unknown people to include
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get the ImageProcessor instance
        processor = ImageProcessor()
        
        # Check if UTKFace dataset is available
        utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "utkface_aligned")
        
        if not os.path.exists(utkface_dir):
            raise ValueError("UTKFace aligned dataset not found. Please download it first.")
        
        # Initialize session state if needed
        init_session_state()
        
        # Update progress safely
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 10
            st.session_state["background_task"]["message"] = "Preparing test dataset..."
        
        # Prepare test dataset
        result = processor.prepare_test_dataset_from_utkface(
            num_known=num_known,
            num_unknown=num_unknown,
            utkface_dir=utkface_dir,
            known_faces_dir=st.session_state.paths["known_faces_dir"],
            output_dir=os.path.join(st.session_state.paths["test_datasets_dir"], "test_images")
        )
        
        # Update progress
        if "background_task" in st.session_state:
            st.session_state["background_task"]["progress"] = 100
            st.session_state["background_task"]["message"] = "Test dataset prepared successfully."
        
        return result
    
    except Exception as e:
        logger.error(f"Error preparing test dataset: {e}")
        raise e

def dataset_management_page():
    """Render the dataset management page."""
    
    st.markdown("# Dataset Management")
    st.markdown("Download and prepare datasets for facial recognition features.")
    
    # Dataset management settings
    with st.expander("Dataset Management Settings", expanded=True):
        updated_config = dataset_management_config_panel(
            st.session_state.config["dataset_management"],
            key_prefix="dm_"
        )
        st.session_state.config["dataset_management"] = updated_config
    
    # Display different forms based on selected action
    action = st.session_state.config["dataset_management"]["action"]
    
    # Create columns for action and execution
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"## {action}")
        
        if action == "Download UTKFace Dataset":
            st.markdown("""
            This action downloads the UTKFace dataset with demographic annotations (age, gender, and ethnicity).
            The dataset will be used for bias testing and other face recognition features.
            """)
            
            # Add detailed download instructions
            with st.expander("📥 UTKFace Dataset Download Instructions", expanded=True):
                st.markdown("""
                ### Download Instructions
                
                1. **Visit the official UTKFace dataset page**: [UTKFace Dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
                
                2. **Download the following ZIP files**:
                   - `UTKFace.tar.gz` (main dataset, ~225MB)
                   - `utkface_aligned.tar.gz` (aligned faces, ~300MB)
                
                3. **Place the downloaded files in the correct location**:
                   - Move the downloaded files to the `data/datasets` directory in your project root
                   - The exact required path is: `/Users/vkamarthi24/Desktop/Personal Projects/Facial-Recognition-Software-Project/data/datasets/`
                   - The system will automatically find and extract files from this location
                   - Do not rename the files - keep the original filenames
                   
                4. **After downloading**:
                   - Click the "Execute Action" button on the right
                   - The system will process the dataset according to your settings
                   - You'll see progress updates here in the UI
                
                ### Important Notes
                
                - The complete dataset contains **over 20,000 images** (24GB uncompressed)
                - The download process can be lengthy depending on your internet connection
                - You need a Kaggle account to download the dataset
                - Select a reasonable `Sample Size` in the settings above based on your system's capabilities
                - For bias testing, we need the demographic information included in filenames
                """)
            
            # Check if the dataset already exists
            utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface")
            if os.path.exists(utkface_dir):
                st.info(f"UTKFace dataset directory already exists at: {utkface_dir}")
                
                # Check if it contains files
                aligned_dir = os.path.join(utkface_dir, "utkface_aligned")
                demo_dir = os.path.join(utkface_dir, "demographic_split")
                
                if os.path.exists(aligned_dir) and os.path.exists(demo_dir):
                    # Count files in directories
                    aligned_files = glob.glob(os.path.join(aligned_dir, "**", "*.jpg"), recursive=True)
                    demo_files = glob.glob(os.path.join(demo_dir, "**", "*.jpg"), recursive=True)
                    
                    if aligned_files or demo_files:
                        st.success(f"UTKFace dataset appears to be already downloaded with {len(aligned_files)} aligned images and {len(demo_files)} demographic split images.")
        
        elif action == "Set up Bias Testing with UTKFace":
            st.markdown("""
            This action prepares the UTKFace dataset for bias testing by organizing images into demographic groups.
            The prepared dataset will be used to analyze recognition accuracy across different ethnicities.
            """)
            
            # Check if the source dataset exists
            utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "demographic_split")
            if not os.path.exists(utkface_dir):
                st.warning(f"UTKFace demographic split dataset not found at: {utkface_dir}")
                st.info("Please download the UTKFace dataset first.")
            
            # Check if the target directory already has content
            target_dir = st.session_state.paths["demographic_split_dir"]
            if os.path.exists(target_dir):
                # Count files in ethnic groups
                ethnicity_counts = {}
                for group in ["white", "black", "asian", "indian", "others"]:
                    group_dir = os.path.join(target_dir, group)
                    if os.path.exists(group_dir):
                        image_files = [f for f in os.listdir(group_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        ethnicity_counts[group] = len(image_files)
                
                if any(ethnicity_counts.values()):
                    st.success(f"Bias testing dataset already contains images:")
                    for group, count in ethnicity_counts.items():
                        st.write(f"- {group.capitalize()}: {count} images")
        
        elif action == "Prepare Known Faces from UTKFace":
            st.markdown("""
            This action extracts reference faces from the UTKFace dataset for face matching.
            These known faces will be used to identify people in the face matching feature.
            """)
            
            # Check if the source dataset exists
            utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "utkface_aligned")
            if not os.path.exists(utkface_dir):
                st.warning(f"UTKFace aligned dataset not found at: {utkface_dir}")
                st.info("Please download the UTKFace dataset first.")
            
            # Check if the target directory already has content
            target_dir = st.session_state.paths["known_faces_dir"]
            if os.path.exists(target_dir):
                image_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    st.success(f"Known faces directory already contains {len(image_files)} reference images.")
        
        elif action == "Prepare Test Dataset from UTKFace":
            st.markdown("""
            This action creates a test dataset from the UTKFace dataset for evaluating face recognition.
            The test dataset will include both known and unknown faces for evaluation.
            """)
            
            # Check if the source dataset exists
            utkface_dir = os.path.join(st.session_state.paths["datasets_dir"], "utkface", "utkface_aligned")
            if not os.path.exists(utkface_dir):
                st.warning(f"UTKFace aligned dataset not found at: {utkface_dir}")
                st.info("Please download the UTKFace dataset first.")
            
            # Check if the known faces directory has content
            known_dir = st.session_state.paths["known_faces_dir"]
            if not os.path.exists(known_dir) or not os.listdir(known_dir):
                st.warning(f"Known faces directory is empty: {known_dir}")
                st.info("Please prepare known faces first.")
            
            # Check if the target directory already has content
            test_dir = os.path.join(st.session_state.paths["test_datasets_dir"], "test_images")
            if os.path.exists(test_dir):
                known_test_dir = os.path.join(test_dir, "known")
                unknown_test_dir = os.path.join(test_dir, "unknown")
                
                known_count = len([f for f in os.listdir(known_test_dir)]) if os.path.exists(known_test_dir) else 0
                unknown_count = len([f for f in os.listdir(unknown_test_dir)]) if os.path.exists(unknown_test_dir) else 0
                
                if known_count or unknown_count:
                    st.success(f"Test dataset already contains {known_count} known face images and {unknown_count} unknown face images.")
    
    with col2:
        # Execute button
        if st.button("Execute Action", type="primary", key="dm_execute_button"):
            # Check directories
            for path_name, path in st.session_state.paths.items():
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created directory: {path}")
            
            # Execute selected action in the background
            if action == "Download UTKFace Dataset":
                run_background_task(
                    download_utkface_dataset,
                    st.session_state.config["dataset_management"]["sample_size"],
                    st.session_state.config["dataset_management"]["ethnicity_selection"],
                    st.session_state.config["dataset_management"]["selected_ethnicities"] if st.session_state.config["dataset_management"]["ethnicity_selection"] == "Custom selection" else None
                )
            
            elif action == "Set up Bias Testing with UTKFace":
                run_background_task(
                    setup_bias_testing,
                    st.session_state.config["dataset_management"]["images_per_ethnicity"]
                )
            
            elif action == "Prepare Known Faces from UTKFace":
                run_background_task(
                    prepare_known_faces,
                    st.session_state.config["dataset_management"]["num_people"],
                    st.session_state.config["dataset_management"]["ethnicity_balanced"]
                )
            
            elif action == "Prepare Test Dataset from UTKFace":
                run_background_task(
                    prepare_test_dataset,
                    st.session_state.config["dataset_management"]["num_known"],
                    st.session_state.config["dataset_management"]["num_unknown"]
                )
    
    # Display task progress
    if st.session_state.background_task["running"] or st.session_state.background_task["complete"]:
        st.markdown("## Task Progress")
        
        # Progress bar
        st.progress(st.session_state.background_task["progress"] / 100)
        
        # Status message
        st.info(st.session_state.background_task["message"])
        
        # Error message if applicable
        if st.session_state.background_task["error"]:
            st.error(f"Error: {st.session_state.background_task['error']}")
        
        # Success message if applicable
        if st.session_state.background_task["complete"] and st.session_state.background_task["success"]:
            st.success("Task completed successfully!")
            
            # Add button to clear task status
            if st.button("Clear Task Status", key="dm_clear_task_button"):
                st.session_state.background_task = {
                    "running": False,
                    "progress": 0,
                    "message": "",
                    "complete": False,
                    "success": False,
                    "error": None
                }
    
    # Create tabs for browsing different datasets
    st.markdown("## Dataset Browser")
    
    tab1, tab2, tab3 = st.tabs(["Demographic Split", "Known Faces", "Test Images"])
    
    with tab1:
        # Browse demographic split dataset
        demographic_dir = st.session_state.paths["demographic_split_dir"]
        
        if not os.path.exists(demographic_dir):
            st.warning(f"Demographic split dataset not found at: {demographic_dir}")
            st.info("Please set up the dataset first.")
        else:
            # Dataset statistics
            dataset_statistics_visualization(
                demographic_dir,
                title="Demographic Split Dataset Statistics",
                key_prefix="dm_demo_stats_"
            )
            
            # Dataset browser
            dataset_browser(demographic_dir, key_prefix="dm_demo_browser_")
    
    with tab2:
        # Browse known faces
        known_faces_dir = st.session_state.paths["known_faces_dir"]
        
        if not os.path.exists(known_faces_dir):
            st.warning(f"Known faces directory not found at: {known_faces_dir}")
            st.info("Please set up known faces first.")
        else:
            # Dataset statistics
            dataset_statistics_visualization(
                known_faces_dir,
                title="Known Faces Statistics",
                key_prefix="dm_known_stats_"
            )
            
            # Dataset browser
            dataset_browser(known_faces_dir, key_prefix="dm_known_browser_")
    
    with tab3:
        # Browse test images
        test_images_dir = os.path.join(st.session_state.paths["test_datasets_dir"], "test_images")
        
        if not os.path.exists(test_images_dir):
            st.warning(f"Test images directory not found at: {test_images_dir}")
            st.info("Please set up the test dataset first.")
        else:
            # Dataset statistics
            dataset_statistics_visualization(
                test_images_dir,
                title="Test Images Statistics",
                key_prefix="dm_test_stats_"
            )
            
            # Dataset browser
            dataset_browser(test_images_dir, key_prefix="dm_test_browser_")
    
    # Information panel
    with st.expander("About UTKFace Dataset", expanded=False):
        st.markdown("""
        ### UTKFace Dataset
        
        The UTKFace dataset (University of Tennessee, Knoxville Face Dataset) is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of almost 30,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc.
        
        ### Dataset Structure
        
        Each image in the dataset is labeled by:
        - **Age**: Ranges from 0 to 116 years
        - **Gender**: Male (0) or Female (1)
        - **Ethnicity** (Race): White (0), Black (1), Asian (2), Indian (3), Others (4)
        
        The filenames follow the pattern:
        `[age]_[gender]_[race]_[date&time].jpg`
        
        ### How We Use the Dataset
        
        In this application, we use the UTKFace dataset for:
        
        1. **Bias Testing**: Measuring and visualizing facial recognition accuracy across different demographic groups
        2. **Reference Faces**: Creating a diverse set of known faces for the face matching feature
        3. **Test Images**: Generating test cases for evaluating face recognition performance
        
        ### Ethical Considerations
        
        When using this dataset, it's important to be aware of:
        
        - **Categorization Limitations**: The ethnicity categorization is simplified and may not fully represent human diversity
        - **Research Purpose**: The dataset is intended for research and educational purposes only
        - **Privacy**: Although these are public images, we should still respect privacy by using anonymization when appropriate
        - **Context**: Understanding that facial recognition technologies can have different impacts across demographic groups
        
        ### Attribution
        
        The UTKFace dataset was made available by the researchers at University of Tennessee, Knoxville.
        """)

if __name__ == "__main__":
    dataset_management_page()
