"""
Main Streamlit application entry point for the Facial Recognition Software Project.

This module serves as the entry point for the Streamlit web interface. It handles
the main layout, navigation, and integration with the backend facial recognition
functionality.
"""

import streamlit as st
import os
import sys
import time
from PIL import Image
import cv2
import numpy as np

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import backend functionality
from src.backend.face_detection import FaceDetector 
from src.backend.face_matching import FaceMatcher
from src.backend.anonymization import FaceAnonymizer
from src.backend.bias_testing import BiasAnalyzer
from src.utils.image_processing import ImageProcessor
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import all functionality directly
from src.ui.streamlit.pages_modules.face_detection import face_detection_page, process_detection_frame
from src.ui.streamlit.pages_modules.face_matching import face_matching_page, process_matching_frame
from src.ui.streamlit.pages_modules.face_anonymization import face_anonymization_page, process_anonymization_frame
from src.ui.streamlit.pages_modules.bias_testing import bias_testing_page, run_bias_test
from src.ui.streamlit.pages_modules.dataset_management import dataset_management_page

# Get configuration and setup logging
config = get_config()
logger = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition Demo",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    
    if "config" not in st.session_state:
        # Initialize configuration for each feature
        st.session_state.config = {
            "detection": {
                "confidence": config.detection.confidence,
                "use_hog": False,
                "batch_size": 4
            },
            "matching": {
                "threshold": config.matching.threshold,
                "comparison_mode": "distance",
                "use_best_match_only": True
            },
            "anonymization": {
                "method": config.anonymization.default_method,
                "intensity": config.anonymization.default_intensity,
                "preview": True,
                "show_boxes": True,
                "show_labels": True
            },
            "bias_testing": {
                "dataset": "demographic_split_set",
                "custom_path": "",
                "selected_groups": ["white", "black", "asian", "indian", "others"],
                "detailed_analysis": False,
                "chart_type": "bar",
                "show_overall_avg": True,
                "color_scheme": "default"
            },
            "dataset_management": {
                "action": "Download UTKFace Dataset",
                "sample_size": 500,
                "ethnicity_selection": "All ethnicities",
                "selected_ethnicities": ["White", "Black", "Asian", "Indian", "Others"],
                "images_per_ethnicity": 25,
                "num_people": 20,
                "ethnicity_balanced": True,
                "num_known": 5,
                "num_unknown": 5
            }
        }
    
    # Data directories from configuration
    if "paths" not in st.session_state:
        st.session_state.paths = {
            "known_faces_dir": config.paths.known_faces_dir,
            "datasets_dir": config.paths.datasets_dir,
            "results_dir": config.paths.results_dir,
            "test_datasets_dir": config.paths.test_datasets_dir,
            "demographic_split_dir": config.paths.demographic_split_set_dir
        }
    
    # Feature instance cache
    if "instances" not in st.session_state:
        st.session_state.instances = {}

# Initialize session state
init_session_state()

# Function to get class instances with caching
def get_instance(class_name, *args, **kwargs):
    """Get or create an instance of a class with caching."""
    if class_name not in st.session_state.instances:
        if class_name == "FaceDetector":
            st.session_state.instances[class_name] = FaceDetector(*args, **kwargs)
        elif class_name == "FaceMatcher":
            st.session_state.instances[class_name] = FaceMatcher(*args, **kwargs)
        elif class_name == "FaceAnonymizer":
            st.session_state.instances[class_name] = FaceAnonymizer(*args, **kwargs)
        elif class_name == "BiasAnalyzer":
            st.session_state.instances[class_name] = BiasAnalyzer(*args, **kwargs)
        elif class_name == "ImageProcessor":
            st.session_state.instances[class_name] = ImageProcessor(*args, **kwargs)
    return st.session_state.instances[class_name]

# CSS customization
def apply_custom_css():
    """Apply custom CSS styles to the Streamlit app."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #4A90E2 !important;
        margin-bottom: 1rem !important;
    }
    .subheader {
        font-size: 1.5rem !important;
        color: #3F3F3F !important;
        margin-bottom: 1rem !important;
    }
    .info-box {
        background-color: #E6F0FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3E6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        margin-bottom: 0.5rem !important;
    }
    .no-padding {
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Import the CSS
def local_css(file_name):
    """Load local CSS."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the custom CSS
style_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(style_path):
    local_css(style_path)

# Apply additional custom CSS
apply_custom_css()

# Navigation functions
def set_page(page_name):
    """Set the current page in session state."""
    st.session_state.page = page_name
    # Reset temporary UI state for the page
    if f"{page_name}_state" in st.session_state:
        del st.session_state[f"{page_name}_state"]

# Sidebar navigation
def render_sidebar():
    """Render the sidebar navigation menu."""
    st.sidebar.markdown('<div class="sidebar-title">Facial Recognition Demo</div>', unsafe_allow_html=True)
    
    # Feature selection
    st.sidebar.subheader("Features")
    
    # Use buttons for navigation instead of radio
    if st.sidebar.button("🏠 Main", 
                type="primary" if st.session_state.page == "Home" else "secondary",
                key="nav_home",
                use_container_width=True):
        set_page("Home")
    
    if st.sidebar.button("📷 Face Detection", 
                type="primary" if st.session_state.page == "Face Detection" else "secondary",
                key="nav_face_detection",
                use_container_width=True):
        set_page("Face Detection")
    
    if st.sidebar.button("🔍 Face Matching", 
                type="primary" if st.session_state.page == "Face Matching" else "secondary",
                key="nav_face_matching",
                use_container_width=True):
        set_page("Face Matching")
    
    if st.sidebar.button("🥸 Face Anonymization", 
                type="primary" if st.session_state.page == "Face Anonymization" else "secondary",
                key="nav_face_anonymization",
                use_container_width=True):
        set_page("Face Anonymization")
    
    if st.sidebar.button("📊 Bias Testing", 
                type="primary" if st.session_state.page == "Bias Testing" else "secondary",
                key="nav_bias_testing",
                use_container_width=True):
        set_page("Bias Testing")
    
    if st.sidebar.button("💾 Dataset Management", 
                type="primary" if st.session_state.page == "Dataset Management" else "secondary",
                key="nav_dataset_management",
                use_container_width=True):
        set_page("Dataset Management")
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Add info section
    st.sidebar.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.sidebar.info(
        "This application demonstrates facial recognition capabilities "
        "while exploring ethical considerations such as privacy and bias."
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Show paths information in an expander
    with st.sidebar.expander("🔧 Data Directories"):
        for name, path in st.session_state.paths.items():
            st.sidebar.code(f"{name}: {path}")
            
            # Add check if directory exists
            if not os.path.exists(path):
                st.sidebar.warning(f"Directory doesn't exist: {path}")
            elif name == "known_faces_dir":
                # Count known faces
                try:
                    face_count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    st.sidebar.text(f"Contains {face_count} known faces")
                except:
                    st.sidebar.text("Error reading directory")

# Home page
def render_home():
    """Render the home page with introduction and feature overview."""
    st.markdown('<h1 class="main-header">Facial Recognition Software Project</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Demonstration and Ethical Exploration</h2>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the Facial Recognition Software Project. This application demonstrates 
    facial recognition technology while exploring important ethical considerations 
    around privacy, bias, and consent.
    
    ## 🌟 Key Features
    """)
    
    # Feature highlights in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📷 Face Detection
        Detect faces in images and webcam feeds using computer vision algorithms.
        
        ### 🔍 Face Matching
        Match detected faces against known references for identification.
        """)
    
    with col2:
        st.markdown("""
        ### 🥸 Face Anonymization
        Apply privacy filters to faces with methods like blurring and pixelation.
        
        ### 📊 Bias Testing
        Analyze algorithm performance across demographic groups to detect bias.
        """)
    
    # Ethical focus
    st.markdown("""
    ## 🤔 Ethical Focus
    
    This project emphasizes important ethical considerations in facial recognition:
    
    - **Privacy**: How to balance utility with protecting individual privacy
    - **Bias**: Measuring and addressing algorithmic bias across demographics
    - **Consent**: Respecting individual agency and informed consent
    - **Transparency**: Understanding how these systems work and their limitations
    """)
    
    # Getting started instructions
    st.markdown("""
    ## 🚀 Getting Started
    
    1. Use the sidebar navigation to explore different features
    2. Start with the Face Detection feature to detect faces in images
    3. Try Face Matching to identify people against references
    4. Explore Face Anonymization to learn about privacy techniques
    5. Use Bias Testing to understand algorithmic fairness issues
    
    For a complete experience, you'll need to set up required datasets using the 
    Dataset Management feature.
    """)
    
    # UTKFace dataset info
    with st.expander("About the UTKFace Dataset"):
        st.markdown("""
        ### UTKFace Dataset
        
        This project uses the UTKFace (University of Tennessee, Knoxville Face) dataset for demographic bias testing.
        
        **Key features of the dataset:**
        - Contains over 20,000 face images with age, gender, and ethnicity annotations
        - Age ranges from 0 to 116 years
        - Ethnicity categories: White, Black, Asian, Indian, and Others
        
        To download and set up the dataset, use the Dataset Management feature.
        """)
    
    # Check data directories
    missing_dirs = []
    for name, path in st.session_state.paths.items():
        if not os.path.exists(path):
            missing_dirs.append(name)
    
    if missing_dirs:
        st.warning(f"Some data directories don't exist: {', '.join(missing_dirs)}. Use the Dataset Management feature to set them up.")
# Main application
def main():
    """Main function to run the Streamlit application."""
    
    # Render sidebar
    render_sidebar()
    
    # Route to the appropriate page
    if st.session_state.page == "Home":
        render_home()
    elif st.session_state.page == "Face Detection":
        face_detection_page()
    elif st.session_state.page == "Face Matching":
        face_matching_page()
    elif st.session_state.page == "Face Anonymization":
        face_anonymization_page()
    elif st.session_state.page == "Bias Testing":
        bias_testing_page()
    elif st.session_state.page == "Dataset Management":
        dataset_management_page()
    else:
        # Default to home
        render_home()

if __name__ == "__main__":
    main()
