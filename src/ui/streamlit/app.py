"""
Main Streamlit application entry point for the Facial Recognition Software Project.

This module serves as the entry point for the Streamlit web interface. It handles
the main layout, navigation, and integration with the core facial recognition
functionality.
"""

import streamlit as st
import os
import sys

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import core functionality
from src.core.face_detection import FaceDetector 
from src.core.face_matching import FaceMatcher
from src.core.anonymization import FaceAnonymizer
from src.core.bias_testing import BiasAnalyzer
from src.utils.image_processing import ImageProcessor
from src.utils.config import get_config

# Get configuration
config = get_config()

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition Demo",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main function to run the Streamlit application."""
    
    # Title and introduction
    st.title("Facial Recognition Software Demo")
    st.subheader("Explore Facial Recognition Technology and Ethics")
    
    # Sidebar navigation
    st.sidebar.title("Features")
    
    # Feature selection
    feature = st.sidebar.radio(
        "Select a feature:",
        [
            "📷 Face Detection",
            "🔍 Face Matching",
            "🥸 Face Anonymization", 
            "📊 Bias Testing",
            "💾 Dataset Management",
            "ℹ️ About"
        ]
    )
    
    # Add a sidebar info section
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application demonstrates facial recognition capabilities "
        "while exploring ethical considerations such as privacy and bias."
    )
    
    # Display selected feature (placeholder)
    if feature == "📷 Face Detection":
        st.header("Face Detection")
        st.markdown(
            "This feature detects faces in images and draws bounding boxes around them."
        )
        st.warning("Streamlit integration is under development. Please use the CLI version for now.")
        
    elif feature == "🔍 Face Matching":
        st.header("Face Matching")
        st.markdown(
            "This feature compares detected faces against known reference faces."
        )
        st.warning("Streamlit integration is under development. Please use the CLI version for now.")
        
    elif feature == "🥸 Face Anonymization":
        st.header("Face Anonymization") 
        st.markdown(
            "This feature applies privacy-preserving filters to faces in images."
        )
        st.warning("Streamlit integration is under development. Please use the CLI version for now.")
        
    elif feature == "📊 Bias Testing":
        st.header("Bias Testing")
        st.markdown(
            "This feature analyzes facial recognition accuracy across demographic groups."
        )
        st.warning("Streamlit integration is under development. Please use the CLI version for now.")
        
    elif feature == "💾 Dataset Management":
        st.header("Dataset Management")
        st.markdown(
            "This feature helps prepare datasets for testing and evaluation."
        )
        st.warning("Streamlit integration is under development. Please use the CLI version for now.")
        
    elif feature == "ℹ️ About":
        st.header("About This Project")
        st.markdown(
            """
            ## Facial Recognition Software Project
            
            This project demonstrates facial recognition technology while exploring ethical 
            considerations. It provides tools for face detection, matching, anonymization,
            and bias testing.
            
            ### Features
            
            - **Face Detection**: Detect faces in images and video streams
            - **Face Matching**: Match faces against known references
            - **Face Anonymization**: Apply privacy filters to faces
            - **Bias Testing**: Analyze accuracy across demographic groups
            - **Dataset Management**: Prepare datasets for testing
            
            ### Ethical Focus
            
            This project emphasizes ethical considerations in facial recognition, including:
            
            - **Privacy**: How to protect individual privacy with anonymization techniques
            - **Bias**: How to detect and measure algorithmic bias across demographics
            - **Consent**: Respecting individual agency in facial recognition
            - **Transparency**: Understanding how these systems work
            """
        )

if __name__ == "__main__":
    main()
