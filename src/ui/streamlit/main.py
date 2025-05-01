"""
Main entry point for Streamlit application as a single-page app.
This avoids the duplicate menu issue by using a single file instead of Streamlit's pages directory.
"""

import streamlit as st
import os
import sys

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import page functionality
from src.ui.streamlit.pages.face_detection import face_detection_page
from src.ui.streamlit.pages.face_matching import face_matching_page
from src.ui.streamlit.pages.face_anonymization import face_anonymization_page
from src.ui.streamlit.pages.bias_testing import bias_testing_page
from src.ui.streamlit.pages.dataset_management import dataset_management_page

# Import from app.py
from src.ui.streamlit.app import render_home, render_about, apply_custom_css, get_config, init_session_state

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition Demo",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
init_session_state()

# Import the CSS
def local_css(file_name):
    """Load local CSS."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load the CSS
local_css(os.path.join(os.path.dirname(__file__), "style.css"))

# Apply the custom CSS from app.py
apply_custom_css()

# Navigation function
def set_page(page_name):
    """Set the current page in session state."""
    st.session_state.page = page_name
    # Reset temporary UI state for the page
    if f"{page_name}_state" in st.session_state:
        del st.session_state[f"{page_name}_state"]

# Sidebar navigation
def render_sidebar():
    """Render the single sidebar navigation menu."""
    st.sidebar.markdown('<div class="sidebar-title">Facial Recognition Demo</div>', unsafe_allow_html=True)
    
    # Feature selection
    st.sidebar.subheader("Features")
    
    # Use buttons for navigation
    if st.sidebar.button("📷 Face Detection", 
                type="primary" if st.session_state.page == "Face Detection" else "secondary",
                key="nav_face_detection",
                use_container_width=True):
        set_page("Face Detection")
        st.experimental_rerun()
        
    if st.sidebar.button("🔍 Face Matching", 
                type="primary" if st.session_state.page == "Face Matching" else "secondary",
                key="nav_face_matching",
                use_container_width=True):
        set_page("Face Matching")
        st.experimental_rerun()
        
    if st.sidebar.button("🥸 Face Anonymization", 
                type="primary" if st.session_state.page == "Face Anonymization" else "secondary",
                key="nav_face_anonymization",
                use_container_width=True):
        set_page("Face Anonymization")
        st.experimental_rerun()
        
    if st.sidebar.button("📊 Bias Testing", 
                type="primary" if st.session_state.page == "Bias Testing" else "secondary",
                key="nav_bias_testing",
                use_container_width=True):
        set_page("Bias Testing")
        st.experimental_rerun()
        
    if st.sidebar.button("💾 Dataset Management", 
                type="primary" if st.session_state.page == "Dataset Management" else "secondary",
                key="nav_dataset_management",
                use_container_width=True):
        set_page("Dataset Management")
        st.experimental_rerun()
        
    if st.sidebar.button("ℹ️ About", 
                type="primary" if st.session_state.page == "About" else "secondary",
                key="nav_about",
                use_container_width=True):
        set_page("About")
        st.experimental_rerun()
    
    # Add a separator
    st.sidebar.markdown("---")
    
    # Add info section
    st.sidebar.info(
        "This application demonstrates facial recognition capabilities "
        "while exploring ethical considerations such as privacy and bias."
    )
    
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

# Main application
def main():
    """Main function to run the Streamlit application as a single-page app."""
    
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
    elif st.session_state.page == "About":
        render_about()
    else:
        # Default to home
        render_home()

if __name__ == "__main__":
    main()
