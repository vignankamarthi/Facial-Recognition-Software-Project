"""
UI Components for the Streamlit Interface

This package contains reusable components for the Streamlit interface
that can be used across different pages.
"""

# Export key components for easier imports
from .webcam import webcam_component, image_upload_component
from .streamlit_webcam import streamlit_webcam_component
from .config_panels import (
    detection_config_panel,
    matching_config_panel,
    anonymization_config_panel,
    bias_testing_config_panel,
    dataset_management_config_panel
)
from .known_faces import (
    known_faces_grid,
    add_known_face,
    remove_known_faces
)
from .visualizations import (
    bias_metrics_visualization,
    dataset_statistics_visualization,
    dataset_browser,
    before_after_comparison
)
