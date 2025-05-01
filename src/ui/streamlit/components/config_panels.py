"""
Configuration Panels for Streamlit Interface

This module provides reusable configuration panels that can be used across
different pages of the Streamlit interface for adjusting parameters and settings.
"""

import streamlit as st
from typing import Dict, Any, Callable, List, Optional, Tuple


def detection_config_panel(config: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    """
    Configuration panel for face detection settings.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    st.subheader("Detection Settings")
    
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Detection confidence slider
    confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=config.get("confidence", 0.5),
        step=0.05,
        format="%.2f",
        help="Higher values require more confidence in face detection, reducing false positives but potentially missing some faces.",
        key=f"{key_prefix}confidence_slider"
    )
    updated_config["confidence"] = confidence
    
    # Advanced Settings section
    st.markdown("### Advanced Settings")
    
    # Additional parameters could be added here
    use_hog = st.checkbox(
        "Use HOG detector (faster but less accurate)",
        value=config.get("use_hog", False),
        help="HOG is faster but less accurate. CNN is slower but more accurate.",
        key=f"{key_prefix}use_hog_checkbox"
    )
    updated_config["use_hog"] = use_hog
    
    if not use_hog:
        # Only show CNN batch size if CNN detector is selected
        batch_size = st.slider(
            "CNN Batch Size",
            min_value=1,
            max_value=16,
            value=config.get("batch_size", 4),
            step=1,
            help="Larger batch sizes can be faster but use more memory.",
            key=f"{key_prefix}batch_size_slider"
        )
        updated_config["batch_size"] = batch_size
    
    return updated_config


def matching_config_panel(config: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    """
    Configuration panel for face matching settings.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    st.subheader("Matching Settings")
    
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Matching threshold slider
    threshold = st.slider(
        "Matching Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.get("threshold", 0.6),
        step=0.05,
        format="%.2f",
        help="Higher values require closer matches. Increase to reduce false matches, decrease to find more potential matches.",
        key=f"{key_prefix}threshold_slider"
    )
    updated_config["threshold"] = threshold
    
    # Advanced Settings section
    st.markdown("### Advanced Settings")
    
    # Comparison mode
    comparison_mode = st.radio(
        "Comparison Mode",
        options=["distance", "similarity"],
        index=0 if config.get("comparison_mode", "distance") == "distance" else 1,
        help="Distance measures difference between faces (lower is better match). Similarity measures likeness (higher is better match).",
        key=f"{key_prefix}comparison_mode_radio"
    )
    updated_config["comparison_mode"] = comparison_mode
    
    # Use all known faces vs best match only
    use_best_match_only = st.checkbox(
        "Use Best Match Only",
        value=config.get("use_best_match_only", True),
        help="If checked, only the best matching face is returned. Otherwise, all matches above threshold are returned.",
        key=f"{key_prefix}best_match_checkbox"
    )
    updated_config["use_best_match_only"] = use_best_match_only
    
    return updated_config


def anonymization_config_panel(config: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    """
    Configuration panel for face anonymization settings.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    st.subheader("Anonymization Settings")
    
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Anonymization method
    method = st.selectbox(
        "Anonymization Method",
        options=["blur", "pixelate", "mask"],
        index=["blur", "pixelate", "mask"].index(config.get("method", "blur")),
        help="Blur: Gaussian blur filter. Pixelate: Block-like effect. Mask: Solid overlay.",
        key=f"{key_prefix}method_select"
    )
    updated_config["method"] = method
    
    # Intensity slider (appropriate label based on method)
    intensity_label = {
        "blur": "Blur Intensity",
        "pixelate": "Pixelation Level",
        "mask": "Opacity",
    }.get(method, "Intensity")
    
    intensity = st.slider(
        intensity_label,
        min_value=1,
        max_value=100,
        value=config.get("intensity", 90),
        step=1,
        help="Higher values provide stronger anonymization effects.",
        key=f"{key_prefix}intensity_slider"
    )
    updated_config["intensity"] = intensity
    
    # Preview anonymization (checkbox)
    preview = st.checkbox(
        "Show Before/After Preview",
        value=config.get("preview", True),
        help="Show both original and anonymized images side by side.",
        key=f"{key_prefix}preview_checkbox"
    )
    updated_config["preview"] = preview
    
    # Additional Options section
    st.markdown("### Additional Options")
    
    # Show boxes around anonymized faces
    show_boxes = st.checkbox(
        "Show Boxes Around Anonymized Faces",
        value=config.get("show_boxes", True),
        help="Display colored outline around anonymized face regions.",
        key=f"{key_prefix}show_boxes_checkbox"
    )
    updated_config["show_boxes"] = show_boxes
    
    # Show anonymization method on image
    show_labels = st.checkbox(
        "Show Anonymization Labels",
        value=config.get("show_labels", True),
        help="Display text labels indicating anonymization method used.",
        key=f"{key_prefix}show_labels_checkbox"
    )
    updated_config["show_labels"] = show_labels
    
    return updated_config


def bias_testing_config_panel(config: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    """
    Configuration panel for bias testing settings.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    st.subheader("Bias Testing Settings")
    
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Dataset selection
    dataset = st.selectbox(
        "Dataset",
        options=["demographic_split_set", "custom"],
        index=0 if config.get("dataset", "demographic_split_set") == "demographic_split_set" else 1,
        help="Select which dataset to use for bias testing.",
        key=f"{key_prefix}dataset_select"
    )
    updated_config["dataset"] = dataset
    
    # If custom dataset, show path input
    if dataset == "custom":
        custom_path = st.text_input(
            "Custom Dataset Path",
            value=config.get("custom_path", ""),
            help="Path to custom dataset directory with demographic subdirectories.",
            key=f"{key_prefix}custom_path_input"
        )
        updated_config["custom_path"] = custom_path
    
    # Demographic groups to include
    all_groups = ["white", "black", "asian", "indian", "others"]
    selected_groups = st.multiselect(
        "Demographic Groups to Test",
        options=all_groups,
        default=config.get("selected_groups", all_groups),
        help="Select which demographic groups to include in the bias testing.",
        key=f"{key_prefix}groups_multiselect"
    )
    updated_config["selected_groups"] = selected_groups
    
    # Analysis level
    detailed_analysis = st.checkbox(
        "Detailed Statistical Analysis",
        value=config.get("detailed_analysis", False),
        help="Perform comprehensive statistical analysis including standard deviation, variance, and mean absolute deviation.",
        key=f"{key_prefix}detailed_analysis_checkbox"
    )
    updated_config["detailed_analysis"] = detailed_analysis
    
    # Visualization options
    # Visualization Options section
    st.markdown("### Visualization Options")
    
    chart_type = st.selectbox(
        "Chart Type",
        options=["bar", "line", "scatter"],
        index=0 if config.get("chart_type", "bar") == "bar" else (1 if config.get("chart_type", "bar") == "line" else 2),
        help="Type of chart to visualize bias results.",
        key=f"{key_prefix}chart_type_select"
    )
    updated_config["chart_type"] = chart_type
    
    show_overall_avg = st.checkbox(
        "Show Overall Average Line",
        value=config.get("show_overall_avg", True),
        help="Display a line showing the overall average across all groups.",
        key=f"{key_prefix}show_avg_checkbox"
    )
    updated_config["show_overall_avg"] = show_overall_avg
    
    color_scheme = st.selectbox(
        "Color Scheme",
        options=["default", "colorblind_friendly", "custom"],
        index=0 if config.get("color_scheme", "default") == "default" else (1 if config.get("color_scheme", "default") == "colorblind_friendly" else 2),
        help="Select color scheme for visualizations.",
        key=f"{key_prefix}color_scheme_select"
    )
    updated_config["color_scheme"] = color_scheme
    
    return updated_config


def dataset_management_config_panel(config: Dict[str, Any], key_prefix: str = "") -> Dict[str, Any]:
    """
    Configuration panel for dataset management settings.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Current configuration dictionary
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Updated configuration dictionary
    """
    st.subheader("Dataset Management Settings")
    
    # Create a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Action selection
    action = st.selectbox(
        "Action",
        options=[
            "Download UTKFace Dataset",
            "Set up Bias Testing with UTKFace",
            "Prepare Known Faces from UTKFace",
            "Prepare Test Dataset from UTKFace"
        ],
        index=0,
        help="Select dataset management action to perform.",
        key=f"{key_prefix}action_select"
    )
    updated_config["action"] = action
    
    # Settings based on selected action
    if action == "Download UTKFace Dataset":
        sample_size = st.slider(
            "Sample Size",
            min_value=100,
            max_value=30000,
            value=config.get("sample_size", 500),
            step=100,
            help="Number of images to download from UTKFace dataset.",
            key=f"{key_prefix}sample_size_slider"
        )
        updated_config["sample_size"] = sample_size
        
        # Ethnicity selection
        ethnicity_options = [
            "All ethnicities",
            "White and Black only",
            "White, Black, and Asian",
            "Custom selection"
        ]
        ethnicity_selection = st.radio(
            "Ethnicities to Include",
            options=ethnicity_options,
            index=0,
            help="Select which ethnicities to include in the dataset.",
            key=f"{key_prefix}ethnicity_radio"
        )
        updated_config["ethnicity_selection"] = ethnicity_selection
        
        # If custom selection, show multiselect
        if ethnicity_selection == "Custom selection":
            all_ethnicities = ["White", "Black", "Asian", "Indian", "Others"]
            selected_ethnicities = st.multiselect(
                "Select Ethnicities",
                options=all_ethnicities,
                default=config.get("selected_ethnicities", all_ethnicities),
                help="Select which ethnicities to include in the dataset.",
                key=f"{key_prefix}ethnicities_multiselect"
            )
            updated_config["selected_ethnicities"] = selected_ethnicities
    
    elif action == "Set up Bias Testing with UTKFace":
        images_per_ethnicity = st.slider(
            "Images per Ethnicity",
            min_value=10,
            max_value=100,
            value=config.get("images_per_ethnicity", 25),
            step=5,
            help="Number of images per ethnicity to include in the bias testing dataset.",
            key=f"{key_prefix}images_per_ethnicity_slider"
        )
        updated_config["images_per_ethnicity"] = images_per_ethnicity
    
    elif action == "Prepare Known Faces from UTKFace":
        num_people = st.slider(
            "Number of People",
            min_value=5,
            max_value=50,
            value=config.get("num_people", 20),
            step=5,
            help="Number of reference faces to extract from UTKFace dataset.",
            key=f"{key_prefix}num_people_slider"
        )
        updated_config["num_people"] = num_people
        
        ethnicity_balanced = st.checkbox(
            "Balance Ethnicity Representation",
            value=config.get("ethnicity_balanced", True),
            help="Ensure equal representation of ethnicities in known faces.",
            key=f"{key_prefix}ethnicity_balanced_checkbox"
        )
        updated_config["ethnicity_balanced"] = ethnicity_balanced
    
    elif action == "Prepare Test Dataset from UTKFace":
        num_known = st.slider(
            "Number of Known People",
            min_value=5,
            max_value=30,
            value=config.get("num_known", 5),
            step=5,
            help="Number of known people to include in test dataset.",
            key=f"{key_prefix}num_known_slider"
        )
        updated_config["num_known"] = num_known
        
        num_unknown = st.slider(
            "Number of Unknown People",
            min_value=5,
            max_value=30,
            value=config.get("num_unknown", 5),
            step=5,
            help="Number of unknown people to include in test dataset.",
            key=f"{key_prefix}num_unknown_slider"
        )
        updated_config["num_unknown"] = num_unknown
    
    return updated_config
