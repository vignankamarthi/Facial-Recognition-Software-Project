"""
Bias Testing page for the Streamlit interface.

This module provides the Streamlit UI for the bias testing feature, allowing users
to analyze facial recognition accuracy across different demographic groups.
"""

import streamlit as st
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import pandas as pd

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))))

# Import core functionality
from src.backend.bias_testing import BiasAnalyzer
from src.utils.config import get_config
from src.utils.logger import get_logger

# Import UI components
from src.ui.streamlit.components import (
    bias_testing_config_panel,
    bias_metrics_visualization,
    dataset_statistics_visualization,
    dataset_browser
)

# Setup logging
logger = get_logger(__name__)

# Get configuration
config = get_config()

def run_bias_test():
    """
    Run the bias testing process.
    
    Returns
    -------
    dict
        Results dictionary from BiasAnalyzer
    """
    # Get bias analyzer instance
    analyzer = BiasAnalyzer(test_datasets_dir=st.session_state.paths["test_datasets_dir"])
    
    # Get bias testing parameters
    bias_params = st.session_state.config["bias_testing"]
    
    # Check if test dataset directory exists
    if bias_params["dataset"] == "demographic_split_set":
        dataset_dir = os.path.join(st.session_state.paths["test_datasets_dir"], "demographic_split_set")
    else:
        dataset_dir = bias_params["custom_path"]
    
    if not os.path.exists(dataset_dir):
        st.error(f"Dataset directory not found: {dataset_dir}")
        st.info("Please set up the dataset first using the Dataset Management feature.")
        return None
    
    # Check if selected groups exist
    missing_groups = []
    for group in bias_params["selected_groups"]:
        group_dir = os.path.join(dataset_dir, group)
        if not os.path.exists(group_dir):
            missing_groups.append(group)
    
    if missing_groups:
        st.error(f"Some selected demographic groups don't exist: {', '.join(missing_groups)}")
        st.info("Please set up the dataset first using the Dataset Management feature.")
        return None
    
    # Show progress message
    with st.spinner("Running bias testing... This may take a few minutes."):
        if bias_params["detailed_analysis"]:
            # Run detailed analysis
            results = analyzer.analyze_demographic_bias(
                dataset_name=bias_params["dataset"],
                detailed=True
            )
        else:
            # Run standard analysis
            results = analyzer.test_recognition_accuracy(bias_params["dataset"])
    
    return results

def bias_testing_page():
    """Render the bias testing page."""
    
    st.markdown("# Bias Testing")
    st.markdown("Analyze facial recognition accuracy across different demographic groups.")
    
    # Create tabs for different functionality
    test_tab, data_tab, info_tab = st.tabs(["Run Bias Testing", "View Dataset", "Ethical Information"])
    
    with test_tab:
        # Bias testing settings
        with st.expander("Bias Testing Settings", expanded=True):
            updated_config = bias_testing_config_panel(
                st.session_state.config["bias_testing"],
                key_prefix="bt_"
            )
            st.session_state.config["bias_testing"] = updated_config
        
        # Check if dataset directories exist
        dataset_dir = os.path.join(
            st.session_state.paths["test_datasets_dir"], 
            st.session_state.config["bias_testing"]["dataset"]
        )
        
        if not os.path.exists(dataset_dir):
            st.warning(f"Dataset directory not found: {dataset_dir}")
            st.info("Please set up the dataset first using the Dataset Management feature.")
        else:
            # Count files in dataset
            file_count = 0
            for group in os.listdir(dataset_dir):
                group_dir = os.path.join(dataset_dir, group)
                if os.path.isdir(group_dir):
                    group_files = [f for f in os.listdir(group_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    file_count += len(group_files)
            
            if file_count == 0:
                st.warning(f"No image files found in dataset: {dataset_dir}")
                st.info("Please set up the dataset first using the Dataset Management feature.")
            else:
                st.info(f"Found {file_count} images in dataset.")
                
                # Run bias testing button
                if st.button("Run Bias Testing", type="primary", key="bt_run_button"):
                    # Run bias testing
                    results = run_bias_test()
                    
                    # Store results in session state
                    if results:
                        st.session_state["bias_results"] = results
                        st.success("Bias testing completed successfully!")
                    else:
                        st.error("Bias testing failed. See error messages above.")
        
        # Display results if available
        if "bias_results" in st.session_state:
            st.markdown("## Bias Testing Results")
            
            # Display visualization
            bias_metrics_visualization(
                st.session_state["bias_results"],
                chart_type=st.session_state.config["bias_testing"]["chart_type"],
                show_overall_avg=st.session_state.config["bias_testing"]["show_overall_avg"],
                color_scheme=st.session_state.config["bias_testing"]["color_scheme"],
                key_prefix="bt_viz_"
            )
            
            # Export results
            with st.expander("Export Results", expanded=False):
                # Convert results to dataframe for display
                try:
                    # Extract data for groups
                    group_data = []
                    for group, stats in st.session_state["bias_results"]["by_demographic"].items():
                        group_data.append({
                            "Demographic Group": group,
                            "Accuracy (%)": stats["accuracy"] * 100,
                            "Detected Faces": stats["detected"],
                            "Total Images": stats["total"]
                        })
                    
                    # Create DataFrame
                    results_df = pd.DataFrame(group_data)
                    
                    # Add overall row
                    overall = st.session_state["bias_results"]["overall"]
                    results_df = pd.concat([
                        results_df,
                        pd.DataFrame([{
                            "Demographic Group": "Overall",
                            "Accuracy (%)": overall["accuracy"] * 100,
                            "Detected Faces": overall["detected"],
                            "Total Images": overall["total"]
                        }])
                    ], ignore_index=True)
                    
                    # Add bias metrics if available
                    if "bias_analysis" in st.session_state["bias_results"]:
                        bias_analysis = st.session_state["bias_results"]["bias_analysis"]
                        
                        # Display bias metrics
                        st.markdown("### Bias Analysis Metrics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Accuracy Range", f"{bias_analysis['accuracy_range']*100:.2f}%")
                            
                            # Determine bias level
                            if bias_analysis['accuracy_range'] > 0.15:
                                bias_level = "🔴 High"
                            elif bias_analysis['accuracy_range'] > 0.05:
                                bias_level = "🟠 Moderate"
                            else:
                                bias_level = "🟢 Low"
                                
                            st.metric("Bias Level", bias_level)
                        
                        with col2:
                            if "std_deviation" in bias_analysis:
                                st.metric("Standard Deviation", f"{bias_analysis['std_deviation']*100:.2f}%")
                            if "variance" in bias_analysis:
                                st.metric("Variance", f"{bias_analysis['variance']*100:.4f}")
                    
                    # Display table
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button for CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="bias_testing_results.csv",
                        mime="text/csv",
                        key="bt_download_csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error preparing results export: {e}")
    
    with data_tab:
        st.markdown("## Dataset Browser")
        
        # Dataset selection
        dataset_options = ["demographic_split_set"]
        custom_datasets = []
        
        # Look for custom datasets in test_datasets_dir
        test_datasets_dir = st.session_state.paths["test_datasets_dir"]
        if os.path.exists(test_datasets_dir):
            for item in os.listdir(test_datasets_dir):
                item_path = os.path.join(test_datasets_dir, item)
                if os.path.isdir(item_path) and item not in dataset_options and not item.startswith("."):
                    custom_datasets.append(item)
        
        if custom_datasets:
            dataset_options.extend(custom_datasets)
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            index=0,
            key="bt_dataset_select"
        )
        
        # Construct dataset path
        dataset_path = os.path.join(test_datasets_dir, selected_dataset)
        
        if not os.path.exists(dataset_path):
            st.warning(f"Selected dataset directory not found: {dataset_path}")
        else:
            # Dataset statistics visualization
            stats = dataset_statistics_visualization(
                dataset_path,
                title=f"Dataset Statistics: {selected_dataset}",
                key_prefix="bt_stats_"
            )
            
            # Dataset browser
            st.markdown("### Browse Dataset Images")
            dataset_browser(dataset_path, key_prefix="bt_browser_")
    
    with info_tab:
        st.markdown("## Ethical Considerations in Facial Recognition")
        
        st.markdown("""
        ### Understanding Algorithmic Bias
        
        Facial recognition systems can exhibit bias across different demographic groups due to various factors:
        
        1. **Training Data Imbalance**: Models trained on datasets that underrepresent certain demographics
        2. **Algorithm Design**: Technical choices in feature extraction and matching that favor certain facial characteristics
        3. **Evaluation Methods**: Testing protocols that may not adequately measure performance across demographics
        4. **Feedback Loops**: Systems that reinforce existing biases through ongoing use and data collection
        
        ### The UTKFace Dataset
        
        This project uses the UTKFace (University of Tennessee, Knoxville Face) dataset which provides:
        
        - **Demographic Labels**: Age, gender, and ethnicity annotations for facial images
        - **Diverse Representation**: Over 20,000 images with varied demographic characteristics
        - **Testing Framework**: Tools to measure performance differences across groups
        
        The dataset enables us to:
        1. Identify potential bias in recognition accuracy
        2. Quantify disparities using statistical measures
        3. Visualize performance differences
        
        ### Bias Testing Methodology
        
        Our bias testing approach:
        
        1. **Separated Testing**: Processes images by demographic group
        2. **Consistent Parameters**: Uses the same detection settings for all groups
        3. **Statistical Analysis**: Measures variance, standard deviation, and other metrics
        4. **Visualization**: Creates visual representations of accuracy differences
        
        ### Ethical Implications
        
        Algorithmic bias in facial recognition raises important ethical questions:
        
        - **Fair Treatment**: Should facial recognition systems work equally well for everyone?
        - **Disparate Impact**: How do accuracy differences affect different communities?
        - **Responsibility**: Who is accountable for addressing bias in systems?
        - **Transparency**: Should users be informed about demographic performance differences?
        
        ### Mitigation Strategies
        
        Several approaches can help address algorithmic bias:
        
        1. **Diverse Training Data**: Using demographically balanced datasets
        2. **Algorithmic Fairness**: Designing algorithms with fairness constraints
        3. **Regular Testing**: Continuous evaluation across demographic groups
        4. **User Education**: Informing users about limitations and potential biases
        """)
        
        # Add a note about the ethical discussion document
        st.info("""
        For a more comprehensive discussion of ethical considerations, see the
        'Ethical Discussion' document in the project documentation.
        """)

if __name__ == "__main__":
    bias_testing_page()
