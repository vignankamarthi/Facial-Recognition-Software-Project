"""
Visualization Components for Streamlit Interface

This module provides reusable visualization components for displaying
bias metrics, dataset statistics, and other visualizations in the
Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional


def bias_metrics_visualization(
    results: Dict[str, Any], 
    chart_type: str = "bar",
    show_overall_avg: bool = True,
    color_scheme: str = "default",
    width: int = 800,
    height: int = 400,
    key_prefix: str = ""
) -> None:
    """
    Display bias testing results in a visualization.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from BiasAnalyzer
    chart_type : str, optional
        Type of chart to display (default: "bar")
    show_overall_avg : bool, optional
        Whether to show overall average line (default: True)
    color_scheme : str, optional
        Color scheme to use (default: "default")
    width : int, optional
        Width of the chart in pixels (default: 800)
    height : int, optional
        Height of the chart in pixels (default: 400)
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
    """
    if not results or "by_demographic" not in results:
        st.warning("No bias testing results to visualize.")
        return
    
    # Extract data from results
    demographics = []
    accuracies = []
    counts = []
    
    for demographic, stats in results["by_demographic"].items():
        demographics.append(demographic)
        accuracies.append(stats["accuracy"] * 100)  # Convert to percentage
        counts.append(stats["total"])
    
    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({
        "Demographic": demographics,
        "Accuracy": accuracies,
        "Count": counts
    })
    
    # Sort by accuracy (highest to lowest)
    data = data.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    
    # Select color scheme
    if color_scheme == "default":
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#34495e", "#7f8c8d"]
    elif color_scheme == "colorblind_friendly":
        colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]
    else:  # custom
        colors = None  # Use matplotlib defaults
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Create the appropriate chart type
    if chart_type == "bar":
        if colors:
            bars = ax.bar(data["Demographic"], data["Accuracy"], color=colors[:len(demographics)])
        else:
            bars = ax.bar(data["Demographic"], data["Accuracy"])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10
            )
    
    elif chart_type == "line":
        if colors:
            ax.plot(data["Demographic"], data["Accuracy"], marker="o", linewidth=2, markersize=8, color=colors[0])
        else:
            ax.plot(data["Demographic"], data["Accuracy"], marker="o", linewidth=2, markersize=8)
    
    elif chart_type == "scatter":
        if colors:
            scatter = ax.scatter(
                data["Demographic"], 
                data["Accuracy"], 
                s=data["Count"]/5,  # Size proportional to count
                alpha=0.7,
                c=range(len(demographics)),
                cmap="viridis"
            )
        else:
            scatter = ax.scatter(
                data["Demographic"], 
                data["Accuracy"], 
                s=data["Count"]/5,  # Size proportional to count
                alpha=0.7
            )
            
        # Add a legend for the sizes
        sizes = [min(counts), max(counts)]
        handles, labels = scatter.legend_elements(prop="sizes", num=2, alpha=0.6)
        size_legend = ax.legend(
            handles, [f"{sizes[0]} images", f"{sizes[1]} images"], 
            loc="upper right", title="Sample Size"
        )
        ax.add_artist(size_legend)
    
    # Add overall average line if requested
    if show_overall_avg and "overall" in results and "accuracy" in results["overall"]:
        overall_acc = results["overall"]["accuracy"] * 100
        ax.axhline(
            y=overall_acc, 
            color="red", 
            linestyle="--", 
            linewidth=2,
            label=f"Overall Average: {overall_acc:.1f}%"
        )
        
        # Add legend
        if chart_type != "scatter" or not colors:
            ax.legend(loc="best")
    
    # Set labels and title
    ax.set_xlabel("Demographic Group")
    ax.set_ylabel("Face Detection Accuracy (%)")
    ax.set_title("Face Detection Accuracy by Demographic Group")
    
    # Set y-axis range
    ax.set_ylim(0, max(accuracies) * 1.15)  # Add some padding above highest value
    
    # Add grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(demographics) > 4 else 0)
    
    # Make the plot tight
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig)
    
    # Display additional statistics
    st.subheader("Detailed Statistics")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Column 1: Accuracy metrics table
    col1.subheader("Accuracy by Group")
    accuracy_df = pd.DataFrame({
        "Group": data["Demographic"],
        "Accuracy (%)": data["Accuracy"].round(2),
        "Sample Size": data["Count"]
    })
    col1.dataframe(accuracy_df, use_container_width=True)
    
    # Column 2: Bias metrics
    col2.subheader("Bias Metrics")
    
    # Calculate bias metrics
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    acc_range = max_acc - min_acc
    std_dev = np.std(accuracies)
    
    # Determine bias level
    if acc_range > 15:
        bias_level = "High"
        bias_color = "🔴"
    elif acc_range > 5:
        bias_level = "Moderate"
        bias_color = "🟠"
    else:
        bias_level = "Low"
        bias_color = "🟢"
    
    # Display metrics
    col2.metric("Maximum Accuracy", f"{max_acc:.2f}%", f"{max_acc - min_acc:.2f}% above minimum")
    col2.metric("Minimum Accuracy", f"{min_acc:.2f}%", f"{min_acc - max_acc:.2f}% below maximum")
    col2.metric("Accuracy Range", f"{acc_range:.2f}%")
    col2.metric("Standard Deviation", f"{std_dev:.2f}%")
    
    if "bias_analysis" in results:
        # Display additional metrics from bias_analysis
        bias_analysis = results["bias_analysis"]
        
        if "std_deviation" in bias_analysis:
            col2.metric("Standard Deviation", f"{bias_analysis['std_deviation'] * 100:.2f}%")
        
        if "variance" in bias_analysis:
            col2.metric("Variance", f"{bias_analysis['variance'] * 100:.4f}")
        
        if "mean_abs_deviation" in bias_analysis:
            col2.metric("Mean Absolute Deviation", f"{bias_analysis['mean_abs_deviation'] * 100:.2f}%")
    
    # Display bias level assessment
    col2.markdown(f"### Bias Level: {bias_color} {bias_level}")
    col2.markdown("""
    **Interpretation**:
    - Range > 15%: High bias potential
    - Range 5-15%: Moderate bias potential
    - Range < 5%: Low bias potential
    """)
    
    # Show best and worst performing groups
    max_group = data.loc[data["Accuracy"].idxmax(), "Demographic"]
    min_group = data.loc[data["Accuracy"].idxmin(), "Demographic"]
    
    st.markdown(f"""
    ### Key Findings
    - Highest accuracy: **{max_group}** ({max_acc:.2f}%)
    - Lowest accuracy: **{min_group}** ({min_acc:.2f}%)
    - Difference: **{acc_range:.2f}%**
    """)


def dataset_statistics_visualization(
    dataset_dir: str,
    title: str = "Dataset Statistics",
    width: int = 800,
    height: int = 400,
    key_prefix: str = ""
) -> Dict[str, Any]:
    """
    Display statistics about a dataset directory.
    
    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory
    title : str, optional
        Title for the visualization (default: "Dataset Statistics")
    width : int, optional
        Width of the chart in pixels (default: 800)
    height : int, optional
        Height of the chart in pixels (default: 400)
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with dataset statistics
    """
    if not os.path.exists(dataset_dir):
        st.warning(f"Dataset directory not found: {dataset_dir}")
        return {}
    
    # Collect statistics
    stats = {
        "total_files": 0,
        "by_subdirectory": {},
        "by_extension": {},
        "avg_filesize": 0,
        "total_size": 0,
        "empty_subdirectories": []
    }
    
    # Walk through the directory
    for root, dirs, files in os.walk(dataset_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        # Get subdirectory name relative to dataset_dir
        if root == dataset_dir:
            subdir = "root"
        else:
            subdir = os.path.relpath(root, dataset_dir)
        
        # Initialize subdirectory stats
        if subdir not in stats["by_subdirectory"]:
            stats["by_subdirectory"][subdir] = 0
        
        # Count files
        image_files = [
            f for f in files 
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        
        stats["by_subdirectory"][subdir] = len(image_files)
        stats["total_files"] += len(image_files)
        
        # Track extensions
        for file in image_files:
            ext = os.path.splitext(file)[1].lower()
            if ext not in stats["by_extension"]:
                stats["by_extension"][ext] = 0
            stats["by_extension"][ext] += 1
            
            # Track file sizes
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            stats["total_size"] += file_size
        
        # Track empty subdirectories
        if not files and not dirs:
            stats["empty_subdirectories"].append(subdir)
    
    # Calculate average file size
    if stats["total_files"] > 0:
        stats["avg_filesize"] = stats["total_size"] / stats["total_files"]
    
    # Create a DataFrame for the subdirectory counts
    subdir_data = pd.DataFrame({
        "Subdirectory": list(stats["by_subdirectory"].keys()),
        "Files": list(stats["by_subdirectory"].values())
    })
    
    # Sort by file count
    subdir_data = subdir_data.sort_values("Files", ascending=False).reset_index(drop=True)
    
    # Display statistics
    st.subheader(title)
    
    # Summary metrics
    columns = st.columns(4)
    columns[0].metric("Total Files", stats["total_files"])
    columns[1].metric("Categories", len(stats["by_subdirectory"]))
    columns[2].metric("Avg File Size", f"{stats['avg_filesize']/1024:.1f} KB")
    columns[3].metric("Total Size", f"{stats['total_size']/(1024*1024):.1f} MB")
    
    # Create the subdirectory bar chart
    if stats["total_files"] > 0:
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # Bar chart of subdirectory counts
        bars = ax.bar(subdir_data["Subdirectory"], subdir_data["Files"])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.1,
                str(int(height)),
                ha="center",
                va="bottom"
            )
        
        # Set labels and title
        ax.set_xlabel("Subdirectory")
        ax.set_ylabel("Number of Files")
        ax.set_title("Files by Subdirectory")
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45 if len(subdir_data) > 4 else 0)
        
        # Make the plot tight
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        
        # Display file extension breakdown in a pie chart
        if stats["by_extension"]:
            fig2, ax2 = plt.subplots(figsize=(width/100, height/100))
            
            # Extract extension data
            extensions = list(stats["by_extension"].keys())
            counts = list(stats["by_extension"].values())
            
            # Create pie chart
            wedges, texts, autotexts = ax2.pie(
                counts, 
                labels=extensions, 
                autopct="%1.1f%%",
                startangle=90,
                shadow=False
            )
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax2.axis("equal")
            
            # Make the labels more readable
            for text in texts:
                text.set_fontsize(12)
            
            for autotext in autotexts:
                autotext.set_fontsize(10)
                autotext.set_color("white")
            
            ax2.set_title("File Extensions")
            
            # Display the chart
            st.pyplot(fig2)
    
    # Display sample images
    if stats["total_files"] > 0:
        st.subheader("Sample Images")
        
        # Find up to 5 sample images from different subdirectories
        sample_images = []
        
        for subdir, count in stats["by_subdirectory"].items():
            if count > 0 and len(sample_images) < 5:
                # Get actual path
                if subdir == "root":
                    subdir_path = dataset_dir
                else:
                    subdir_path = os.path.join(dataset_dir, subdir)
                
                # Find first image in this subdirectory
                for file in os.listdir(subdir_path):
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        sample_images.append({
                            "path": os.path.join(subdir_path, file),
                            "name": file,
                            "subdir": subdir
                        })
                        break
        
        # Display images in a row
        if sample_images:
            cols = st.columns(min(5, len(sample_images)))
            
            for i, img_data in enumerate(sample_images):
                try:
                    image = Image.open(img_data["path"])
                    cols[i].image(
                        image, 
                        caption=f"{img_data['subdir']}/{img_data['name']}",
                        use_column_width=True
                    )
                except Exception as e:
                    cols[i].error(f"Error loading image: {str(e)}")
    
    return stats


def dataset_browser(
    dataset_dir: str,
    max_images: int = 24,
    key_prefix: str = ""
) -> None:
    """
    Interactive browser for exploring dataset images.
    
    Parameters
    ----------
    dataset_dir : str
        Path to the dataset directory
    max_images : int, optional
        Maximum number of images to display (default: 24)
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
    """
    if not os.path.exists(dataset_dir):
        st.warning(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Find all subdirectories
    subdirs = ["root"]
    for item in os.listdir(dataset_dir):
        item_path = os.path.join(dataset_dir, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            subdirs.append(item)
    
    # Allow filtering by subdirectory
    selected_subdir = st.selectbox(
        "Select Category",
        options=subdirs,
        key=f"{key_prefix}subdir_select"
    )
    
    # Get the actual directory path
    if selected_subdir == "root":
        subdir_path = dataset_dir
    else:
        subdir_path = os.path.join(dataset_dir, selected_subdir)
    
    # Get all image files in the selected subdirectory
    image_files = []
    for file in os.listdir(subdir_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_files.append(os.path.join(subdir_path, file))
    
    # Initialize session state for pagination
    if f"{key_prefix}browser_page" not in st.session_state:
        st.session_state[f"{key_prefix}browser_page"] = 0
    
    # Calculate total pages
    images_per_page = 12
    total_pages = (len(image_files) + images_per_page - 1) // images_per_page
    
    # Display image count
    st.write(f"Found {len(image_files)} images in {selected_subdir}")
    
    # Page navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("Previous Page", key=f"{key_prefix}browser_prev", disabled=st.session_state[f"{key_prefix}browser_page"] <= 0):
            st.session_state[f"{key_prefix}browser_page"] -= 1
            st.rerun()
    
    with col2:
        st.write(f"Page {st.session_state[f'{key_prefix}browser_page'] + 1} of {max(1, total_pages)}")
        
    with col3:
        if st.button("Next Page", key=f"{key_prefix}browser_next", disabled=st.session_state[f"{key_prefix}browser_page"] >= total_pages - 1):
            st.session_state[f"{key_prefix}browser_page"] += 1
            st.rerun()
    
    # Get images for current page
    start_idx = st.session_state[f"{key_prefix}browser_page"] * images_per_page
    end_idx = min(start_idx + images_per_page, len(image_files))
    current_page_files = image_files[start_idx:end_idx]
    
    # Display image grid
    if current_page_files:
        cols_per_row = 3
        rows = (len(current_page_files) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                
                if idx < len(current_page_files):
                    img_path = current_page_files[idx]
                    img_name = os.path.basename(img_path)
                    
                    try:
                        image = Image.open(img_path)
                        cols[col].image(image, caption=img_name, use_column_width=True)
                        
                        # Add image info in expander
                        with cols[col].expander("Image Info"):
                            # Get image size and dimensions
                            file_size = os.path.getsize(img_path) / 1024  # KB
                            width, height = image.size
                            
                            st.write(f"Dimensions: {width}x{height}")
                            st.write(f"Size: {file_size:.1f} KB")
                            st.write(f"Format: {image.format}")
                            
                            # Add view full size button
                            if st.button("View Full Size", key=f"{key_prefix}view_{idx}"):
                                st.session_state[f"{key_prefix}view_image"] = img_path
                                st.rerun()
                    
                    except Exception as e:
                        cols[col].error(f"Error loading {img_name}: {str(e)}")
    else:
        st.info("No images found in this directory.")
    
    # Full-size image view
    if f"{key_prefix}view_image" in st.session_state and st.session_state[f"{key_prefix}view_image"]:
        img_path = st.session_state[f"{key_prefix}view_image"]
        
        if os.path.exists(img_path):
            st.subheader(f"Full Size: {os.path.basename(img_path)}")
            
            try:
                image = Image.open(img_path)
                st.image(image, caption="Full Size View")
                
                # Add close button
                if st.button("Close Full View", key=f"{key_prefix}close_view"):
                    st.session_state[f"{key_prefix}view_image"] = None
                    st.rerun()
            
            except Exception as e:
                st.error(f"Error loading full-size image: {str(e)}")


def before_after_comparison(
    original_image: np.ndarray,
    processed_image: np.ndarray,
    title: str = "Before / After Comparison",
    key_prefix: str = ""
) -> None:
    """
    Display before/after comparison of an image processing operation.
    
    Parameters
    ----------
    original_image : np.ndarray
        Original image (BGR format)
    processed_image : np.ndarray
        Processed image (BGR format)
    title : str, optional
        Title for the comparison (default: "Before / After Comparison")
    key_prefix : str, optional
        Prefix for widget keys to avoid conflicts (default: "")
    """
    st.subheader(title)
    
    # Convert BGR to RGB for display
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original_image
        
    if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    else:
        processed_rgb = processed_image
    
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # Display original image
    col1.image(original_rgb, caption="Original Image", use_column_width=True)
    
    # Display processed image
    col2.image(processed_rgb, caption="Processed Image", use_column_width=True)
    
    # Add a slider for interactive comparison
    st.subheader("Interactive Comparison")
    st.write("Drag the slider to compare before and after")
    
    # Create a composite image that can be slid between before and after
    if original_rgb.shape == processed_rgb.shape:
        try:
            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Display the images
            ax.imshow(original_rgb)
            ax.set_title("Drag slider to compare")
            ax.set_axis_off()
            
            # Create a second axes for the processed image
            ax2 = ax.imshow(processed_rgb)
            
            # Add a slider to control the alpha blend
            blend = st.slider(
                "Comparison",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"{key_prefix}blend_slider"
            )
            
            # Apply the slider value to the alpha of the second image
            ax2.set_alpha(blend)
            
            # Display the blended image
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error creating interactive comparison: {str(e)}")
            
            # Fallback: Just show basic before/after again
            st.write("Interactive comparison not available. Using basic side-by-side comparison.")
            st.image([original_rgb, processed_rgb], caption=["Original", "Processed"], width=300)
    else:
        st.warning("Interactive comparison not available because images have different dimensions.")
        
        # Show basic side-by-side again with dimensions
        st.write(f"Original Image: {original_rgb.shape}")
        st.write(f"Processed Image: {processed_rgb.shape}")
