"""
Bias Testing Module

This module provides functionality for testing facial recognition accuracy
across different demographic groups to identify potential biases.
"""

import os
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class BiasAnalyzer:
    """A class to analyze bias in facial recognition systems."""

    def __init__(self, test_datasets_dir="./data/test_datasets"):
        """
        Initialize the bias analyzer.

        Args:
            test_datasets_dir (str): Directory containing test datasets
        """
        self.test_datasets_dir = test_datasets_dir
        self.results = {}
        self.ethnicity_colors = {
            'white': '#3498db',  # Blue
            'black': '#2ecc71',  # Green
            'asian': '#e74c3c',  # Red
            'indian': '#f39c12',  # Orange
            'others': '#9b59b6',  # Purple
            # Fallbacks for generic group names
            'group_a': '#3498db',
            'group_b': '#2ecc71',
            'group_c': '#e74c3c'
        }

    def load_test_dataset(self, dataset_name):
        """
        Load a test dataset from the specified directory.

        Args:
            dataset_name (str): Name of the dataset directory

        Returns:
            dict: Dictionary containing image paths and demographic information
        """
        dataset_path = os.path.join(self.test_datasets_dir, dataset_name)

        if not os.path.exists(dataset_path):
            print(f"Dataset directory not found: {dataset_path}")
            return None

        dataset = {"images": [], "demographics": []}

        # Walk through dataset directory
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Get demographic information from directory structure
                    # -- Assuming structure like: dataset_name/demographic_group/image.jpg --
                    demographic = os.path.basename(root)

                    image_path = os.path.join(root, file)
                    dataset["images"].append(image_path)
                    dataset["demographics"].append(demographic)

        print(f"Loaded {len(dataset['images'])} images from dataset '{dataset_name}'")
        return dataset

    def create_demographic_split_set(self):
        """
        Create a sample dataset structure for demographic bias testing.
        
        Args:
            use_utkface (bool): If True, suggests UTKFace dataset instead of generic groups
            
        Returns:
            str: Path to the created demographic directory
        """
        # Create base directory if it doesn't exist
        if not os.path.exists(self.test_datasets_dir):
            os.makedirs(self.test_datasets_dir)

        # Create demographic split set directory
        demographic_split_dir = os.path.join(self.test_datasets_dir, "demographic_split_set")
        if not os.path.exists(demographic_split_dir):
            os.makedirs(demographic_split_dir)

        # Create a results directory
        results_dir = os.path.join(self.test_datasets_dir, "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Create ethnicity-based directories for UTKFace
        groups = ["white", "black", "asian", "indian", "others"]

        # Create the group directories
        for group in groups:
            group_dir = os.path.join(demographic_split_dir, group)
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

        print(f"\nCreated demographic split set structure at {demographic_split_dir}")
        print("\nFor bias testing to work correctly:")
        print("1. Add test images with faces to each demographic group directory:")
        for group in groups:
            print(f"   - {os.path.join(demographic_split_dir, group)}")
        print("2. Each group should represent a different demographic category")



        print("3. Make sure faces are clearly visible in the images\n")

        # Suggest UTKFace dataset
        print("Tip: You can use the UTKFace dataset for ethical bias testing:")
        print("     Run: processor.download_and_extract_utkface_dataset()")
        print("     Then: processor.prepare_utkface_for_bias_testing()")

        return demographic_split_dir

    def test_recognition_accuracy(self, dataset_name):
        """
        Test face recognition accuracy across different demographic groups.

        Args:
            dataset_name (str): Name of the dataset to test

        Returns:
            dict: Results of the accuracy test by demographic group
        """
        dataset = self.load_test_dataset(dataset_name)

        if not dataset:
            print(f"Error: Could not load dataset '{dataset_name}'")
            return None

        if len(dataset["images"]) == 0:
            print(f"Error: Dataset '{dataset_name}' contains no images")
            print(f"Please add test images to each demographic group directory in: {os.path.join(self.test_datasets_dir, dataset_name)}")
            return None

        # Check if we have images for each demographic group
        demographics = set(dataset["demographics"])
        if len(demographics) == 0:
            print(f"Error: No valid demographic groups found in dataset '{dataset_name}'")
            return None

        print(f"Found {len(demographics)} demographic groups: {', '.join(demographics)}")

        results = {"overall": {"detected": 0, "total": 0}, "by_demographic": {}}

        # Initialize results for each demographic group
        for demographic in demographics:
            results["by_demographic"][demographic] = {"detected": 0, "total": 0}
            
        # Add progress tracking
        total_images = len(dataset["images"])
        print(f"\nProcessing {total_images} images...")
        
        # Process each image with progress bar
        import time
        start_time = time.time()
        
        for i, (image_path, demographic) in enumerate(zip(dataset["images"], dataset["demographics"])):
            # Update progress bar every image
            progress = (i + 1) / total_images
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Calculate time metrics
            elapsed_time = time.time() - start_time
            images_per_second = (i + 1) / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate remaining time
            if images_per_second > 0:
                remaining_images = total_images - (i + 1)
                eta_seconds = remaining_images / images_per_second
                eta_str = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta_str = "ETA: calculating..."
            
            # Print progress bar
            print(f"\r[{bar}] {(progress*100):5.1f}% | {i+1}/{total_images} | {images_per_second:.1f} img/s | {eta_str}", end='')
            
            try:
                # Load the image
                image = face_recognition.load_image_file(image_path)

                # Detect faces
                face_locations = face_recognition.face_locations(image)

                # Track results
                detected = len(face_locations) > 0

                results["overall"]["total"] += 1
                results["by_demographic"][demographic]["total"] += 1

                if detected:
                    results["overall"]["detected"] += 1
                    results["by_demographic"][demographic]["detected"] += 1

            except Exception as e:
                print(f"\nError processing {image_path}: {e}")

        # Print newline after progress bar completes
        print("\n")
        
        # Calculate accuracy metrics safely
        try:
            if results["overall"]["total"] > 0:
                results["overall"]["accuracy"] = (
                    results["overall"]["detected"] / results["overall"]["total"]
                )
            else:
                results["overall"]["accuracy"] = 0
                print("Warning: No valid images processed for overall accuracy calculation")

            for demographic, stats in results["by_demographic"].items():
                if stats["total"] > 0:
                    stats["accuracy"] = stats["detected"] / stats["total"]
                else:
                    stats["accuracy"] = 0
                    print(f"Warning: No valid images processed for demographic '{demographic}'")

            # Verify that we have valid accuracy values
            if all(stats["total"] == 0 for _, stats in results["by_demographic"].items()):
                print("Error: No valid images could be processed in any demographic group")
                return None
                
            # Print total processing time
            total_time = time.time() - start_time
            print(f"Total processing time: {int(total_time//60)} minutes {int(total_time%60)} seconds")
            print(f"Average processing speed: {total_images/total_time:.1f} images/second")

        except Exception as e:
            print(f"Error calculating accuracy metrics: {e}")
            return None

        self.results[dataset_name] = results
        return results

    def visualize_results(self, dataset_name=None):
        """
        Visualize the results of bias testing.

        Args:
            dataset_name (str, optional): Name of the dataset to visualize

        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        try:
            if not dataset_name:
                # Use the most recent result if none specified
                if not self.results:
                    print("No test results available to visualize.")
                    return None
                dataset_name = list(self.results.keys())[-1]

            if dataset_name not in self.results:
                print(f"No results found for dataset '{dataset_name}'.")
                return None

            results = self.results[dataset_name]

            # Check if we have demographic data with accuracy values
            if not results["by_demographic"]:
                print("No demographic data available for visualization.")
                return None

            # Extract demographic groups and their accuracy
            demographics = []
            # Only include groups that have data (more than 0 images)
            for demo, stats in results["by_demographic"].items():
                if stats["total"] > 0:
                    demographics.append(demo)
                    
            if not demographics:
                print("No demographic groups found with data for visualization.")
                return None

            # Check if we have accuracy metrics
            missing_accuracy = False
            for demo in demographics:
                if "accuracy" not in results["by_demographic"][demo]:
                    print(f"Warning: Missing accuracy data for demographic '{demo}'")
                    missing_accuracy = True

            if missing_accuracy:
                print("Cannot visualize results due to missing accuracy metrics.")
                return None

            # Calculate accuracies only for demographics that have data
            accuracies = [
                results["by_demographic"][demographic]["accuracy"] * 100
                for demographic in demographics
                if results["by_demographic"][demographic]["total"] > 0
            ]

            # Create the figure with enough space for legend
            plt.figure(figsize=(12, 8))
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get colors for each demographic
            colors = []
            for demo in demographics:
                demo_lower = demo.lower()
                if demo_lower in self.ethnicity_colors:
                    colors.append(self.ethnicity_colors[demo_lower])
                else:
                    colors.append('skyblue')  # Default color

            # Plot the bar chart with demographic-specific colors
            bars = ax.bar(demographics, accuracies, color=colors)

            # Add the overall accuracy line
            if "accuracy" in results["overall"]:
                overall_accuracy = results["overall"]["accuracy"] * 100
                ax.axhline(
                    y=overall_accuracy,
                    color="red",
                    linestyle="-"
                )

            # Add labels and title
            ax.set_xlabel("Demographic Group")
            ax.set_ylabel("Face Detection Accuracy as a %")
            ax.set_title(f"Face Detection Accuracy by Demographic Group - {dataset_name}")
            ax.set_ylim(0, 105)  # Set y-axis limit to 0-105%
            
            # Make more room for the legend by adjusting bottom margin
            plt.subplots_adjust(bottom=0.2)

            # Add data labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.2f}%",
                    ha="center",
                    va="bottom",
                )

            # Position legend below the chart with proper labels for each group
            # Include demographic name with its accuracy in the legend
            legend_labels = []
            legend_handles = []
            
            # First add the overall accuracy line
            if "accuracy" in results["overall"]:
                overall_accuracy = results["overall"]["accuracy"] * 100
                legend_labels.append(f"Overall: {overall_accuracy:.2f}%")
                legend_handles.append(plt.Line2D([0], [0], color="red", linewidth=2))
            
            # Add each demographic group with its color
            for i, demo in enumerate(demographics):
                if demo in results["by_demographic"] and results["by_demographic"][demo]["total"] > 0:
                    acc = results["by_demographic"][demo]["accuracy"] * 100
                    color = colors[i] if i < len(colors) else 'skyblue'
                    legend_labels.append(f"{demo}: {acc:.2f}%")
                    legend_handles.append(plt.Rectangle((0,0), 1, 1, color=color))
            
            # Create the legend
            ax.legend(legend_handles, legend_labels, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.15), ncol=3)
            
            # Add sample size information to bottom right corner
            sample_sizes_text = "Images used per group:\n"
            # Only show stats for groups that are actually in the chart
            for demo in demographics:
                if demo in results["by_demographic"] and results["by_demographic"][demo]["total"] > 0:
                    count = results["by_demographic"][demo]["total"]
                    sample_sizes_text += f"{demo}: {count}\n"
            
            # Add the text box with sample sizes
            plt.figtext(
                0.95, 0.05,  # x, y position (bottom right)
                sample_sizes_text,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            )
            
            plt.tight_layout()

            # Save the figure
            output_dir = os.path.join(self.test_datasets_dir, "results")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, f"{dataset_name}_bias_results.png")
            plt.savefig(output_path)

            print(f"Results visualization saved to {output_path}")
            return fig

        except Exception as e:
            print(f"Error visualizing results: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_bias_demonstration(self):
        """
        Run a complete bias testing demonstration.

        Returns:
            None
        """
        try:
            # Create demographic split set structure if needed
            demographic_split_path = os.path.join(self.test_datasets_dir, "demographic_split_set")
            if not os.path.exists(demographic_split_path):
                self.create_demographic_split_set()
                print("\nDemographic split set structure created at:")
                print(f"  {demographic_split_path}")
                print("\nPlease follow these steps before running bias testing again:")
                print("1. Run the following commands to prepare the dataset:")
                print("   processor = ImageProcessor()")
                print("   processor.download_and_extract_utkface_dataset()")
                print("   processor.prepare_utkface_for_bias_testing()")
                print("2. Then run the bias testing demonstration again\n")

                return

            # Using standard demographic groups
            groups = ["white", "black", "asian", "indian", "others"]

            # Check if the sample directories have any images
            has_images = False
            for group in groups:
                group_dir = os.path.join(demographic_split_path, group)
                if os.path.exists(group_dir):
                    image_files = [f for f in os.listdir(group_dir) 
                                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if image_files:
                        has_images = True
                        break

            if not has_images:
                print("\nNo test images found in the sample dataset directories.")
                print("\nPlease use the UTKFace dataset for testing:")
                print("   processor = ImageProcessor()")
                print("   processor.download_and_extract_utkface_dataset()")
                print("   processor.prepare_utkface_for_bias_testing()")
                return

            # Test recognition accuracy
            print("\nRunning recognition accuracy tests...")
            results = self.test_recognition_accuracy("demographic_split_set")

            if not results:
                print("\nBias testing failed. Please check the error messages above.")
                return

            # Print results summary
            print("\nBias Testing Results:")
            print(f"Overall Accuracy: {results['overall']['accuracy']*100:.2f}%")
            print("\nAccuracy by Demographic Group:")

            for demographic, stats in results["by_demographic"].items():
                print(f"  {demographic}: {stats['accuracy']*100:.2f}%")

            # Check for potential bias
            accuracies = [stats["accuracy"] for stats in results["by_demographic"].values()]
            if len(accuracies) >= 2:  # Need at least 2 groups to compare
                max_acc = max(accuracies)
                min_acc = min(accuracies)
                if max_acc - min_acc > 0.1:  # More than 10% difference
                    print(
                        "\nPotential bias detected: Significant accuracy difference between demographic groups."
                    )

                    # Get the groups with highest and lowest accuracy
                    max_group = None
                    min_group = None
                    for group, stats in results["by_demographic"].items():
                        if stats["accuracy"] == max_acc:
                            max_group = group
                        if stats["accuracy"] == min_acc:
                            min_group = group

                    if max_group and min_group:
                        print(f"  - Highest accuracy: {max_group} ({max_acc*100:.2f}%)")
                        print(f"  - Lowest accuracy: {min_group} ({min_acc*100:.2f}%)")
                        print(f"  - Difference: {(max_acc-min_acc)*100:.2f}%")
                else:
                    print("\nNo significant bias detected between demographic groups.")
            else:
                print("\nNot enough demographic groups to detect bias. Need at least 2 groups.")

            # Visualize the results
            print("\nGenerating visualization...")
            self.visualize_results("demographic_split_set")

            print("\nBias testing demonstration complete.")

        except Exception as e:
            print(f"\nBias Testing Results:\nAn error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def analyze_demographic_bias(self, dataset_name="demographic_split_set", detailed=False):
        """
        Perform a more detailed analysis of demographic bias in the dataset.
        
        Args:
            dataset_name (str): Name of the dataset to analyze
            detailed (bool): Whether to perform detailed statistical analysis
            
        Returns:
            dict: Results of bias analysis
        """
        # First run the standard accuracy test
        results = self.test_recognition_accuracy(dataset_name)
        if not results:
            return None

        # Extract demographic groups
        demographics = list(results["by_demographic"].keys())
        if len(demographics) < 2:
            print("Not enough demographic groups for bias analysis")
            return results

        # Calculate basic statistics
        accuracies = [results["by_demographic"][d]["accuracy"] for d in demographics]
        avg_accuracy = results["overall"]["accuracy"]
        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)
        accuracy_range = max_accuracy - min_accuracy

        # For detailed analysis, use standard deviation and variance
        if detailed:
            try:
                std_dev = np.std(accuracies)
                variance = np.var(accuracies)

                # Additional metrics
                mean_abs_deviation = np.mean([abs(acc - avg_accuracy) for acc in accuracies])

                # Add detailed stats to results
                results["bias_analysis"] = {
                    "std_deviation": std_dev,
                    "variance": variance,
                    "mean_abs_deviation": mean_abs_deviation,
                    "accuracy_range": accuracy_range,
                    "max_accuracy": max_accuracy,
                    "min_accuracy": min_accuracy
                }

                # Print detailed analysis
                print("\nDetailed Bias Analysis:")
                print(f"Standard Deviation: {std_dev:.4f}")
                print(f"Variance: {variance:.4f}")
                print(f"Mean Absolute Deviation: {mean_abs_deviation:.4f}")
                print(f"Accuracy Range: {accuracy_range:.4f}")

                # Interpretation guide
                if accuracy_range > 0.15:
                    bias_level = "High"
                elif accuracy_range > 0.05:
                    bias_level = "Moderate"
                else:
                    bias_level = "Low"

                print(f"\nBias Level: {bias_level}")
                print(f"  - Range > 0.15: High bias potential")
                print(f"  - Range 0.05-0.15: Moderate bias potential")
                print(f"  - Range < 0.05: Low bias potential")

            except Exception as e:
                print(f"Error in detailed analysis: {e}")
        else:
            # Basic analysis
            results["bias_analysis"] = {
                "accuracy_range": accuracy_range,
                "max_accuracy": max_accuracy,
                "min_accuracy": min_accuracy
            }

        return results

if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()
