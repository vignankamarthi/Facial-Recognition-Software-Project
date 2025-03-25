"""
Bias Testing Module

This module provides functionality for testing facial recognition accuracy
across different demographic groups to identify potential biases.
"""

import os
import face_recognition
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class BiasAnalyzer:
    """A class to analyze bias in facial recognition systems."""

    def __init__(self, test_datasets_dir="../data/test_datasets"):
        """
        Initialize the bias analyzer.

        Args:
            test_datasets_dir (str): Directory containing test datasets
        """
        self.test_datasets_dir = test_datasets_dir
        self.results = {}

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
                    # --**Assuming structure like: dataset_name/demographic_group/image.jpg**--
                    demographic = os.path.basename(root)

                    image_path = os.path.join(root, file)
                    dataset["images"].append(image_path)
                    dataset["demographics"].append(demographic)

        print(f"Loaded {len(dataset['images'])} images from dataset '{dataset_name}'")
        return dataset

    def create_sample_dataset(self):
        """
        Create a sample dataset structure for demonstration purposes.
        This is a placeholder for actual diverse datasets.
        """
        # Create base directory if it doesn't exist (for safety)
        if not os.path.exists(self.test_datasets_dir):
            os.makedirs(self.test_datasets_dir)

        # Create sample dataset directory (for safety)
        sample_dataset_dir = os.path.join(self.test_datasets_dir, "sample_dataset")
        if not os.path.exists(sample_dataset_dir):
            os.makedirs(sample_dataset_dir)

        # Create demographic group directories
        groups = ["group_a", "group_b", "group_c"]
        for group in groups:
            group_dir = os.path.join(sample_dataset_dir, group)
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

        print(f"Created sample dataset structure at {sample_dataset_dir}")
        print("Please add test images to each demographic group directory.")

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
            return None

        results = {"overall": {"detected": 0, "total": 0}, "by_demographic": {}}

        # Initialize results for each demographic group
        for demographic in set(dataset["demographics"]):
            results["by_demographic"][demographic] = {"detected": 0, "total": 0}

        # Process each image
        for image_path, demographic in zip(dataset["images"], dataset["demographics"]):
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
                print(f"Error processing {image_path}: {e}")

        # Calculate accuracy metrics
        if results["overall"]["total"] > 0:
            results["overall"]["accuracy"] = (
                results["overall"]["detected"] / results["overall"]["total"]
            )

        for demographic, stats in results["by_demographic"].items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["detected"] / stats["total"]

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

        # Extract demographic groups and their accuracy
        demographics = list(results["by_demographic"].keys())
        accuracies = [
            results["by_demographic"][demo]["accuracy"] * 100 for demo in demographics
        ]

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bar chart
        bars = ax.bar(demographics, accuracies, color="skyblue")

        # Add the overall accuracy line
        overall_accuracy = results["overall"]["accuracy"] * 100
        ax.axhline(
            y=overall_accuracy,
            color="red",
            linestyle="-",
            label=f"Overall: {overall_accuracy:.1f}%",
        )

        # Add labels and title
        ax.set_xlabel("Demographic Group")
        ax.set_ylabel("Face Detection Accuracy (%)")
        ax.set_title(f"Face Detection Accuracy by Demographic Group - {dataset_name}")
        ax.set_ylim(0, 105)  # Set y-axis limit to 0-105%

        # Add data labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        ax.legend()
        plt.tight_layout()

        # Save the figure
        output_dir = os.path.join(self.test_datasets_dir, "results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, f"{dataset_name}_bias_results.png")
        plt.savefig(output_path)

        print(f"Results visualization saved to {output_path}")
        return fig

    def run_bias_demonstration(self):
        """
        Run a complete bias testing demonstration.

        Returns:
            None
        """
        # Create sample dataset structure if needed
        if not os.path.exists(os.path.join(self.test_datasets_dir, "sample_dataset")):
            self.create_sample_dataset()
            print(
                "Please add test images to the sample dataset directories, then run again."
            )
            return

        # Test recognition accuracy
        results = self.test_recognition_accuracy("sample_dataset")

        if not results:
            return

        # Print results summary
        print("\nBias Testing Results:")
        print(f"Overall Accuracy: {results['overall']['accuracy']*100:.1f}%")
        print("\nAccuracy by Demographic Group:")

        for demographic, stats in results["by_demographic"].items():
            print(f"  {demographic}: {stats['accuracy']*100:.1f}%")

        # Check for potential bias
        accuracies = [stats["accuracy"] for stats in results["by_demographic"].values()]
        if max(accuracies) - min(accuracies) > 0.1:  # More than 10% difference
            print(
                "\nPotential bias detected: Significant accuracy difference between demographic groups."
            )

        # Visualize the results
        self.visualize_results("sample_dataset")

        print("\nBias testing demonstration complete.")


if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()
