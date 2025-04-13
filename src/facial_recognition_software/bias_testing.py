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

    def __init__(self, test_datasets_dir="./data/test_datasets"):
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

    def create_demographic_split_set(self):
        """
        Create a sample dataset structure for demonstration purposes.
        This is a placeholder for actual diverse datasets.
        """
        # Create base directory if it doesn't exist (for safety)
        if not os.path.exists(self.test_datasets_dir):
            os.makedirs(self.test_datasets_dir)

        # Create demographic split set directory (for safety)
        demographic_split_dir = os.path.join(self.test_datasets_dir, "demographic_split_set")
        if not os.path.exists(demographic_split_dir):
            os.makedirs(demographic_split_dir)

        # Create demographic group directories
        groups = ["group_a", "group_b", "group_c"]
        for group in groups:
            group_dir = os.path.join(demographic_split_dir, group)
            if not os.path.exists(group_dir):
                os.makedirs(group_dir)

        # Create a results directory
        results_dir = os.path.join(self.test_datasets_dir, "results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        print(f"\nCreated demographic split set structure at {demographic_split_dir}")
        print("\nFor bias testing to work correctly:")
        print("1. Add test images with faces to each demographic group directory:")
        for group in groups:
            print(f"   - {os.path.join(demographic_split_dir, group)}")
        print("2. Each group should represent a different demographic category")
        print("   (e.g., group_a = asian, group_b = african, group_c = european)")
        print("3. Make sure faces are clearly visible in the images\n")

        # Try to find images from dataset setup in LFW directory
        try:
            lfw_path = os.path.join(os.path.dirname(self.test_datasets_dir), "datasets", "lfw", "lfw")
            if os.path.exists(lfw_path):
                print(f"Tip: You can copy sample images from the LFW dataset at:")
                print(f"  {lfw_path}")
                print("  Run option 5 (Dataset Setup & Management) to download this dataset if needed.\n")
        except Exception:
            pass

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
            demographics = list(results["by_demographic"].keys())
            if not demographics:
                print("No demographic groups found for visualization.")
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

            accuracies = [
                results["by_demographic"][demographic]["accuracy"] * 100
                for demographic in demographics
            ]

            # Create the figure
            plt.figure(figsize=(10, 6))
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the bar chart
            bars = ax.bar(demographics, accuracies, color="skyblue")

            # Add the overall accuracy line
            if "accuracy" in results["overall"]:
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
                print("1. Add test images to each demographic group directory:")
                for group in ["group_a", "group_b", "group_c"]:
                    print(f"   - {os.path.join(demographic_split_path, group)}")
                print("2. Images should contain faces from different demographic groups")
                print("3. Then run the bias testing demonstration again\n")
                return

            # Check if the sample directories have any images
            has_images = False
            for group in ["group_a", "group_b", "group_c"]:
                group_dir = os.path.join(demographic_split_path, group)
                if os.path.exists(group_dir):
                    image_files = [f for f in os.listdir(group_dir) 
                                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if image_files:
                        has_images = True
                        break

            if not has_images:
                print("\nNo test images found in the sample dataset directories.")

                # Try to copy sample images from LFW dataset if it exists
                lfw_path = os.path.join(os.path.dirname(self.test_datasets_dir), "datasets", "lfw", "lfw")
                if os.path.exists(lfw_path):
                    print("\nFound LFW dataset. Attempting to create sample test data...")
                    success = self.copy_sample_images_from_lfw()
                    if success:
                        print("Successfully created sample test data. Continuing with bias testing...")
                    else:
                        print("\nCould not automatically create sample test data.")
                        print("Please add test images manually to at least one demographic group directory:")
                        for group in ["group_a", "group_b", "group_c"]:
                            print(f"  - {os.path.join(demographic_split_path, group)}")
                        return
                else:
                    print("Please add test images to at least one demographic group directory:")
                    for group in ["group_a", "group_b", "group_c"]:
                        print(f"  - {os.path.join(demographic_split_path, group)}")
                    print("\nTip: You can run option 5 (Dataset Setup & Management) to download")
                    print("     the LFW dataset, which can be used for bias testing.")
                    return

            # Test recognition accuracy
            print("\nRunning recognition accuracy tests...")
            results = self.test_recognition_accuracy("demographic_split_set")

            if not results:
                print("\nBias testing failed. Please check the error messages above.")
                return

            # Print results summary
            print("\nBias Testing Results:")
            print(f"Overall Accuracy: {results['overall']['accuracy']*100:.1f}%")
            print("\nAccuracy by Demographic Group:")

            for demographic, stats in results["by_demographic"].items():
                print(f"  {demographic}: {stats['accuracy']*100:.1f}%")

            # Check for potential bias
            accuracies = [stats["accuracy"] for stats in results["by_demographic"].values()]
            if len(accuracies) >= 2:  # Need at least 2 groups to compare
                if max(accuracies) - min(accuracies) > 0.1:  # More than 10% difference
                    print(
                        "\nPotential bias detected: Significant accuracy difference between demographic groups."
                    )
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

    def copy_sample_images_from_lfw(self):
        """
        Try to copy sample images from LFW dataset to the test dataset groups.
        This is a convenience function for quick demo setup.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import random
            import shutil

            # Find the LFW dataset
            lfw_path = os.path.join(os.path.dirname(self.test_datasets_dir), "datasets", "lfw", "lfw")
            if not os.path.exists(lfw_path):
                print(f"LFW dataset not found at: {lfw_path}")
                print("Run option 5 (Dataset Setup & Management) to download this dataset first.")
                return False

            # Get list of person directories with multiple images
            person_dirs = []
            for person in os.listdir(lfw_path):
                person_dir = os.path.join(lfw_path, person)
                if os.path.isdir(person_dir):
                    images = [f for f in os.listdir(person_dir) 
                             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                    if len(images) >= 2:
                        person_dirs.append((person, person_dir, images))

            if not person_dirs:
                print("No suitable person directories found in LFW dataset.")
                return False

            # Sample up to 10 people for each demographic group
            if len(person_dirs) < 3:
                print("Not enough people in LFW dataset to create sample groups.")
                return False

            # Shuffle the person directories
            random.shuffle(person_dirs)

            # Create demographic split set directory
            demographic_split_dir = os.path.join(self.test_datasets_dir, "demographic_split_set")
            if not os.path.exists(demographic_split_dir):
                os.makedirs(demographic_split_dir)

            # Copy images to each demographic group
            groups = ["group_a", "group_b", "group_c"]
            people_per_group = min(5, len(person_dirs) // 3)

            for i, group in enumerate(groups):
                group_dir = os.path.join(demographic_split_dir, group)
                if not os.path.exists(group_dir):
                    os.makedirs(group_dir)

                # Get people for this group
                start_idx = i * people_per_group
                end_idx = start_idx + people_per_group
                group_people = person_dirs[start_idx:end_idx]

                # Copy one image from each person to this group
                for person, person_dir, images in group_people:
                    # Choose a random image
                    image = random.choice(images)
                    src_path = os.path.join(person_dir, image)
                    dst_path = os.path.join(group_dir, f"{person}_{image}")
                    shutil.copy2(src_path, dst_path)

            print(f"\nCopied sample images from LFW dataset to {demographic_split_dir}")
            print(f"- Added {people_per_group} people to each demographic group")
            print("Note: These groups don't represent real demographic differences.")
            print("      For real bias testing, use appropriately labeled datasets.\n")
            return True

        except Exception as e:
            print(f"Error copying sample images: {e}")
            return False

if __name__ == "__main__":
    # Run a simple test if this module is executed directly
    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()
