"""
Image Processing Module

This module provides functionality for loading and processing static images
for face detection, matching, and anonymization.
"""

import os
import cv2
import shutil
import zipfile
import re
import urllib.request
import random
import numpy as np
from facial_recognition_software.face_detection import FaceDetector
from facial_recognition_software.face_matching import FaceMatcher
from facial_recognition_software.anonymization import FaceAnonymizer


class ImageProcessor:
    """A class to handle static image processing operations."""

    def __init__(self, known_faces_dir="./data/known_faces"):
        """
        Initialize the image processor.

        Args:
            known_faces_dir (str): Directory containing known face images
        """
        self.detector = FaceDetector()
        self.matcher = FaceMatcher(known_faces_dir)
        self.anonymizer = FaceAnonymizer()

    def load_image(self, image_path):
        """
        Load an image from the specified path.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Loaded image or None if failed
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return None

        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image: {image_path}")
                return None

            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def process_image(self, image, detect=True, match=False, anonymize=False):
        """
        Process an image with selected operations.

        Args:
            image (numpy.ndarray): Image to process
            detect (bool): Whether to detect faces
            match (bool): Whether to match faces against known faces
            anonymize (bool): Whether to anonymize faces

        Returns:
            tuple: (processed_image, results_dict)
        """
        if image is None:
            return None, {}

        results = {"face_count": 0, "face_locations": [], "identified_faces": []}

        # Create a copy of the image to display results
        display_image = image.copy()

        # Always detect faces first (required for other operations)
        face_locations, face_encodings = self.detector.detect_faces(image)
        results["face_count"] = len(face_locations)
        results["face_locations"] = face_locations

        if detect and not match and not anonymize:
            # Just draw boxes around faces
            display_image = self.detector.draw_face_boxes(image, face_locations)

        if match:
            # Identify the faces
            display_image, face_names = self.matcher.identify_faces(
                image, face_locations, face_encodings
            )
            results["identified_faces"] = face_names

        if anonymize:
            # Anonymize the faces
            display_image = self.anonymizer.anonymize_frame(image, face_locations)

        return display_image, results

    def process_image_file(
        self,
        image_path,
        detect=True,
        match=False,
        anonymize=False,
        save_result=False,
        output_dir=None,
    ):
        """
        Process an image file with selected operations.

        Args:
            image_path (str): Path to the image file
            detect (bool): Whether to detect faces
            match (bool): Whether to match faces against known faces
            anonymize (bool): Whether to anonymize faces
            save_result (bool): Whether to save the processed image
            output_dir (str): Directory to save results (if save_result is True)

        Returns:
            tuple: (processed_image, results_dict)
        """
        # Load the image
        image = self.load_image(image_path)
        if image is None:
            return None, {}

        # Process the image
        processed_image, results = self.process_image(image, detect, match, anonymize)

        # Add the image path to the results
        results["image_path"] = image_path

        # Save the processed image if requested
        if save_result and processed_image is not None:
            if output_dir is None:
                # Default to a 'results' directory in the same location as the input
                parent_dir = os.path.dirname(image_path)
                output_dir = os.path.join(parent_dir, "results")

            # Create the output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Generate output filename
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # Add suffixes based on operations
            suffix = "_"
            if detect:
                suffix += "detected"
            if match:
                suffix += "_matched"
            if anonymize:
                suffix += "_anonymized"

            output_path = os.path.join(output_dir, f"{name}{suffix}{ext}")

            # Check if file already exists to avoid overwriting
            if os.path.exists(output_path):
                # Add timestamp to filename to make it unique
                import time

                timestamp = int(time.time())
                output_path = os.path.join(
                    output_dir, f"{name}{suffix}_{timestamp}{ext}"
                )

            # Save the image
            cv2.imwrite(output_path, processed_image)
            results["output_path"] = output_path
            print(f"Saved processed image to: {output_path}")

        return processed_image, results

    def process_directory(
        self,
        directory_path,
        detect=True,
        match=False,
        anonymize=False,
        save_results=False,
        output_dir=None,
        display_results=True,
    ):
        """
        Process all images in a directory.

        Args:
            directory_path (str): Path to the directory containing images
            detect (bool): Whether to detect faces
            match (bool): Whether to match faces against known faces
            anonymize (bool): Whether to anonymize faces
            save_results (bool): Whether to save the processed images
            output_dir (str): Directory to save results (if save_results is True)
            display_results (bool): Whether to display results in windows

        Returns:
            dict: Results for all processed images
        """
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            print(f"Error: Directory not found: {directory_path}")
            return {}

        # Initialize results dictionary
        all_results = {}

        # Create output directory if needed
        if save_results and output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process each image file in the directory
        for filename in os.listdir(directory_path):
            # Check if the file is an image
            if not any(
                filename.lower().endswith(ext)
                for ext in [".jpg", ".jpeg", ".png", ".bmp"]
            ):
                continue

            image_path = os.path.join(directory_path, filename)

            # Process the image
            processed_image, results = self.process_image_file(
                image_path, detect, match, anonymize, save_results, output_dir
            )

            # Add to results dictionary
            all_results[image_path] = results

            # Display the results if requested
            if display_results and processed_image is not None:
                # Resize large images for display
                h, w = processed_image.shape[:2]
                max_dim = 800
                if max(h, w) > max_dim:
                    # Calculate new dimensions
                    if h > w:
                        new_h = max_dim
                        new_w = int(w * (max_dim / h))
                    else:
                        new_w = max_dim
                        new_h = int(h * (max_dim / w))

                    # Resize the image for display
                    display_img = cv2.resize(processed_image, (new_w, new_h))
                else:
                    display_img = processed_image

                # Show the image
                window_name = os.path.basename(image_path)
                cv2.imshow(window_name, display_img)
                cv2.waitKey(0)  # Wait for key press

        # Close all windows if displayed
        if display_results:
            cv2.destroyAllWindows()

        # Print summary
        print(f"\nProcessed {len(all_results)} images in {directory_path}")
        total_faces = sum(
            result.get("face_count", 0) for result in all_results.values()
        )
        print(f"Total faces detected: {total_faces}")

        return all_results

    def download_and_extract_utkface_dataset(
        self,
        target_dir="./data/datasets/utkface",
        sample_size=500,
        specific_ethnicities=None,
    ):
        """
        Download and extract a sample of the UTKFace dataset.

        Args:
            target_dir (str): Directory to save the dataset
            sample_size (int): Number of total images to include (default: 500)
            specific_ethnicities (list, optional): List of ethnicity codes to include (default: "all" for all ethnicities)
                0: White, 1: Black, 2: Asian, 3: Indian, 4: Others

        Returns:
            bool: True if successful, False otherwise
        """
        print("Starting UTKFace dataset download...")

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # URL for the UTKFace dataset aligned & cropped version
        utkface_url = "https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk"
        zip_file = os.path.join(target_dir, "utkface.zip")
        extract_dir = os.path.join(target_dir, "utkface_aligned")

        # Dictionary to map race codes to ethnicity names
        ethnicity_names = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

        try:
            # Check if we need to download
            if os.path.exists(extract_dir) and os.path.isdir(extract_dir):
                files = [
                    f
                    for f in os.listdir(extract_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if (
                    len(files) > 200
                ):  # Assume we already have the dataset if many files exist
                    print(
                        f"UTKFace dataset seems to be already extracted at {extract_dir}"
                    )
                    extract_complete = True
                else:
                    extract_complete = False
            else:
                extract_complete = False

            # Download if needed
            if not extract_complete:
                if (
                    os.path.exists(zip_file) and os.path.getsize(zip_file) > 100000000
                ):  # >100MB!!
                    print(f"UTKFace zip file already downloaded at {zip_file}")
                else:
                    print(f"Downloading UTKFace dataset...")
                    print("This may take several minutes. Please be patient.")

                    # Direct download can be problematic for Google Drive
                    # Instead, we'll use gdown if available, or suggest manual download
                    try:
                        import gdown

                        gdown.download(utkface_url, zip_file, quiet=False)
                    except ImportError:
                        print("Note: The 'gdown' package is not installed.")
                        print(
                            "For automatic download from Google Drive, install it with:"
                        )
                        print("  pip install gdown")
                        print("\nAlternatively, you can download manually from:")
                        print("  https://susanqq.github.io/UTKFace/")
                        print(f"And place the zip file at: {zip_file}")

                        # Ask user if they want to proceed with manual download
                        response = input(
                            "Would you like to try downloading without gdown? (y/n): "
                        )
                        if response.lower() == "y":
                            print(
                                "Attempting direct download (this may fail with Google Drive)..."
                            )
                            urllib.request.urlretrieve(utkface_url, zip_file)
                        else:
                            print("Download skipped. Please download manually.")
                            return False

                # Extract if zip file exists
                if os.path.exists(zip_file):
                    # Create or clear the extraction directory
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir)
                    os.makedirs(extract_dir)

                    print("Extracting dataset...")
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print("Extraction complete!")
                else:
                    print("Zip file not found. Download may have failed.")
                    return False

            # Create demographic directories
            demographics_dir = os.path.join(target_dir, "demographic_split")
            if not os.path.exists(demographics_dir):
                os.makedirs(demographics_dir)

            # Initialize or clear the ethnicity directories
            for ethnicity_code, ethnicity_name in ethnicity_names.items():
                # Skip unwanted ethnicities if specified
                if (
                    specific_ethnicities is not None
                    and ethnicity_code not in specific_ethnicities
                ):
                    continue

                ethnicity_dir = os.path.join(demographics_dir, ethnicity_name.lower())
                if not os.path.exists(ethnicity_dir):
                    os.makedirs(ethnicity_dir)
                else:
                    # Clear existing files to avoid duplicates
                    for file in os.listdir(ethnicity_dir):
                        os.remove(os.path.join(ethnicity_dir, file))

            # Process and organize images
            files = [
                f
                for f in os.listdir(extract_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Filter for valid files with proper naming
            valid_files = []
            for file in files:
                try:
                    # UTKFace format: [age]_[gender]_[race]_[date&time].jpg
                    # We need to extract race information
                    parts = file.split("_")
                    if len(parts) >= 3:
                        age = int(parts[0])
                        gender = int(parts[1])
                        race = int(parts[2])
                        if race in ethnicity_names:
                            # Skip unwanted ethnicities if specified
                            if (
                                specific_ethnicities is not None
                                and race not in specific_ethnicities
                            ):
                                continue
                            valid_files.append((file, race))
                except (ValueError, IndexError):
                    # Skip files that don't match the expected format
                    continue

            # Check if we have enough files
            if len(valid_files) == 0:
                print("Error: No valid UTKFace images found.")
                return False

            # Random sample if we have more files than requested
            if len(valid_files) > sample_size:
                valid_files = random.sample(valid_files, sample_size)

            # Copy files to demographic directories
            print(f"Organizing {len(valid_files)} images by ethnicity...")
            ethnicity_counts = {code: 0 for code in ethnicity_names}

            for file, race in valid_files:
                src_path = os.path.join(extract_dir, file)
                ethnicity_name = ethnicity_names[race].lower()
                ethnicity_dir = os.path.join(demographics_dir, ethnicity_name)
                dst_path = os.path.join(ethnicity_dir, file)

                shutil.copy2(src_path, dst_path)
                ethnicity_counts[race] += 1

            # Report what was done
            print("\nUTKFace dataset processing complete.")
            print(f"Total images organized: {len(valid_files)}")
            print("\nImages by ethnicity:")
            for code, name in ethnicity_names.items():
                if code in ethnicity_counts:
                    print(f"  - {name}: {ethnicity_counts[code]} images")

            print(f"\nDemographic split dataset created at {demographics_dir}")
            print("This dataset can now be used for bias testing.")

            return True

        except Exception as e:
            print(f"Error downloading or processing UTKFace dataset: {e}")
            import traceback

            traceback.print_exc()
            return False

    def prepare_utkface_for_bias_testing(
        self,
        utkface_dir="./data/datasets/utkface/demographic_split",
        test_datasets_dir="./data/test_datasets/demographic_split_set",
        images_per_ethnicity=25,
    ):
        """
        Prepare UTKFace dataset for bias testing by copying images to the test directory.

        Args:
            utkface_dir (str): Source directory with ethnicity-separated UTKFace images
            test_datasets_dir (str): Target directory for bias testing
            images_per_ethnicity (int): Maximum number of images per ethnicity group

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if source directory exists
            if not os.path.exists(utkface_dir):
                print(
                    f"Error: UTKFace demographic split directory not found at {utkface_dir}"
                )
                print("Run download_and_extract_utkface_dataset() first.")
                return False

            # Create or clear the test datasets directory
            if not os.path.exists(test_datasets_dir):
                os.makedirs(test_datasets_dir)

            # Get list of ethnicity directories
            ethnicity_dirs = [
                d
                for d in os.listdir(utkface_dir)
                if os.path.isdir(os.path.join(utkface_dir, d))
            ]

            if not ethnicity_dirs:
                print("Error: No ethnicity directories found in the UTKFace dataset.")
                return False

            print(
                f"Found {len(ethnicity_dirs)} ethnicity groups: {', '.join(ethnicity_dirs)}"
            )

            # Process each ethnicity
            for ethnicity in ethnicity_dirs:
                src_dir = os.path.join(utkface_dir, ethnicity)
                dst_dir = os.path.join(test_datasets_dir, ethnicity)

                # Create destination directory
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                else:
                    # Clear existing files
                    for file in os.listdir(dst_dir):
                        file_path = os.path.join(dst_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                # Get all image files
                image_files = [
                    f
                    for f in os.listdir(src_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                # Sample if we have more than requested
                if len(image_files) > images_per_ethnicity:
                    selected_files = random.sample(image_files, images_per_ethnicity)
                else:
                    selected_files = image_files

                # Copy files
                for file in selected_files:
                    src_path = os.path.join(src_dir, file)
                    dst_path = os.path.join(dst_dir, file)
                    shutil.copy2(src_path, dst_path)

                print(f"  - {ethnicity}: {len(selected_files)} images copied")

            # Create a results directory
            results_dir = os.path.join(os.path.dirname(test_datasets_dir), "results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            print(f"\nBias testing dataset prepared at {test_datasets_dir}")
            print(f"Use this dataset with the BiasAnalyzer class.")
            return True

        except Exception as e:
            print(f"Error preparing UTKFace for bias testing: {e}")
            import traceback

            traceback.print_exc()
            return False

    def prepare_known_faces_from_utkface(
        self,
        num_people=20,
        ethnicity_balanced=True,
        utkface_dir="./data/datasets/utkface/utkface_aligned",
        output_dir="./data/known_faces",
    ):
        """
        Prepare a set of known faces from the UTKFace dataset.

        Args:
            num_people (int): Number of people to include (default: 20)
            ethnicity_balanced (bool): Whether to balance selection across ethnicities
            utkface_dir (str): Directory containing the UTKFace dataset
            output_dir (str): Directory to save the known faces

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(utkface_dir):
            print(f"Error: UTKFace dataset not found at {utkface_dir}")
            return False

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Dictionary to map race codes to ethnicity names
            ethnicity_names = {
                0: "White",
                1: "Black",
                2: "Asian",
                3: "Indian",
                4: "Others",
            }

            # Get all image files
            files = [
                f
                for f in os.listdir(utkface_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Group by ethnicity if balanced selection is requested
            if ethnicity_balanced:
                ethnicity_files = {code: [] for code in ethnicity_names}

                # Sort files by ethnicity
                for file in files:
                    try:
                        parts = file.split("_")
                        if len(parts) >= 3:
                            race = int(parts[2])
                            if race in ethnicity_names:
                                ethnicity_files[race].append(file)
                    except (ValueError, IndexError):
                        continue

                # Calculate how many from each ethnicity
                num_ethnicities = len(
                    [code for code, files in ethnicity_files.items() if files]
                )
                per_ethnicity = max(1, num_people // num_ethnicities)

                # Select files from each ethnicity
                selected_files = []
                for race, eth_files in ethnicity_files.items():
                    if eth_files:
                        # Select up to per_ethnicity files from this ethnicity
                        to_select = min(per_ethnicity, len(eth_files))
                        ethnicity_selection = random.sample(eth_files, to_select)
                        selected_files.extend(
                            [
                                (file, f"{ethnicity_names[race]}_{i}")
                                for i, file in enumerate(ethnicity_selection)
                            ]
                        )

                # If we need more files, add extras
                if len(selected_files) < num_people:
                    all_remaining = []
                    for race, eth_files in ethnicity_files.items():
                        # Get files we didn't already select
                        already_selected = [
                            f[0] for f in selected_files if f[0] in eth_files
                        ]
                        remaining = [f for f in eth_files if f not in already_selected]
                        all_remaining.extend(
                            [
                                (file, f"{ethnicity_names[race]}_extra_{i}")
                                for i, file in enumerate(remaining)
                            ]
                        )

                    # Select additional files if needed
                    additional_needed = num_people - len(selected_files)
                    if all_remaining and additional_needed > 0:
                        additional = random.sample(
                            all_remaining, min(additional_needed, len(all_remaining))
                        )
                        selected_files.extend(additional)
            else:
                # Simple random selection without ethnicity balancing
                if len(files) <= num_people:
                    selected_files = [
                        (file, f"Person_{i}") for i, file in enumerate(files)
                    ]
                else:
                    selected_files = [
                        (file, f"Person_{i}")
                        for i, file in enumerate(random.sample(files, num_people))
                    ]

            # Copy selected files to output directory
            processed_count = 0
            for file, person_name in selected_files:
                src_path = os.path.join(utkface_dir, file)
                dst_path = os.path.join(output_dir, f"{person_name}.jpg")

                # Avoid duplicate filenames
                if os.path.exists(dst_path):
                    dst_path = os.path.join(
                        output_dir, f"{person_name}_{processed_count}.jpg"
                    )

                shutil.copy2(src_path, dst_path)
                processed_count += 1

            print(f"\nPrepared {processed_count} known faces in {output_dir}")
            if ethnicity_balanced:
                print("Selected faces were balanced across ethnicities.")
            return True

        except Exception as e:
            print(f"Error preparing known faces from UTKFace: {e}")
            import traceback

            traceback.print_exc()
            return False

    def prepare_test_dataset_from_utkface(
        self,
        num_known=5,
        num_unknown=5,
        utkface_dir="./data/datasets/utkface/utkface_aligned",
        known_faces_dir="./data/known_faces",
        output_dir="./data/test_images",
    ):
        """
        Prepare a test dataset from the UTKFace dataset.

        Args:
            num_known (int): Number of known people to include
            num_unknown (int): Number of unknown people to include
            utkface_dir (str): Directory containing the UTKFace dataset
            known_faces_dir (str): Directory containing known faces
            output_dir (str): Directory to save test images

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(utkface_dir):
            print(f"Error: UTKFace dataset not found at {utkface_dir}")
            return False

        if not os.path.exists(known_faces_dir):
            print(f"Error: Known faces directory not found at {known_faces_dir}")
            return False

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create subdirectories for known and unknown
        known_output_dir = os.path.join(output_dir, "known")
        unknown_output_dir = os.path.join(output_dir, "unknown")

        if not os.path.exists(known_output_dir):
            os.makedirs(known_output_dir)

        if not os.path.exists(unknown_output_dir):
            os.makedirs(unknown_output_dir)

        try:
            # Dictionary to map race codes to ethnicity names
            ethnicity_names = {
                0: "White",
                1: "Black",
                2: "Asian",
                3: "Indian",
                4: "Others",
            }

            # Get all image files from UTKFace
            utk_files = [
                f
                for f in os.listdir(utkface_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Extract identities from known faces
            known_identities = []
            for filename in os.listdir(known_faces_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Extract identity from filename
                    identity = os.path.splitext(filename)[0]
                    # Remove any index suffixes like _1, _2, etc.
                    if "_" in identity and identity.split("_")[-1].isdigit():
                        identity = "_".join(identity.split("_")[:-1])
                    known_identities.append(identity)

            # For UTKFace, we don't have identity information, so we'll use gender+ethnicity+age
            # as a proxy for "similar looking people"

            # Prepare "known" test images (different images of known identities)
            # Here we'll create similar-looking faces based on the UTK metadata
            known_test_files = []

            # Process each file and group by gender+ethnicity
            utk_metadata = {}
            for file in utk_files:
                try:
                    parts = file.split("_")
                    if len(parts) >= 3:
                        age = int(parts[0])
                        gender = int(parts[1])
                        race = int(parts[2])
                        key = f"{gender}_{race}"

                        if key not in utk_metadata:
                            utk_metadata[key] = []

                        utk_metadata[key].append((file, age))
                except (ValueError, IndexError):
                    continue

            # Select images for known test set
            for i in range(min(num_known, len(known_identities))):
                if i < len(known_identities):
                    identity = known_identities[i]

                    # Try to find similar faces
                    if len(utk_metadata) > 0:
                        # Randomly select a demographic group
                        group_key = random.choice(list(utk_metadata.keys()))
                        group_files = utk_metadata[group_key]

                        # Select up to 2 images from this group
                        num_to_select = min(2, len(group_files))
                        if num_to_select > 0:
                            selected = random.sample(group_files, num_to_select)
                            known_test_files.extend(
                                [
                                    (f[0], f"{identity}_test_{j}")
                                    for j, f in enumerate(selected)
                                ]
                            )

            # Prepare "unknown" test images (people not in known faces)
            # For this, we'll randomly select images from different gender+ethnicity groups
            unknown_test_files = []
            available_groups = list(utk_metadata.keys())

            for i in range(min(num_unknown, len(available_groups))):
                if i < len(available_groups):
                    group_key = available_groups[i]
                    group_files = utk_metadata[group_key]

                    # Select one image from this group
                    if group_files:
                        selected = random.choice(group_files)
                        unknown_test_files.append((selected[0], f"Unknown_{i}"))

            # Copy known test files
            for file, name in known_test_files:
                src_path = os.path.join(utkface_dir, file)
                dst_path = os.path.join(known_output_dir, f"{name}.jpg")

                # Avoid overwriting existing files
                if os.path.exists(dst_path):
                    import time

                    timestamp = int(time.time())
                    dst_path = os.path.join(known_output_dir, f"{name}_{timestamp}.jpg")

                shutil.copy2(src_path, dst_path)

            # Copy unknown test files
            for file, name in unknown_test_files:
                src_path = os.path.join(utkface_dir, file)
                dst_path = os.path.join(unknown_output_dir, f"{name}.jpg")

                # Avoid overwriting existing files
                if os.path.exists(dst_path):
                    import time

                    timestamp = int(time.time())
                    dst_path = os.path.join(
                        unknown_output_dir, f"{name}_{timestamp}.jpg"
                    )

                shutil.copy2(src_path, dst_path)

            # Print summary
            print(f"\nPrepared test dataset in {output_dir}")
            print(f"  - Known people: {len(known_test_files)} images")
            print(f"  - Unknown people: {len(unknown_test_files)} images")

            return True

        except Exception as e:
            print(f"Error preparing test dataset from UTKFace: {e}")
            import traceback

            traceback.print_exc()
            return False
