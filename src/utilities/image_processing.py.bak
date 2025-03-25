"""
Image Processing Module

This module provides functionality for loading and processing static images
for face detection, matching, and anonymization.
"""

import os
import cv2
import shutil
from src.facial_recognition_software.face_detection import FaceDetector
from src.facial_recognition_software.face_matching import FaceMatcher
from src.facial_recognition_software.anonymization import FaceAnonymizer


class ImageProcessor:
    """A class to handle static image processing operations."""

    def __init__(self, known_faces_dir="../data/sample_faces"):
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

    def download_and_extract_lfw_dataset(
        self, target_dir="../data/datasets/lfw", sample_size=None
    ):
        """
        Download and extract a sample of the LFW dataset.

        Args:
            target_dir (str): Directory to save the dataset
            sample_size (int, optional): Number of people to include (None for all)

        Returns:
            bool: True if successful, False otherwise
        """
        import urllib.request
        import tarfile
        import random
        import shutil

        print("Starting LFW dataset download...")

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # URL for the LFW dataset
        lfw_url = "https://ndownloader.figshare.com/files/5976018"
        tgz_file = os.path.join(target_dir, "lfw.tgz")
        extract_dir = os.path.join(target_dir, "lfw")

        try:
            # Download the dataset if not already downloaded
            if not os.path.exists(tgz_file):
                print(f"Downloading LFW dataset from {lfw_url}...")
                urllib.request.urlretrieve(lfw_url, tgz_file)
                print("Download complete!")

            # Extract the dataset if not already extracted
            if not os.path.exists(extract_dir):
                print("Extracting dataset...")
                with tarfile.open(tgz_file) as tar:
                    tar.extractall(path=target_dir)
                print("Extraction complete!")

            # If sample_size is specified, create a random sample
            if sample_size is not None:
                print(f"Creating sample of {sample_size} people...")

                # Get all person directories
                person_dirs = [
                    d
                    for d in os.listdir(extract_dir)
                    if os.path.isdir(os.path.join(extract_dir, d))
                ]

                # Select a random sample
                if sample_size > len(person_dirs):
                    sample_size = len(person_dirs)

                selected_persons = random.sample(person_dirs, sample_size)

                # Create sample directory
                sample_dir = os.path.join(target_dir, "lfw_sample")
                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)
                os.makedirs(sample_dir)

                # Copy selected person directories to sample directory
                for person in selected_persons:
                    src_dir = os.path.join(extract_dir, person)
                    dst_dir = os.path.join(sample_dir, person)
                    shutil.copytree(src_dir, dst_dir)

                print(f"Sample dataset created at {sample_dir}")

            return True

        except Exception as e:
            print(f"Error downloading or extracting LFW dataset: {e}")
            return False

    def prepare_known_faces_from_lfw(
        self,
        num_people=5,
        num_images_per_person=1,
        lfw_dir="../data/datasets/lfw/lfw",
        output_dir="../data/sample_faces",
    ):
        """
        Prepare a set of known faces from the LFW dataset.

        Args:
            num_people (int): Number of people to include
            num_images_per_person (int): Number of images per person to include
            lfw_dir (str): Directory containing the LFW dataset
            output_dir (str): Directory to save the known faces

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(lfw_dir):
            print(f"Error: LFW dataset not found at {lfw_dir}")
            return False

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Get people with multiple images
            people = []
            for person_dir in os.listdir(lfw_dir):
                dir_path = os.path.join(lfw_dir, person_dir)
                if os.path.isdir(dir_path):
                    image_count = len(
                        [
                            f
                            for f in os.listdir(dir_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        ]
                    )
                    if image_count >= num_images_per_person + 1:  # +1 for test images
                        people.append((person_dir, image_count))

            # Select random people
            if len(people) < num_people:
                print(f"Warning: Only {len(people)} people with enough images found")
                num_people = len(people)

            import random

            selected_people = random.sample(people, num_people)

            # Copy images to output directory
            for person, _ in selected_people:
                # Get all images for this person
                person_dir = os.path.join(lfw_dir, person)
                images = [
                    f
                    for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                # Select random images
                selected_images = random.sample(images, num_images_per_person)

                # Copy each selected image
                for img in selected_images:
                    src_path = os.path.join(person_dir, img)
                    dst_path = os.path.join(output_dir, f"{person}.jpg")
                    shutil.copy2(src_path, dst_path)

            print(f"Prepared {num_people} known faces in {output_dir}")
            return True

        except Exception as e:
            print(f"Error preparing known faces: {e}")
            return False

    def prepare_test_dataset_from_lfw(
        self,
        num_people=5,
        num_test_images=3,
        lfw_dir="../data/datasets/lfw/lfw",
        known_faces_dir="../data/sample_faces",
        output_dir="../data/test_images",
    ):
        """
        Prepare a test dataset from the LFW dataset.

        This will select images of people in the known_faces_dir (should match)
        and images of people not in known_faces_dir (should not match).

        Args:
            num_people (int): Number of known people to include in test set
            num_test_images (int): Number of test images per known person
            lfw_dir (str): Directory containing the LFW dataset
            known_faces_dir (str): Directory containing known faces
            output_dir (str): Directory to save test images

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(lfw_dir):
            print(f"Error: LFW dataset not found at {lfw_dir}")
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
            # Get list of known people
            known_people = set()
            for filename in os.listdir(known_faces_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Extract person name from filename
                    person = os.path.splitext(filename)[0]
                    known_people.add(person)

            # Prepare test images for known people
            import random

            for person in list(known_people)[:num_people]:
                # Get all images for this person
                person_dir = os.path.join(lfw_dir, person)
                if not os.path.exists(person_dir):
                    print(f"Warning: Directory for {person} not found in LFW dataset")
                    continue

                images = [
                    f
                    for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                # Skip known face image
                known_image = None
                for img in os.listdir(known_faces_dir):
                    if person in img:
                        known_image = img
                        break

                if known_image:
                    known_path = os.path.join(known_faces_dir, known_image)
                    known_filename = os.path.basename(known_path)
                    if known_filename in images:
                        images.remove(known_filename)

                # Select random test images
                num_to_select = min(num_test_images, len(images))
                if num_to_select > 0:
                    selected_images = random.sample(images, num_to_select)

                    # Copy each selected image
                    for i, img in enumerate(selected_images):
                        src_path = os.path.join(person_dir, img)
                        dst_path = os.path.join(known_output_dir, f"{person}_{i+1}.jpg")
                        shutil.copy2(src_path, dst_path)

            # Prepare test images for unknown people
            all_people = set()
            for person_dir in os.listdir(lfw_dir):
                if os.path.isdir(os.path.join(lfw_dir, person_dir)):
                    all_people.add(person_dir)

            unknown_people = all_people - known_people
            selected_unknown = random.sample(list(unknown_people), num_people)

            for person in selected_unknown:
                # Get all images for this person
                person_dir = os.path.join(lfw_dir, person)
                images = [
                    f
                    for f in os.listdir(person_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                # Select one random image
                if images:
                    selected_image = random.choice(images)
                    src_path = os.path.join(person_dir, selected_image)
                    dst_path = os.path.join(unknown_output_dir, f"{person}.jpg")
                    shutil.copy2(src_path, dst_path)

            print(f"Prepared test dataset in {output_dir}")
            print(f"  - Known people: {len(os.listdir(known_output_dir))} images")
            print(f"  - Unknown people: {len(os.listdir(unknown_output_dir))} images")

            return True

        except Exception as e:
            print(f"Error preparing test dataset: {e}")
            return False
