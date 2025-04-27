"""
Image Processing Module

This module provides functionality for loading and processing static images
for face detection, matching, and anonymization. It also handles dataset management
for the UTKFace dataset used in bias testing.

The module contains the ImageProcessor class which serves as a bridge between
the various facial recognition components and implements utilities for image
operations, dataset preparation, and batch processing.

Functions and Classes
-------------------
ImageProcessor
    Main class for image processing and dataset management

See Also
--------
facial_recognition_software.face_detection : Face detection functionality
facial_recognition_software.face_matching : Face matching functionality
facial_recognition_software.anonymization : Face anonymization functionality
facial_recognition_software.bias_testing : Bias testing with demographic datasets

Examples
--------
>>> from utilities.image_processing import ImageProcessor
>>> processor = ImageProcessor()
>>> # Process a single image
>>> result_image, info = processor.process_image_file('image.jpg', detect=True)
>>> # Set up datasets
>>> processor.download_and_extract_utkface_dataset()
>>> processor.prepare_utkface_for_bias_testing()
"""

import os
import cv2
import shutil
import zipfile
import re
import urllib.request
import random
import numpy as np

# Remove direct imports to avoid circular dependency
# We'll import these classes only when needed


class ImageProcessor:
    """
    A class to handle static image processing operations.
    
    This class provides methods for processing static images using the
    facial recognition components (detection, matching, anonymization).
    It also provides utilities for dataset preparation and management,
    particularly for working with the UTKFace dataset for bias testing.
    
    The class uses lazy initialization to avoid circular dependencies
    between modules, only importing the required components when needed.
    
    Parameters
    ----------
    known_faces_dir : str, optional
        Directory containing known face images used for matching
        (default: './data/known_faces')
        
    Attributes
    ----------
    known_faces_dir : str
        Path to directory containing known face images
    _detector : FaceDetector or None
        Lazily initialized face detector instance
    _matcher : FaceMatcher or None
        Lazily initialized face matcher instance
    _anonymizer : FaceAnonymizer or None
        Lazily initialized face anonymizer instance
        
    Examples
    --------
    >>> processor = ImageProcessor('./path/to/known_faces')
    >>> # Process an image with face detection
    >>> processed_img, results = processor.process_image(image, detect=True)
    >>> # Set up datasets for bias testing
    >>> processor.prepare_utkface_for_bias_testing()
    """

    def __init__(self, known_faces_dir="./data/known_faces"):
        """
        Initialize the image processor.

        Parameters
        ----------
        known_faces_dir : str, optional
            Directory containing known face images used for matching
            (default: './data/known_faces')
            
        Notes
        -----
        Components like the detector, matcher, and anonymizer are initialized
        lazily when first accessed to avoid circular import dependencies.
        """
        self.known_faces_dir = known_faces_dir
        # Use lazy initialization of components to avoid circular imports
        self._detector = None
        self._matcher = None
        self._anonymizer = None
        
    @property
    def detector(self):
        """
        Lazy initialization of face detector.
        
        Returns
        -------
        FaceDetector
            Face detector instance
            
        Notes
        -----
        This property initializes the detector on first access to avoid
        circular import dependencies.
        """
        if self._detector is None:
            # Import here to avoid circular dependency
            from facial_recognition_software.face_detection import FaceDetector
            self._detector = FaceDetector()
        return self._detector
        
    @property
    def matcher(self):
        """
        Lazy initialization of face matcher.
        
        Returns
        -------
        FaceMatcher
            Face matcher instance configured with known_faces_dir
            
        Notes
        -----
        This property initializes the matcher on first access to avoid
        circular import dependencies.
        """
        if self._matcher is None:
            # Import here to avoid circular dependency
            from facial_recognition_software.face_matching import FaceMatcher
            self._matcher = FaceMatcher(self.known_faces_dir)
        return self._matcher
        
    @property
    def anonymizer(self):
        """
        Lazy initialization of face anonymizer.
        
        Returns
        -------
        FaceAnonymizer
            Face anonymizer instance
            
        Notes
        -----
        This property initializes the anonymizer on first access to avoid
        circular import dependencies.
        """
        if self._anonymizer is None:
            # Import here to avoid circular dependency
            from facial_recognition_software.anonymization import FaceAnonymizer
            self._anonymizer = FaceAnonymizer()
        return self._anonymizer

    def load_image(self, image_path):
        """
        Load an image from the specified path.

        Parameters
        ----------
        image_path : str
            Path to the image file to load

        Returns
        -------
        numpy.ndarray or None
            Loaded image as a NumPy array, or None if loading failed
            
        Examples
        --------
        >>> processor = ImageProcessor()
        >>> image = processor.load_image('path/to/image.jpg')
        >>> if image is not None:
        ...     # Process the image
        ...     height, width = image.shape[:2]
        ...     print(f"Loaded image with dimensions: {width}x{height}")
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

        Parameters
        ----------
        image : numpy.ndarray
            Image to process
        detect : bool, optional
            Whether to detect faces (default: True)
        match : bool, optional
            Whether to match faces against known faces (default: False)
        anonymize : bool, optional
            Whether to anonymize faces (default: False)

        Returns
        -------
        tuple
            (processed_image, results_dict) where:
            - processed_image : numpy.ndarray or None
              The processed image, or None if processing failed
            - results_dict : dict
              Dictionary containing metadata about the processing:
              - "face_count" : int
                Number of faces detected
              - "face_locations" : list
                List of face location tuples
              - "identified_faces" : list
                List of identified face names (if match=True)
              
        Notes
        -----
        At least one of detect, match, or anonymize must be True.
        If match=True or anonymize=True, face detection is always performed
        first as it's required for these operations.
              
        Examples
        --------
        >>> # Detect faces in an image
        >>> processed, results = processor.process_image(image, detect=True)
        >>> print(f"Found {results['face_count']} faces")
        >>> 
        >>> # Match faces against known references
        >>> processed, results = processor.process_image(image, match=True)
        >>> for name in results['identified_faces']:
        ...     print(f"Identified: {name}")
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

        Parameters
        ----------
        image_path : str
            Path to the image file to process
        detect : bool, optional
            Whether to detect faces (default: True)
        match : bool, optional
            Whether to match faces against known faces (default: False)
        anonymize : bool, optional
            Whether to anonymize faces (default: False)
        save_result : bool, optional
            Whether to save the processed image (default: False)
        output_dir : str, optional
            Directory to save results if save_result is True
            (default: 'results' subdirectory in input file's directory)

        Returns
        -------
        tuple
            (processed_image, results_dict) where:
            - processed_image : numpy.ndarray or None
              The processed image, or None if processing failed
            - results_dict : dict
              Dictionary containing metadata about the processing:
              - "face_count" : int
                Number of faces detected
              - "face_locations" : list
                List of face location tuples
              - "identified_faces" : list
                List of identified face names (if match=True)
              - "image_path" : str
                Path to the original image file
              - "output_path" : str
                Path where the result was saved (if save_result=True)
                
        Examples
        --------
        >>> # Process an image file with face detection and save the result
        >>> image, results = processor.process_image_file(
        ...     'path/to/image.jpg',
        ...     detect=True,
        ...     save_result=True,
        ...     output_dir='./output_folder'
        ... )
        >>> if 'output_path' in results:
        ...     print(f"Result saved to: {results['output_path']}")
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

        Parameters
        ----------
        directory_path : str
            Path to the directory containing images to process
        detect : bool, optional
            Whether to detect faces (default: True)
        match : bool, optional
            Whether to match faces against known faces (default: False)
        anonymize : bool, optional
            Whether to anonymize faces (default: False)
        save_results : bool, optional
            Whether to save the processed images (default: False)
        output_dir : str, optional
            Directory to save results if save_results is True
            (default: None, which creates a 'results' subdirectory)
        display_results : bool, optional
            Whether to display results in windows (default: True)

        Returns
        -------
        dict
            Dictionary mapping image paths to their respective result dictionaries.
            Each result dictionary contains the same fields as returned by
            process_image_file() for that specific image.
            
        Notes
        -----
        If display_results is True, each processed image is shown in a separate
        window, with a key press required to advance to the next image.
        All supported image formats (jpg, jpeg, png, bmp) in the directory
        are processed.
            
        Examples
        --------
        >>> # Process all images in a directory with face detection
        >>> results = processor.process_directory(
        ...     './photos',
        ...     detect=True,
        ...     save_results=True,
        ...     output_dir='./processed_photos',
        ...     display_results=False  # Don't show windows
        ... )
        >>> print(f"Processed {len(results)} images")
        >>> # Count total faces found
        >>> total_faces = sum(result.get('face_count', 0) for result in results.values())
        >>> print(f"Found {total_faces} faces total")
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
        
        This method downloads the UTKFace dataset with demographic annotations,
        extracts it, and organizes the images by ethnicity for bias testing.

        Parameters
        ----------
        target_dir : str, optional
            Directory to save the dataset (default: './data/datasets/utkface')
        sample_size : int, optional
            Number of total images to include (default: 500)
        specific_ethnicities : list, optional
            List of ethnicity codes to include (default: None, which includes all)
            Ethnicity codes:
            - 0: White
            - 1: Black
            - 2: Asian
            - 3: Indian
            - 4: Others

        Returns
        -------
        bool
            True if successful, False otherwise
            
        Notes
        -----
        This method requires manual download of the dataset files from Google Drive
        due to download limitations. It prompts the user to download the files and
        place them in the specified directory.
        
        Each image in the UTKFace dataset has a filename in the format:
        [age]_[gender]_[race]_[date&time].jpg
        - gender: 0 (male), 1 (female)
        - race/ethnicity: 0 (White), 1 (Black), 2 (Asian), 3 (Indian), 4 (Others)
            
        Examples
        --------
        >>> # Download all ethnicities with default sample size
        >>> processor.download_and_extract_utkface_dataset()
        >>> 
        >>> # Download only White and Black ethnicities with 300 samples
        >>> processor.download_and_extract_utkface_dataset(
        ...     sample_size=300,
        ...     specific_ethnicities=[0, 1]
        ... )
        """
        print("Starting UTKFace dataset download...")

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # URL for the UTKFace dataset - using the "in the wild" folder
        utkface_folder_url = "https://drive.google.com/drive/folders/1HROmgviy4jUUUaCdvvrQ8PcqtNg2jn3G?usp=share_link"
        archive_dir = os.path.join(target_dir, "archives")
        extract_dir = os.path.join(target_dir, "utkface_data")

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
                # Create archive directory
                if not os.path.exists(archive_dir):
                    os.makedirs(archive_dir)
                    
                print(f"Downloading UTKFace dataset...")
                print("This may take several minutes. Please be patient.")

                # Alternative to using gdown for a folder - we'll use a manual approach
                print("\nDue to download limitations, please follow these steps:")
                print(f"1. Visit: {utkface_folder_url}")
                print("2. Download the available files manually")
                print(f"3. Place them in: {archive_dir}")
                
                manual_download = input("\nHave you downloaded the files manually? (y/n): ")
                if manual_download.lower() != 'y':
                    print("Download process cancelled.")
                    return False
                    
                # Check if files were actually downloaded
                archive_files = [f for f in os.listdir(archive_dir) if f.endswith(('.zip', '.tar.gz', '.tgz'))]
                if not archive_files:
                    print(f"No archive files found in {archive_dir}. Please download the files manually.")
                    return False
                    
                # Create extraction directory
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
                os.makedirs(extract_dir)
                
                # Extract all archive files
                for archive_file in archive_files:
                    archive_path = os.path.join(archive_dir, archive_file)
                    print(f"Extracting {archive_file}...")
                    
                    # Handle different archive types
                    if archive_file.endswith('.zip'):
                        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                    elif archive_file.endswith(('.tar.gz', '.tgz')):
                        import tarfile
                        with tarfile.open(archive_path, 'r:gz') as tar_ref:
                            tar_ref.extractall(extract_dir)
                            
                print("Extraction complete!")

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
            files = []
            # Recursively search all files in extract_dir and its subdirectories
            for root, _, filenames in os.walk(extract_dir):
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        files.append(os.path.join(root, filename))

            print(f"Found {len(files)} total images in extracted dataset")
            
            # Filter for valid files with proper naming
            valid_files = []
            for file_path in files:
                try:
                    # Get just the filename
                    file = os.path.basename(file_path)
                    
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
                            valid_files.append((file_path, race))
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

            for file_path, race in valid_files:
                # Get just the filename without the path
                file = os.path.basename(file_path)
                ethnicity_name = ethnicity_names[race].lower()
                ethnicity_dir = os.path.join(demographics_dir, ethnicity_name)
                dst_path = os.path.join(ethnicity_dir, file)

                shutil.copy2(file_path, dst_path)
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
        
        This method copies a balanced sample of images from each ethnicity group
        in the UTKFace dataset to a directory structure suitable for bias testing.

        Parameters
        ----------
        utkface_dir : str, optional
            Source directory with ethnicity-separated UTKFace images
            (default: './data/datasets/utkface/demographic_split')
        test_datasets_dir : str, optional
            Target directory for bias testing
            (default: './data/test_datasets/demographic_split_set')
        images_per_ethnicity : int, optional
            Maximum number of images per ethnicity group (default: 25)

        Returns
        -------
        bool
            True if successful, False otherwise
            
        Notes
        -----
        This method requires that the UTKFace dataset has been downloaded and
        organized by ethnicity first. You should call download_and_extract_utkface_dataset()
        before using this method.
        
        The resulting directory structure will be:
        test_datasets_dir/
            ├── white/
            ├── black/
            ├── asian/
            ├── indian/
            └── others/
            
        Examples
        --------
        >>> # First download the dataset
        >>> processor.download_and_extract_utkface_dataset()
        >>> # Then prepare for bias testing with 30 images per ethnicity
        >>> processor.prepare_utkface_for_bias_testing(images_per_ethnicity=30)
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
        utkface_dir="./data/datasets/utkface/utkface_data",
        output_dir="./data/known_faces",
    ):
        """
        Prepare a set of known faces from the UTKFace dataset for face matching.
        
        This method selects a subset of faces from the UTKFace dataset to use as
        reference faces for the face matching feature. It can optionally select faces
        with balanced representation across ethnicities.

        Parameters
        ----------
        num_people : int, optional
            Number of reference people to include (default: 20)
        ethnicity_balanced : bool, optional
            Whether to balance selection across ethnicities (default: True)
        utkface_dir : str, optional
            Directory containing the UTKFace dataset
            (default: './data/datasets/utkface/utkface_data')
        output_dir : str, optional
            Directory to save the known faces
            (default: './data/known_faces')

        Returns
        -------
        bool
            True if successful, False otherwise
            
        Notes
        -----
        When ethnicity_balanced is True, the method attempts to select an equal
        number of faces from each ethnicity group to ensure diversity in the
        reference set. This is important for reducing bias in the face matching
        feature.
        
        The files are named with ethnicity information and an index, e.g.,
        'White_0.jpg' or 'Asian_1.jpg', which serves as the identity label
        in the face matching system.
            
        Examples
        --------
        >>> # Create a balanced set of 30 reference faces
        >>> processor.prepare_known_faces_from_utkface(
        ...     num_people=30,
        ...     ethnicity_balanced=True
        ... )
        >>> # Create an unbalanced set (random selection)
        >>> processor.prepare_known_faces_from_utkface(
        ...     num_people=15,
        ...     ethnicity_balanced=False
        ... )
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

            # Get all image files recursively
            files = []
            for root, _, filenames in os.walk(utkface_dir):
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        # Store full path and filename
                        file_path = os.path.join(root, filename)
                        files.append((file_path, filename))

            # Group by ethnicity if balanced selection is requested
            if ethnicity_balanced:
                ethnicity_files = {code: [] for code in ethnicity_names}

                # Sort files by ethnicity
                for file_path, filename in files:
                    try:
                        parts = filename.split("_")
                        if len(parts) >= 3:
                            race = int(parts[2])
                            if race in ethnicity_names:
                                ethnicity_files[race].append((file_path, filename))
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
                                (file_path, f"{ethnicity_names[race]}_{i}")
                                for i, (file_path, filename) in enumerate(ethnicity_selection)
                            ]
                        )

                # If we need more files, add extras
                if len(selected_files) < num_people:
                    all_remaining = []
                    for race, eth_files in ethnicity_files.items():
                        # Get files we didn't already select
                        already_selected_paths = [f[0] for f in selected_files]
                        remaining = [f for f in eth_files if f[0] not in already_selected_paths]
                        all_remaining.extend(
                            [
                                (file_path, f"{ethnicity_names[race]}_extra_{i}")
                                for i, (file_path, filename) in enumerate(remaining)
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
                        (file_path, f"Person_{i}") for i, (file_path, filename) in enumerate(files)
                    ]
                else:
                    selected_files = [
                        (file_path, f"Person_{i}")
                        for i, (file_path, filename) in enumerate(random.sample(files, num_people))
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
        utkface_dir="./data/datasets/utkface/utkface_data",
        known_faces_dir="./data/known_faces",
        output_dir="./data/test_images",
    ):
        """
        Prepare a test dataset from the UTKFace dataset for face matching evaluation.
        
        This method creates two sets of test images from the UTKFace dataset:
        1. Known people - Additional images of people in the known faces directory
        2. Unknown people - Images of people not in the known faces directory
        
        These images can be used to evaluate the accuracy of the face matching system.

        Parameters
        ----------
        num_known : int, optional
            Number of known people to include in test set (default: 5)
        num_unknown : int, optional
            Number of unknown people to include in test set (default: 5)
        utkface_dir : str, optional
            Directory containing the UTKFace dataset
            (default: './data/datasets/utkface/utkface_data')
        known_faces_dir : str, optional
            Directory containing known faces
            (default: './data/known_faces')
        output_dir : str, optional
            Directory to save test images
            (default: './data/test_images')

        Returns
        -------
        bool
            True if successful, False otherwise
            
        Notes
        -----
        Since the UTKFace dataset doesn't have identity labels, this method uses
        gender+ethnicity+age as a proxy for identity. It attempts to find faces
        with similar demographic characteristics to serve as additional images
        of "known" people.
        
        The resulting directory structure will be:
        output_dir/
            ├── known/    # Additional images of people in the known set
            └── unknown/  # Images of people not in the known set
            
        Examples
        --------
        >>> # Create a test dataset with default settings
        >>> processor.prepare_test_dataset_from_utkface()
        >>> # Create a larger test set
        >>> processor.prepare_test_dataset_from_utkface(
        ...     num_known=10,
        ...     num_unknown=15
        ... )
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

            # Get all image files from UTKFace recursively
            utk_files = []
            for root, _, filenames in os.walk(utkface_dir):
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        file_path = os.path.join(root, filename)
                        utk_files.append((file_path, filename))

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
            for file_path, filename in utk_files:
                try:
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        age = int(parts[0])
                        gender = int(parts[1])
                        race = int(parts[2])
                        key = f"{gender}_{race}"

                        if key not in utk_metadata:
                            utk_metadata[key] = []

                        utk_metadata[key].append((file_path, filename, age))
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
                                    (file_path, f"{identity}_test_{j}")
                                    for j, (file_path, filename, age) in enumerate(selected)
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
                        file_path, filename, age = selected
                        unknown_test_files.append((file_path, f"Unknown_{i}"))

            # Copy known test files
            for file_path, name in known_test_files:
                dst_path = os.path.join(known_output_dir, f"{name}.jpg")

                # Avoid overwriting existing files
                if os.path.exists(dst_path):
                    import time

                    timestamp = int(time.time())
                    dst_path = os.path.join(known_output_dir, f"{name}_{timestamp}.jpg")

                shutil.copy2(file_path, dst_path)

            # Copy unknown test files
            for file_path, name in unknown_test_files:
                dst_path = os.path.join(unknown_output_dir, f"{name}.jpg")

                # Avoid overwriting existing files
                if os.path.exists(dst_path):
                    import time

                    timestamp = int(time.time())
                    dst_path = os.path.join(
                        unknown_output_dir, f"{name}_{timestamp}.jpg"
                    )

                shutil.copy2(file_path, dst_path)

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
