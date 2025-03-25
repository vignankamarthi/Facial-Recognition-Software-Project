"""
Main Application Module

This module ties together all components of the facial recognition system.
It provides a unified interface for using the system's various features.
"""

import os
import argparse
import cv2
from src.facial_recognition_software.face_detection import FaceDetector
from src.facial_recognition_software.face_matching import FaceMatcher
from src.facial_recognition_software.anonymization import FaceAnonymizer
from src.facial_recognition_software.bias_testing import BiasAnalyzer
from src.utilities.image_processing import ImageProcessor


def get_user_choice(prompt, options):
    """
    Get a valid choice from the user.

    Args:
        prompt (str): Prompt to display to the user
        options (list): Valid options

    Returns:
        str: User's choice
    """
    while True:
        print(prompt)
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(
                    f"Invalid choice. Please enter a number between 1 and {len(options)}."
                )
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_face_detection_demo(anonymize=False):
    """
    Run the face detection demo.

    Args:
        anonymize (bool): Whether to enable anonymization

    Returns:
        None
    """
    print("\n--- Face Detection Demo ---")
    print("Starting webcam for face detection...")

    if anonymize:
        print("Anonymization enabled. Faces will be obscured.")
        anonymizer = FaceAnonymizer()
        method = get_user_choice(
            "Select anonymization method:", ["blur", "pixelate", "mask"]
        )
        anonymizer.set_method(method)

    detector = FaceDetector()
    detector.detect_faces_webcam(anonymize)


def run_face_matching_demo():
    """
    Run the face matching demo.

    Returns:
        None
    """
    print("\n--- Face Matching Demo ---")

    # Check if sample faces directory exists and has images
    sample_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample_faces")
    )

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not os.listdir(sample_dir):
        print(f"No face images found in {sample_dir}")
        print("Please add reference face images to compare against.")
        print(
            "Each image should contain one face and be named with the person's name (e.g., john_doe.jpg)."
        )
        return

    print(f"Using reference faces from: {sample_dir}")
    print("Starting webcam for face matching...")

    matcher = FaceMatcher(sample_dir)
    matcher.match_faces_webcam()


def run_bias_testing_demo():
    """
    Run the bias testing demo.

    Returns:
        None
    """
    print("\n--- Bias Testing Demo ---")

    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()


def run_static_image_demo(image_path=None, directory_path=None, detect=True, match=False, anonymize=False):
    """
    Run the static image processing demo.

    Args:
        image_path (str): Path to a single image file
        directory_path (str): Path to a directory of images
        detect (bool): Whether to detect faces
        match (bool): Whether to match faces
        anonymize (bool): Whether to anonymize faces

    Returns:
        None
    """
    print("\n--- Static Image Processing Demo ---")
    processor = ImageProcessor()

    if image_path and os.path.exists(image_path):
        # Process a single image
        print(f"Processing image: {image_path}")
        processed_image, results = processor.process_image_file(
            image_path, detect, match, anonymize, save_result=True
        )

        if processed_image is not None:
            # Display the processed image
            cv2.imshow("Processed Image", processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Print results
            print(f"\nResults for {os.path.basename(image_path)}:")
            print(f"  Faces detected: {results['face_count']}")
            if match and results['face_count'] > 0:
                print("  Identified faces:")
                for name in results['identified_faces']:
                    print(f"    - {name}")

    elif directory_path and os.path.exists(directory_path):
        # Process all images in a directory
        print(f"Processing images in directory: {directory_path}")
        processor.process_directory(
            directory_path, detect, match, anonymize, save_results=True
        )
    else:
        print("No valid image path or directory provided.")
        print("Please specify a valid image file or directory.")

# TODO: PLEASE REVIEW THIS
def run_dataset_setup_demo():
    """
    Run the dataset setup demo.

    Returns:
        None
    """
    print("\n--- Dataset Setup Demo ---")
    processor = ImageProcessor()

    # Menu options
    options = [
        "Download LFW dataset sample",
        "Prepare known faces from LFW",
        "Prepare test dataset from LFW",
        "Return to main menu"
    ]

    while True:
        print("\nDataset Setup Options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input(f"Enter your choice (1-{len(options)}): "))

            if choice == 1:
                # Download LFW dataset sample
                sample_size = int(input("Enter number of people to include (10-100): "))
                processor.download_and_extract_lfw_dataset(sample_size=sample_size)
            elif choice == 2:
                # Prepare known faces
                num_people = int(input("Enter number of people to include as known faces: "))
                processor.prepare_known_faces_from_lfw(num_people=num_people)
            elif choice == 3:
                # Prepare test dataset
                num_people = int(input("Enter number of known people to include in test set: "))
                num_images = int(input("Enter number of test images per person: "))
                processor.prepare_test_dataset_from_lfw(num_people=num_people, num_test_images=num_images)
            elif choice == 4:
                # Return to main menu
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An error occurred: {e}")


def main():
    """
    Main function to run the facial recognition system.

    Returns:
        None
    """
    print("-" * 50)
    print("Facial Recognition System")
    print("A demonstration of technology and ethics")
    print("-" * 50)

    while True:
        print("\nMain Menu:")
        options = [
            "Face Detection (Webcam)",
            "Face Detection with Anonymization (Webcam)",
            "Face Matching (Webcam)",
            "Static Image Processing",
            "Dataset Setup & Management",
            "Bias Testing Demonstration",
            "Exit",
        ]

        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input(f"Enter your choice (1-{len(options)}): "))

            if choice == 1:
                run_face_detection_demo(anonymize=False)
            elif choice == 2:
                run_face_detection_demo(anonymize=True)
            elif choice == 3:
                run_face_matching_demo()
            elif choice == 4:
                # Static image processing submenu
                sub_options = [
                    "Process a single image (detection only)",
                    "Process a single image (with face matching)",
                    "Process a single image (with anonymization)",
                    "Process a directory of images",
                    "Return to main menu"
                ]
                
                print("\nStatic Image Processing Options:")
                for i, option in enumerate(sub_options, 1):
                    print(f"{i}. {option}")
                    
                sub_choice = int(input(f"Enter your choice (1-{len(sub_options)}): "))
                
                if sub_choice == 1:
                    image_path = input("Enter the path to the image file: ")
                    run_static_image_demo(image_path=image_path, detect=True, match=False, anonymize=False)
                elif sub_choice == 2:
                    image_path = input("Enter the path to the image file: ")
                    run_static_image_demo(image_path=image_path, detect=True, match=True, anonymize=False)
                elif sub_choice == 3:
                    image_path = input("Enter the path to the image file: ")
                    run_static_image_demo(image_path=image_path, detect=True, match=False, anonymize=True)
                elif sub_choice == 4:
                    dir_path = input("Enter the path to the directory: ")
                    run_static_image_demo(directory_path=dir_path, detect=True, match=True, anonymize=False)
                elif sub_choice == 5:
                    continue
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(sub_options)}.")
            elif choice == 5:
                run_dataset_setup_demo()
            elif choice == 6:
                run_bias_testing_demo()
            elif choice == 7:
                print("Exiting program. Goodbye!")
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument("--detect", action="store_true", help="Run face detection demo")
    parser.add_argument(
        "--anonymize", action="store_true", help="Run face detection with anonymization"
    )
    parser.add_argument("--match", action="store_true", help="Run face matching demo")
    parser.add_argument("--bias", action="store_true", help="Run bias testing demo")
    parser.add_argument("--image", type=str, help="Process a single image file")
    parser.add_argument("--dir", type=str, help="Process a directory of images")
    parser.add_argument("--setup-dataset", action="store_true", help="Run dataset setup and management")

    args = parser.parse_args()

    # If arguments are provided, run the specific demo
    if args.image:
        run_static_image_demo(image_path=args.image, match=args.match, anonymize=args.anonymize)
    elif args.dir:
        run_static_image_demo(directory_path=args.dir, match=args.match, anonymize=args.anonymize)
    elif args.setup_dataset:
        run_dataset_setup_demo()
    elif args.detect:
        run_face_detection_demo(anonymize=False)
    elif args.anonymize:
        run_face_detection_demo(anonymize=True)
    elif args.match:
        run_face_matching_demo()
    elif args.bias:
        run_bias_testing_demo()
    else:
        # If no arguments are provided, run the interactive menu
        main()
