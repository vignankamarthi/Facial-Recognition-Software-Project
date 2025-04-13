"""
Main Application Module

This module ties together all components of the facial recognition system.
It provides a unified interface for using the system's various features.
"""
import os
import argparse
import cv2
from facial_recognition_software.face_detection import FaceDetector
from facial_recognition_software.face_matching import FaceMatcher
from facial_recognition_software.anonymization import FaceAnonymizer
from facial_recognition_software.bias_testing import BiasAnalyzer
from utilities.image_processing import ImageProcessor


def run_face_detection_demo(anonymize=False):
    """
    Run the face detection demo.

    Args:
        anonymize (bool): Whether to enable anonymization

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("Face Detection Demo")
    print("=" * 50)
    print("Starting webcam for face detection...")

    if anonymize:
        print("Anonymization enabled. Faces will be obscured.")
        anonymizer = FaceAnonymizer()
        
        # Simplified method selection
        print("\nSelect anonymization method:")
        print("1. Blur - Gaussian blur effect")
        print("2. Pixelate - Pixelated censorship effect")
        print("3. Mask - Solid mask with face icon")
        
        methods = ["blur", "pixelate", "mask"]
        try:
            choice = int(input("Enter your choice (1-3): "))
            if 1 <= choice <= 3:
                method = methods[choice-1]
                anonymizer.set_method(method)
            else:
                print("Invalid choice. Using default (blur).")
                method = "blur"
        except ValueError:
            print("Invalid input. Using default (blur).")
            method = "blur"
    
    print("\nControls:")
    if anonymize:
        print("- Press 'b' to switch to blur mode")
        print("- Press 'p' to switch to pixelate mode")
        print("- Press 'm' to switch to mask mode")
    print("- Press 'q' to quit")
    print("\nStarting camera...\n")

    detector = FaceDetector()
    detector.detect_faces_webcam(anonymize)


def run_face_matching_demo():
    """
    Run the face matching demo.

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("Face Matching Demo")
    print("=" * 50)

    # Check if sample faces directory exists and has images
    sample_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "known_faces")
    )

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not os.listdir(sample_dir):
        print(f"No face images found in {sample_dir}")
        print("\nYou need to add reference face images before using face matching.")
        print("\nOptions:")
        print("1. Add your own images to the directory:")
        print(f"   {sample_dir}")
        print("   (Name files with the person's name, e.g., john_doe.jpg)")
        print("\n2. Use the dataset setup to create sample faces:")
        print("   python run_demo.py --setup-dataset")
        print("   Then select option 2: \"Prepare known faces from LFW\"")
        return

    print(f"Using reference faces from: {sample_dir}")
    print(f"Found {len(os.listdir(sample_dir))} reference faces")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("\nStarting camera...\n")

    matcher = FaceMatcher(sample_dir)
    matcher.match_faces_webcam()


def process_single_image():
    """Simplified function to process a single image with options"""
    print("\n" + "=" * 50)
    print("Static Image Processing")
    print("=" * 50)
    
    image_path = input("Enter the path to the image file: ")
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return
        
    print("\nProcessing Options:")
    print("1. Basic face detection")
    print("2. Face detection with matching")
    print("3. Face detection with anonymization")
    
    try:
        option = int(input("Select processing option (1-3): "))
        match = option == 2
        anonymize = option == 3
        
        processor = ImageProcessor()
        processed_image, results = processor.process_image_file(
            image_path, detect=True, match=match, anonymize=anonymize, save_result=True
        )
        
        if processed_image is not None:
            # Display the processed image
            cv2.imshow("Processed Image", processed_image)
            print("\nImage processed successfully!")
            print(f"Faces detected: {results['face_count']}")
            
            if match and results['face_count'] > 0:
                print("Identified faces:")
                for name in results['identified_faces']:
                    print(f"  - {name}")
            
            if 'output_path' in results:
                print(f"Result saved to: {results['output_path']}")
                
            print("\nPress any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                    
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred: {e}")


def process_image_directory():
    """Simplified function to process a directory of images"""
    print("\n" + "=" * 50)
    print("Directory Processing")
    print("=" * 50)
    
    dir_path = input("Enter the path to the directory: ")
    if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
        print(f"Error: Directory not found at {dir_path}")
        return
        
    print("\nProcessing Options:")
    print("1. Basic face detection")
    print("2. Face detection with matching")
    print("3. Face detection with anonymization")
    
    try:
        option = int(input("Select processing option (1-3): "))
        match = option == 2
        anonymize = option == 3
        
        print(f"\nProcessing all images in: {dir_path}")
        print("Each image will be displayed briefly.")
        print("Press any key to advance to the next image.")
        print("Results will be saved to the data/results directory.\n")
        
        processor = ImageProcessor()
        results = processor.process_directory(
            dir_path, detect=True, match=match, anonymize=anonymize, 
            save_results=True, display_results=True
        )
        
        print(f"\nProcessed {len(results)} images")
        total_faces = sum(result.get("face_count", 0) for result in results.values())
        print(f"Total faces detected: {total_faces}")
        
    except ValueError:
        print("Invalid input. Please enter a number.")
    except Exception as e:
        print(f"An error occurred: {e}")


def run_bias_testing_demo():
    """
    Run the bias testing demo.

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("Bias Testing Demonstration")
    print("=" * 50)
    print("This feature demonstrates how facial recognition accuracy")
    print("can vary across different demographic groups.\n")

    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()


def run_dataset_setup_demo():
    """
    Run the dataset setup demo with improved interface.

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("Dataset Setup & Management")
    print("=" * 50)
    print("This tool helps prepare datasets for face matching and bias testing.\n")

    processor = ImageProcessor()

    # Menu options
    options = [
        "Download LFW dataset sample - Get sample face images",
        "Prepare known faces from LFW - Create reference faces for matching",
        "Prepare test dataset from LFW - Create test images for evaluation",
        "Return to main menu"
    ]

    while True:
        print("\nDataset Setup Options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))

            if choice == 1:
                # Download LFW dataset sample
                print("\n--- Downloading LFW Dataset Sample ---")
                print("This will download a subset of the Labeled Faces in the Wild dataset.")
                print("The download is about 200MB and may take several minutes.")

                try:
                    sample_size = int(input("\nEnter number of people to include (10-100 recommended): "))
                    print("\nDownloading and extracting dataset...")
                    processor.download_and_extract_lfw_dataset(sample_size=sample_size)
                    print("Download complete!")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 2:
                # Prepare known faces
                print("\n--- Preparing Known Faces ---")
                print("This will create reference faces for the face matching feature.")
                print("Images will be saved to data/known_faces/")

                try:
                    num_people = int(input("\nEnter number of people to include as known faces: "))
                    print("\nPreparing known faces...")
                    processor.prepare_known_faces_from_lfw(num_people=num_people)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 3:
                # Prepare test dataset
                print("\n--- Preparing Test Dataset ---")
                print("This will create test images for face recognition evaluation.")
                print("Images will be saved to data/test_images/")

                try:
                    num_people = int(input("\nEnter number of known people to include in test set: "))
                    num_images = int(input("Enter number of test images per person: "))
                    print("\nPreparing test dataset...")
                    processor.prepare_test_dataset_from_lfw(num_people=num_people, num_test_images=num_images)
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 4:
                # Return to main menu
                print("Returning to main menu...")
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

            # Pause after operation completes
            input("\nPress Enter to continue...")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation interrupted. Returning to menu...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to run the facial recognition system with a simplified menu.
    """
    print("\n" + "=" * 50)
    print("Facial Recognition System")
    print("A demonstration of technology and ethics")
    print("=" * 50)

    # Define all options
    options = [
        # Webcam features
        {"name": "📷 Face Detection Demo", "webcam": True, "function": run_face_detection_demo, "args": {"anonymize": False}},
        {"name": "📷 Face Anonymization Demo", "webcam": True, "function": run_face_detection_demo, "args": {"anonymize": True}},
        {"name": "📷 Face Matching Demo", "webcam": True, "function": run_face_matching_demo, "args": {}},
        
        # Non-webcam features
        {"name": "🖼️ Process Single Image", "webcam": False, "function": process_single_image, "args": {}},
        {"name": "📁 Process Image Directory", "webcam": False, "function": process_image_directory, "args": {}},
        {"name": "📊 Run Bias Testing", "webcam": False, "function": run_bias_testing_demo, "args": {}},
        {"name": "💾 Dataset Management", "webcam": False, "function": run_dataset_setup_demo, "args": {}},
        
        # Exit option
        {"name": "❌ Exit Program", "webcam": None, "function": None, "args": {}}
    ]

    while True:
        # Display menu with webcam/non-webcam sections
        print("\nMain Menu:")
        print("\n[Webcam Required]")
        webcam_options = [opt for opt in options if opt["webcam"] is True]
        for i, option in enumerate(webcam_options, 1):
            print(f"{i}. {option['name']}")
            
        print("\n[No Webcam Required]")
        non_webcam_options = [opt for opt in options if opt["webcam"] is False]
        for i, option in enumerate(non_webcam_options, len(webcam_options)+1):
            print(f"{i}. {option['name']}")
            
        # Exit option
        exit_option = next(opt for opt in options if opt["webcam"] is None)
        print(f"\n{len(options)}. {exit_option['name']}")
        
        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))
            
            if choice < 1 or choice > len(options):
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
                continue
                
            selected_option = options[choice-1]
            
            # Exit program if selected
            if selected_option["webcam"] is None:
                print("Exiting program. Goodbye!")
                break
                
            # Run the selected function with arguments
            if selected_option["function"]:
                selected_option["function"](**selected_option["args"])
                
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
        if args.match and args.anonymize:
            print("Error: Cannot use both --match and --anonymize together. Please choose one.")
        elif args.match:
            process_single_image()  # Will prompt for options
        elif args.anonymize:
            process_single_image()  # Will prompt for options
        else:
            process_single_image()  # Will prompt for options
    elif args.dir:
        process_image_directory()  # Will prompt for options
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
