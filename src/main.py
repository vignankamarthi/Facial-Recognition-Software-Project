"""
Main Application Module

This module provides functionality for the facial recognition system.
It serves as the entry point for various features and demonstrations.
"""
import os
import sys
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

    anonymizer = None
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
        except KeyboardInterrupt:
            print("\nSetup interrupted. Returning to menu.")
            return
    
    print("\nControls:")
    if anonymize:
        print("- Press 'b' to switch to blur mode")
        print("- Press 'p' to switch to pixelate mode")
        print("- Press 'm' to switch to mask mode")
    print("- Press 'q' to quit")
    print("\nStarting camera...\n")

    try:
        detector = FaceDetector()
        detector.detect_faces_webcam(anonymize, anonymizer)
    except KeyboardInterrupt:
        print("\nFace detection interrupted by user.")
        # Make sure to release any OpenCV resources
        cv2.destroyAllWindows()


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
        print("   Then select option to prepare known faces from UTKFace dataset")
        return

    print(f"Using reference faces from: {sample_dir}")
    print(f"Found {len(os.listdir(sample_dir))} reference faces")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("\nStarting camera...\n")

    try:
        matcher = FaceMatcher(sample_dir)
        matcher.match_faces_webcam()
    except KeyboardInterrupt:
        print("\nFace matching interrupted by user.")
        # Make sure to release any OpenCV resources
        cv2.destroyAllWindows()


def process_single_image():
    """Simplified function to process a single image with options"""
    print("\n" + "=" * 50)
    print("Static Image Processing")
    print("=" * 50)
    
    try:
        image_path = input("Enter the path to the image file: ")
        if not os.path.exists(image_path):
            print(f"Error: File not found at {image_path}")
            return
            
        print("\nProcessing Options:")
        print("1. Basic face detection")
        print("2. Face detection with matching")
        print("3. Face detection with anonymization")
        
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
    except KeyboardInterrupt:
        print("\nImage processing interrupted by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()


def process_image_directory():
    """Simplified function to process a directory of images"""
    print("\n" + "=" * 50)
    print("Directory Processing")
    print("=" * 50)
    
    try:
        dir_path = input("Enter the path to the directory: ")
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            print(f"Error: Directory not found at {dir_path}")
            return
            
        print("\nProcessing Options:")
        print("1. Basic face detection")
        print("2. Face detection with matching")
        print("3. Face detection with anonymization")
        
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
    except KeyboardInterrupt:
        print("\nDirectory processing interrupted by user.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()


def run_bias_testing_demo(use_utkface=True):
    """
    Run the bias testing demo.

    Args:
        use_utkface (bool): Whether to use UTKFace dataset (True) or generic groups (False)

    Returns:
        None
    """
    print("\n" + "=" * 50)
    print("Bias Testing Demonstration")
    print("=" * 50)
    print("This feature demonstrates how facial recognition accuracy")
    print("can vary across different demographic groups.\n")

    if use_utkface:
        print("Using UTKFace dataset with actual demographic categories")
        print("This provides a realistic demonstration of potential bias in")
        print("facial recognition systems across different ethnicities.\n")
    else:
        print("Using generic dataset groups")
        print("This is a simplified demonstration without real demographic data.\n")

    # Option to perform detailed analysis
    detailed = False
    try:
        choice = input("Would you like to perform detailed statistical analysis? (y/n): ")
        detailed = choice.lower() == 'y'
    except KeyboardInterrupt:
        print("\nSetup interrupted. Returning to menu.")
        return

    try:
        analyzer = BiasAnalyzer()
        
        # Run the standard bias demonstration
        analyzer.run_bias_demonstration(use_utkface=use_utkface)
        
        # If detailed analysis was requested and we have results
        if detailed and analyzer.results:
            # Run detailed analysis on the most recent dataset
            dataset_name = list(analyzer.results.keys())[-1] if analyzer.results else "demographic_split_set"
            analyzer.analyze_demographic_bias(dataset_name=dataset_name, detailed=True)
    except KeyboardInterrupt:
        print("\nBias testing interrupted by user.")
        # Close any open matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')


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

    # Menu options with UTKFace dataset options
    options = [
        "Download UTKFace dataset - Aligned images with demographic labels",
        "Set up bias testing with UTKFace - Prepare data for demographic analysis",
        "Prepare known faces from UTKFace - Create reference faces for matching",
        "Prepare test dataset from UTKFace - Create test images for evaluation",
        "Return to main menu"
    ]

    while True:
        print("\nDataset Setup Options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        try:
            choice = int(input(f"\nEnter your choice (1-{len(options)}): "))

            if choice == 1:
                # Download UTKFace dataset
                print("\n--- Downloading UTKFace Dataset ---")
                print("This will download the UTKFace dataset with demographic information.")
                print("The dataset includes faces labeled with age, gender, and ethnicity.")
                print("This is ideal for bias testing of facial recognition.")

                try:
                    sample_size = int(input("\nEnter number of total images to include (500 recommended): "))
                    
                    print("\nSelect ethnicities to include:")
                    print("1. All ethnicities")
                    print("2. White and Black only (for pronounced contrast)")
                    print("3. White, Black, and Asian (most common groups)")
                    print("4. Custom selection")
                    
                    ethnicity_choice = int(input("Enter your choice (1-4): "))
                    specific_ethnicities = None
                    
                    if ethnicity_choice == 2:
                        specific_ethnicities = [0, 1]  # White and Black
                    elif ethnicity_choice == 3:
                        specific_ethnicities = [0, 1, 2]  # White, Black, Asian
                    elif ethnicity_choice == 4:
                        print("\nSelect ethnicities (comma-separated numbers):")
                        print("0: White, 1: Black, 2: Asian, 3: Indian, 4: Others")
                        eth_input = input("Enter numbers (e.g., 0,1,2): ")
                        try:
                            specific_ethnicities = [int(x.strip()) for x in eth_input.split(',')]
                        except:
                            print("Invalid input. Using all ethnicities.")
                            specific_ethnicities = None
                    
                    print("\nDownloading and extracting dataset...")
                    processor.download_and_extract_utkface_dataset(
                        sample_size=sample_size,
                        specific_ethnicities=specific_ethnicities
                    )
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nDownload interrupted by user.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 2:
                # Set up bias testing with UTKFace
                print("\n--- Setting Up Bias Testing Dataset ---")
                print("This will organize UTKFace images by ethnicity for bias testing.")
                print("Images will be copied to data/test_datasets/demographic_split_set/")

                try:
                    # Check if UTKFace dataset exists
                    utkface_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "data", "datasets", 
                                     "utkface", "demographic_split")
                    )
                    
                    if not os.path.exists(utkface_dir):
                        print("UTKFace dataset not found. Please download it first (Option 1).")
                        continue
                        
                    images_per_ethnicity = int(input("\nEnter number of images per ethnicity group (20-50 recommended): "))
                    
                    print("\nPreparing bias testing dataset...")
                    processor.prepare_utkface_for_bias_testing(
                        images_per_ethnicity=images_per_ethnicity
                    )
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nSetup interrupted by user.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 3:
                # Prepare known faces from UTKFace
                print("\n--- Preparing Known Faces from UTKFace ---")
                print("This will create reference faces for the face matching feature.")
                print("Images will be saved to data/known_faces/")

                try:
                    # Check if UTKFace dataset exists
                    utkface_dir = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), "..", "data", "datasets", 
                                     "utkface", "utkface_aligned")
                    )
                    
                    if not os.path.exists(utkface_dir):
                        print("UTKFace dataset not found. Please download it first (Option 1).")
                        continue
                        
                    num_people = int(input("\nEnter number of people to include as known faces: "))
                    
                    print("\nBalance faces across ethnic groups?")
                    print("1. Yes - equal number from each ethnicity")
                    print("2. No - random selection")
                    
                    balance_choice = int(input("Enter your choice (1-2): "))
                    ethnicity_balanced = (balance_choice == 1)
                    
                    print("\nPreparing known faces...")
                    processor.prepare_known_faces_from_utkface(
                        num_people=num_people,
                        ethnicity_balanced=ethnicity_balanced
                    )
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nPreparation interrupted by user.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 4:
                # Prepare test dataset 
                print("\n--- Preparing Test Dataset ---")
                print("This will create test images for face recognition evaluation.")
                print("Images will be saved to data/test_images/")

                try:
                    num_known = int(input("\nEnter number of known people to include: "))
                    num_unknown = int(input("Enter number of unknown people to include: "))
                    print("\nPreparing test dataset from UTKFace...")
                    processor.prepare_test_dataset_from_utkface(
                        num_known=num_known,
                        num_unknown=num_unknown
                    )
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nPreparation interrupted by user.")
                except Exception as e:
                    print(f"An error occurred: {e}")

            elif choice == 5:
                # Return to main menu
                print("Returning to main menu...")
                break
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")

            # Pause after operation completes
            try:
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\nReturning to menu...")

        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation interrupted. Returning to menu...")
            break
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
        {"name": "📊 Demographic Bias Testing (UTKFace)", "webcam": False, "function": run_bias_testing_demo, "args": {"use_utkface": True}},
        {"name": "📊 Generic Bias Testing (Legacy)", "webcam": False, "function": run_bias_testing_demo, "args": {"use_utkface": False}},
        {"name": "💾 Dataset Management", "webcam": False, "function": run_dataset_setup_demo, "args": {}},
        
        # Exit option
        {"name": "❌ Exit Program", "webcam": None, "function": None, "args": {}}
    ]

    while True:
        try:
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
            print("\nProgram interrupted by user.")
            confirm_exit = input("Do you want to exit the program? (y/n): ")
            if confirm_exit.lower() == 'y':
                print("Exiting program. Goodbye!")
                break
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
    parser.add_argument("--utkface", action="store_true", help="Use UTKFace dataset for bias testing")
    parser.add_argument("--image", type=str, help="Process a single image file")
    parser.add_argument("--dir", type=str, help="Process a directory of images")
    parser.add_argument("--setup-dataset", action="store_true", help="Run dataset setup and management")

    try:
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
            run_bias_testing_demo(use_utkface=args.utkface)
        else:
            # If no arguments are provided, run the interactive menu
            main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting gracefully.")
        # Make sure to release any OpenCV resources
        cv2.destroyAllWindows()
        sys.exit(0)
