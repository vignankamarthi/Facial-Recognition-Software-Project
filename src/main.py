"""
Main Application Module

This module ties together all components of the facial recognition system.
It provides a unified interface for using the system's various features.
"""

import os
import cv2
import numpy as np
import argparse
from face_detection import FaceDetector
from face_matching import FaceMatcher
from anonymization import FaceAnonymizer
from bias_testing import BiasAnalyzer


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
                print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
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
    print("\n=== Face Detection Demo ===")
    print("Starting webcam for face detection...")
    
    if anonymize:
        print("Anonymization enabled. Faces will be obscured.")
        anonymizer = FaceAnonymizer()
        method = get_user_choice(
            "Select anonymization method:",
            ["blur", "pixelate", "mask"]
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
    print("\n=== Face Matching Demo ===")
    
    # Check if sample faces directory exists and has images
    sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_faces'))
    
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        
    if not os.listdir(sample_dir):
        print(f"No face images found in {sample_dir}")
        print("Please add reference face images to compare against.")
        print("Each image should contain one face and be named with the person's name (e.g., john_doe.jpg).")
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
    print("\n=== Bias Testing Demo ===")
    
    analyzer = BiasAnalyzer()
    analyzer.run_bias_demonstration()


def main():
    """
    Main function to run the facial recognition system.
    
    Returns:
        None
    """
    print("=" * 50)
    print("Facial Recognition System")
    print("A demonstration of technology and ethics")
    print("=" * 50)
    
    while True:
        print("\nMain Menu:")
        options = [
            "Face Detection",
            "Face Detection with Anonymization",
            "Face Matching (Identity Verification)",
            "Bias Testing Demonstration",
            "Exit"
        ]
        
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
            
        try:
            choice = int(input("Enter your choice (1-5): "))
            
            if choice == 1:
                run_face_detection_demo(anonymize=False)
            elif choice == 2:
                run_face_detection_demo(anonymize=True)
            elif choice == 3:
                run_face_matching_demo()
            elif choice == 4:
                run_bias_testing_demo()
            elif choice == 5:
                print("Exiting program. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation interrupted by user.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Facial Recognition System")
    parser.add_argument('--detect', action='store_true', help='Run face detection demo')
    parser.add_argument('--anonymize', action='store_true', help='Run face detection with anonymization')
    parser.add_argument('--match', action='store_true', help='Run face matching demo')
    parser.add_argument('--bias', action='store_true', help='Run bias testing demo')
    
    args = parser.parse_args()
    
    # If arguments are provided, run the specific demo
    if args.detect:
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
