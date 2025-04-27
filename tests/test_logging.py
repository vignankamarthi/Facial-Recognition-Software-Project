#!/usr/bin/env python3
"""
Test script for the new logging system.

This script demonstrates how to use the logging system and error handling
features of the Facial Recognition Software Project.

Usage:
    python tests/test_logging.py
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our logging and utility modules
from src.utilities.logger import get_logger, log_exception, log_method_call
from src.utilities.common_utils import (
    FaceRecognitionError, DetectionError, FileError, safely_close_windows
)

# Initialize logger for this module
logger = get_logger(__name__)


@log_method_call(logger)
def simulate_successful_operation(name, duration=1.0):
    """
    Simulate a successful operation.
    
    Parameters
    ----------
    name : str
        Name of the operation
    duration : float, optional
        Duration of the operation in seconds (default: 1.0)
    
    Returns
    -------
    tuple
        (True, dict) where dict contains operation information
    """
    logger.info(f"Starting operation: {name}")
    
    # Simulate work
    for i in range(10):
        # Log progress
        logger.debug(f"Operation {name}: step {i+1}/10")
        time.sleep(duration / 10)
    
    logger.info(f"Successfully completed operation: {name}")
    return True, {"name": name, "duration": duration, "status": "success"}


@log_method_call(logger)
def simulate_failing_operation(name, fail_at=5, duration=1.0):
    """
    Simulate an operation that fails.
    
    Parameters
    ----------
    name : str
        Name of the operation
    fail_at : int, optional
        Step at which to fail (default: 5)
    duration : float, optional
        Duration of the operation in seconds (default: 1.0)
    
    Returns
    -------
    tuple
        (False, dict) where dict contains error information
    
    Raises
    ------
    DetectionError
        If the operation fails
    """
    logger.info(f"Starting operation: {name}")
    
    # Simulate work
    for i in range(10):
        # Log progress
        logger.debug(f"Operation {name}: step {i+1}/10")
        
        # Fail at specified step
        if i + 1 == fail_at:
            error_msg = f"Operation {name} failed at step {i+1}"
            logger.error(error_msg)
            raise DetectionError(error_msg, f"Step {i+1}/{10}")
        
        time.sleep(duration / 10)
    
    # This should not be reached if we're failing
    logger.info(f"Successfully completed operation: {name}")
    return True, {"name": name, "duration": duration, "status": "success"}


@log_method_call(logger)
def process_image_with_error_handling(file_path, operation_type="detect"):
    """
    Process an image with proper error handling.
    
    Parameters
    ----------
    file_path : str
        Path to the image file
    operation_type : str, optional
        Type of operation to perform (default: "detect")
    
    Returns
    -------
    tuple
        (success, result_dict) where:
        - success : bool
          True if processing was successful, False otherwise
        - result_dict : dict
          Contains processing results or error information
    """
    # Validate input
    if not file_path:
        error_msg = "No file path provided"
        logger.error(error_msg)
        return False, {"error": error_msg, "type": "ValueError"}
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File does not exist: {file_path}"
            logger.error(error_msg)
            raise FileError(error_msg)
        
        # Try to load the image
        logger.info(f"Loading image: {file_path}")
        image = cv2.imread(file_path)
        
        if image is None:
            error_msg = f"Failed to load image: {file_path}"
            logger.error(error_msg)
            raise FileError(error_msg, "Image format may be unsupported")
        
        # Log image information
        height, width, channels = image.shape
        logger.info(f"Image loaded: {width}x{height}, {channels} channels")
        
        # Simulate different operations
        if operation_type == "detect":
            # Simulate face detection
            logger.info("Starting face detection")
            time.sleep(0.5)  # Simulate processing time
            
            # Simulate finding faces (create random face locations)
            num_faces = np.random.randint(0, 5)  # Random number of faces
            face_locations = []
            
            for _ in range(num_faces):
                # Generate random face location
                top = np.random.randint(0, height // 2)
                right = np.random.randint(width // 2, width)
                bottom = np.random.randint(height // 2, height)
                left = np.random.randint(0, width // 2)
                face_locations.append((top, right, bottom, left))
            
            logger.info(f"Detected {num_faces} faces")
            
            # Create output image with boxes
            result_image = image.copy()
            for i, (top, right, bottom, left) in enumerate(face_locations):
                # Draw rectangle around face
                cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
                logger.debug(f"Drew box for face {i+1} at location ({left}, {top}, {right}, {bottom})")
            
            return True, {
                "image": result_image,
                "face_count": num_faces,
                "face_locations": face_locations
            }
            
        elif operation_type == "fail":
            # Simulate an error condition
            logger.info("This operation is designed to fail")
            raise DetectionError("Simulated detection failure", "Test error condition")
            
        else:
            # Unknown operation
            error_msg = f"Unknown operation type: {operation_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except FileError as e:
        # Handle file errors
        log_exception(logger, f"File error processing {file_path}", e)
        return False, {"error": str(e), "type": "FileError"}
        
    except DetectionError as e:
        # Handle detection errors
        log_exception(logger, f"Detection error processing {file_path}", e)
        return False, {"error": str(e), "type": "DetectionError"}
        
    except Exception as e:
        # Handle unexpected errors
        log_exception(logger, f"Unexpected error processing {file_path}", e)
        return False, {"error": str(e), "type": "Unexpected"}


def main():
    """
    Main test function.
    """
    print("\n===== LOGGING SYSTEM TEST =====\n")
    print("This script tests the new logging and error handling systems.")
    print("Check the logs directory for the generated log files.")
    
    # Create logs directory if it doesn't exist
    Path(os.path.join(project_root, "logs")).mkdir(exist_ok=True)
    
    # Test basic logging
    print("\n1. Testing basic logging levels...")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test successful operation
    print("\n2. Testing successful operation...")
    success, result = simulate_successful_operation("TestOperation", 0.5)
    if success:
        print(f"  Operation succeeded: {result}")
    else:
        print(f"  Operation failed: {result}")
    
    # Test failing operation with try-except
    print("\n3. Testing failing operation with try-except...")
    try:
        success, result = simulate_failing_operation("FailingOperation", 3, 0.5)
        if success:
            print(f"  Operation succeeded: {result}")
        else:
            print(f"  Operation failed: {result}")
    except DetectionError as e:
        print(f"  Caught exception: {e}")
    
    # Test failing operation with result tuple
    print("\n4. Testing operation with error handling return value...")
    try:
        # Find a non-existent image file
        test_image = os.path.join(project_root, "non_existent.jpg")
        success, result = process_image_with_error_handling(test_image)
        if success:
            print(f"  Processing succeeded: {result.get('face_count')} faces found")
        else:
            print(f"  Processing failed: {result.get('error')}")
    except Exception as e:
        log_exception(logger, "Unexpected error in main", e)
        print(f"  Unexpected error: {e}")
    
    # Test successful image processing
    print("\n5. Testing successful image processing (create test image if needed)...")
    try:
        # Create a test image if no sample image is available
        test_dir = os.path.join(project_root, "tests", "data")
        os.makedirs(test_dir, exist_ok=True)
        test_image = os.path.join(test_dir, "test_image.jpg")
        
        # Check if test image exists, create it if not
        if not os.path.exists(test_image):
            # Create a simple test image
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            # Draw something on it
            cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 255), -1)
            cv2.circle(img, (200, 150), 100, (255, 0, 0), -1)
            # Add face-like circles
            cv2.circle(img, (150, 120), 30, (0, 255, 255), -1)  # Left eye
            cv2.circle(img, (250, 120), 30, (0, 255, 255), -1)  # Right eye
            cv2.ellipse(img, (200, 180), (60, 40), 0, 0, 180, (0, 255, 255), -1)  # Mouth
            # Save the image
            cv2.imwrite(test_image, img)
            logger.info(f"Created test image: {test_image}")
            print(f"  Created test image: {test_image}")
        
        # Process the test image
        success, result = process_image_with_error_handling(test_image)
        if success:
            print(f"  Image processing succeeded: {result.get('face_count')} faces found")
            
            # Show the result image for a brief moment
            if 'image' in result:
                cv2.imshow("Test Result", result['image'])
                print("  Showing result image - press any key to continue")
                cv2.waitKey(3000)  # Wait for 3 seconds or key press
                cv2.destroyAllWindows()
        else:
            print(f"  Image processing failed: {result.get('error')}")
    
    except Exception as e:
        log_exception(logger, "Unexpected error in image processing test", e)
        print(f"  Unexpected error: {e}")
        
    finally:
        # Make sure to close any OpenCV windows
        safely_close_windows()
    
    # Test deliberately failing operation
    print("\n6. Testing deliberately failing operation...")
    success, result = process_image_with_error_handling(
        os.path.join(test_dir, "test_image.jpg"), 
        operation_type="fail"
    )
    if success:
        print(f"  Operation succeeded (unexpected): {result}")
    else:
        print(f"  Operation failed as expected: {result.get('error')}")
    
    # Summary
    print("\n===== TEST COMPLETE =====")
    print("\nCheck the logs directory for detailed logs generated during this test:")
    for log_file in ["debug.log", "info.log", "error.log"]:
        log_path = os.path.join(project_root, "logs", log_file)
        if os.path.exists(log_path):
            size = os.path.getsize(log_path)
            print(f"  - {log_file}: {size/1024:.1f} KB")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        log_exception(logger, "Fatal error in test script", e)
        print(f"Fatal error: {e}")
    finally:
        # Make sure to release any OpenCV resources
        safely_close_windows()
