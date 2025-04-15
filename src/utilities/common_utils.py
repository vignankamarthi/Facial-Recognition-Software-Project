"""
Common Utilities Module

This module provides centralized utility functions used across the facial recognition project.
It consolidates previously duplicated functionality for improved maintainability.
"""

import os
import sys
import cv2
import time
import shutil
import subprocess
from pathlib import Path


# ===== Path Configuration =====

# Import centralized configuration
try:
    from .config import (
        PROJECT_ROOT, DATA_DIR, KNOWN_FACES_DIR, TEST_DATASETS_DIR, 
        RESULTS_DIR, ensure_dir_exists
    )
except ImportError:
    # Fallback if config module is not available
    def get_project_root():
        """Get the absolute path to the project root directory."""
        # Navigate up from this file's location to find project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.dirname(os.path.dirname(current_dir))  # up two levels
    
    PROJECT_ROOT = get_project_root()
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
    TEST_DATASETS_DIR = os.path.join(DATA_DIR, "test_datasets")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    
    def ensure_dir_exists(directory_path):
        """Ensure a directory exists, creating it if necessary."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

def get_project_root():
    """Get the absolute path to the project root directory."""
    return PROJECT_ROOT

def get_data_dir():
    """Get the absolute path to the data directory."""
    return DATA_DIR

def get_known_faces_dir():
    """Get the path to the known faces directory, ensuring it exists."""
    return ensure_dir_exists(KNOWN_FACES_DIR)

def get_test_datasets_dir():
    """Get the path to the test datasets directory, ensuring it exists."""
    return ensure_dir_exists(TEST_DATASETS_DIR)

def get_results_dir():
    """Get the path to the results directory, ensuring it exists."""
    return ensure_dir_exists(RESULTS_DIR)


# ===== OpenCV Window Management =====

def create_resizable_window(window_name):
    """Create a named OpenCV window and set it to be resizable."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Ensure it gets focus
    return window_name

def safely_close_windows(window_name=None, video_capture=None):
    """
    Safely close OpenCV windows and release video capture resources with multiple attempts.
    
    Args:
        window_name (str, optional): Specific window to close, or None to close all windows
        video_capture (cv2.VideoCapture, optional): Video capture object to release
    """
    print("Cleaning up resources...")
    
    # Release video capture if provided
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
        print("Video capture released.")
    
    print("Closing OpenCV windows...")
    
    # Try to get focus first
    if window_name:
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # Window might already be closed
    
    cv2.waitKey(200)  # Wait for focus
    
    # First close attempt
    try:
        if window_name:
            cv2.destroyWindow(window_name)
        else:
            cv2.destroyAllWindows()
    except:
        pass  # Ignore errors in window closing
    
    time.sleep(0.2)  # Give time for windows to close
    
    # Second attempt with all windows
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    time.sleep(0.2)
    
    # Third attempt with a loop and delays
    for i in range(3):
        cv2.waitKey(200)  # Longer wait
        try:
            cv2.destroyAllWindows()
        except:
            pass
        time.sleep(0.2)  # Direct sleep
    
    print("Windows closed.")


# ===== Error Handling =====

# ===== Enhanced Error Handling =====

class FaceRecognitionError(Exception):
    """Base exception class for facial recognition errors."""
    pass

class CameraError(FaceRecognitionError):
    """Exception raised for camera-related errors."""
    pass

class DetectionError(FaceRecognitionError):
    """Exception raised for face detection errors."""
    pass

class MatchingError(FaceRecognitionError):
    """Exception raised for face matching errors."""
    pass

class AnonymizationError(FaceRecognitionError):
    """Exception raised for face anonymization errors."""
    pass

class DatasetError(FaceRecognitionError):
    """Exception raised for dataset-related errors."""
    pass

def handle_opencv_error(func):
    """
    Decorator to handle OpenCV errors gracefully.
    
    Example usage:
        @handle_opencv_error
        def process_frame(frame):
            # Process the frame...
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cv2.error as e:
            print(f"OpenCV error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except CameraError as e:
            print(f"Camera error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except FaceRecognitionError as e:
            print(f"Facial recognition error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            safely_close_windows()
            return None
    return wrapper

def format_error(error_type, message):
    """
    Format error messages consistently.
    
    Args:
        error_type (str): Type of error (e.g., "Camera", "Detection")
        message (str): Error message
        
    Returns:
        str: Formatted error message
    """
    return f"ERROR [{error_type}]: {message}"


# ===== File Operations =====

def clean_directory(directory_path, pattern=None):
    """
    Clean up files in a directory, optionally matching a pattern.
    
    Args:
        directory_path (str): Path to clean
        pattern (str, optional): Glob pattern to match files
    
    Returns:
        int: Number of files deleted
    """
    if not os.path.exists(directory_path):
        return 0
        
    count = 0
    
    if pattern:
        for path in Path(directory_path).glob(pattern):
            if path.is_file():
                path.unlink()
                count += 1
    else:
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
                count += 1
    
    return count


# ===== Process Management =====

def run_command(command):
    """
    Run a shell command and print its output.
    
    Args:
        command (str): Command to run
        
    Returns:
        bool: True if successful (return code 0), False otherwise
    """
    print(f"Executing: {command}")
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line.strip())
        
        for line in process.stderr:
            print(f"ERROR: {line.strip()}")
            
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


# ===== Progress Display =====

class ProgressBar:
    """
    A simple progress bar for console output.
    
    Example usage:
        progress = ProgressBar(total=100, prefix='Processing:', suffix='Complete', length=50)
        for i in range(100):
            # Do work...
            progress.update(i + 1)
    """
    def __init__(self, total, prefix='', suffix='', length=50, fill='█', print_end='\r'):
        """Initialize the progress bar."""
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.start_time = time.time()
        self.update(0)
    
    def update(self, iteration):
        """Update the progress bar."""
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '░' * (self.length - filled_length)
        
        # Calculate speed and ETA
        elapsed = time.time() - self.start_time
        items_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        if items_per_sec > 0 and iteration < self.total:
            remaining = self.total - iteration
            eta_seconds = remaining / items_per_sec
            eta = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
        else:
            elapsed_str = f"Time: {int(elapsed//60)}m {int(elapsed%60)}s"
            eta = elapsed_str
        
        # Print the progress bar
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} | {iteration}/{self.total} | {items_per_sec:.1f} it/s | {eta}', end=self.print_end)
        
        # Print a newline when complete
        if iteration == self.total:
            print()
