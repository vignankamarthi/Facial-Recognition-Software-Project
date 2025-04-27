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
        RESULTS_DIR, DATASETS_DIR, UTKFACE_DIR, DEMOGRAPHIC_SPLIT_SET_DIR,
        ensure_dir_exists
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
    DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
    UTKFACE_DIR = os.path.join(DATASETS_DIR, "utkface")
    DEMOGRAPHIC_SPLIT_SET_DIR = os.path.join(TEST_DATASETS_DIR, "demographic_split_set")
    
    def ensure_dir_exists(directory_path):
        """Ensure a directory exists, creating it if necessary."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return directory_path

# Path utility functions
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
    
def get_datasets_dir():
    """Get the path to the datasets directory, ensuring it exists."""
    return ensure_dir_exists(DATASETS_DIR)
    
def get_utkface_dir():
    """Get the path to the UTKFace dataset directory, ensuring it exists."""
    return ensure_dir_exists(UTKFACE_DIR)
    
def get_demographic_split_dir():
    """Get the path to the demographic split directory, ensuring it exists."""
    return ensure_dir_exists(DEMOGRAPHIC_SPLIT_SET_DIR)
    
def is_image_file(filename):
    """Check if a filename is an image file based on extension.
    
    Args:
        filename (str): Filename to check
        
    Returns:
        bool: True if the file has an image extension, False otherwise
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    return filename.lower().endswith(image_extensions)

def get_image_files(directory_path):
    """Get all image files in a directory.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        list: List of full paths to image files
    """
    if not os.path.exists(directory_path):
        return []
        
    return [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if is_image_file(filename)
    ]
    
def get_subdirectories(directory_path):
    """Get all subdirectories in a directory.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        list: List of subdirectory names
    """
    if not os.path.exists(directory_path):
        return []
        
    return [
        name for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]


# ===== OpenCV Window Management =====

def create_resizable_window(window_name, width=800, height=600):
    """
    Create a named OpenCV window and set it to be resizable with optional size.
    
    Args:
        window_name (str): Name of the window to create
        width (int, optional): Initial window width (default: 800)
        height (int, optional): Initial window height (default: 600)
        
    Returns:
        str: The window name
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Ensure it gets focus
    cv2.resizeWindow(window_name, width, height)  # Set initial size
    return window_name

def create_control_window(window_name, width=400, height=200):
    """
    Create a window for controls and UI elements.
    
    Args:
        window_name (str): Name of the window to create
        width (int, optional): Initial window width (default: 400)
        height (int, optional): Initial window height (default: 200)
        
    Returns:
        str: The window name
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    return window_name

def safely_close_windows(window_name=None, video_capture=None, verbose=True):
    """
    Safely close OpenCV windows and release video capture resources with multiple attempts.
    
    Args:
        window_name (str, optional): Specific window to close, or None to close all windows
        video_capture (cv2.VideoCapture, optional): Video capture object to release
        verbose (bool, optional): Whether to print status messages (default: True)
    """
    if verbose:
        print("Cleaning up resources...")
    
    # Release video capture if provided
    if video_capture is not None and video_capture.isOpened():
        video_capture.release()
        if verbose:
            print("Video capture released.")
    
    if verbose:
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
    
    if verbose:
        print("Windows closed.")
        
def show_image_with_delay(image, window_name="Image", delay=0, resize=True, max_dim=800):
    """
    Display an image with optional resize and wait for a key press or delay.
    
    Args:
        image (numpy.ndarray): Image to display
        window_name (str, optional): Name of the window (default: "Image")
        delay (int, optional): Delay in milliseconds, 0 means wait for key (default: 0)
        resize (bool, optional): Whether to resize large images (default: True)
        max_dim (int, optional): Maximum dimension for resizing (default: 800)
        
    Returns:
        int: Key code pressed, or -1 if no key was pressed
    """
    if image is None:
        print(f"Warning: Cannot display None image in window '{window_name}'")
        return -1
        
    display_img = image.copy()
    
    # Resize large images for display if needed
    if resize:
        h, w = display_img.shape[:2]
        if max(h, w) > max_dim:
            # Calculate new dimensions
            if h > w:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            else:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
                
            # Resize the image for display
            display_img = cv2.resize(display_img, (new_w, new_h))
    
    # Create window and show image
    create_resizable_window(window_name)
    cv2.imshow(window_name, display_img)
    
    # Wait for key press or delay
    return cv2.waitKey(delay) & 0xFF


# ===== Error Handling =====

class FaceRecognitionError(Exception):
    """Base exception class for facial recognition errors."""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(self.format_message())
        
    def format_message(self):
        """Format the error message with optional details."""
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message

# Core functionality exceptions
class CameraError(FaceRecognitionError):
    """Exception raised for camera-related errors (e.g., can't open webcam)."""
    pass

class DetectionError(FaceRecognitionError):
    """Exception raised for face detection errors (e.g., face detection failed)."""
    pass

class MatchingError(FaceRecognitionError):
    """Exception raised for face matching errors (e.g., no known faces found)."""
    pass

class AnonymizationError(FaceRecognitionError):
    """Exception raised for face anonymization errors (e.g., invalid method)."""
    pass

# Data handling exceptions
class DatasetError(FaceRecognitionError):
    """Exception raised for dataset-related errors (e.g., can't download dataset)."""
    pass

class FileError(FaceRecognitionError):
    """Exception raised for file operations errors (e.g., can't read/write files)."""
    pass

class ConfigurationError(FaceRecognitionError):
    """Exception raised for configuration errors (e.g., invalid settings)."""
    pass

# System-level exceptions
class DependencyError(FaceRecognitionError):
    """Exception raised for missing dependencies (e.g., face_recognition not available)."""
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
            error_msg = f"OpenCV error in {func.__name__}: {e}"
            print(error_msg)
            safely_close_windows()
            return None
        except CameraError as e:
            print(f"Camera error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except (DetectionError, MatchingError, AnonymizationError) as e:
            print(f"Facial recognition error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except (DatasetError, FileError) as e:
            print(f"Data error in {func.__name__}: {e}")
            safely_close_windows()
            return None
        except Exception as e:
            print(f"Unexpected error in {func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            safely_close_windows()
            return None
    return wrapper

def format_error(error_type, message, details=None):
    """
    Format error messages consistently.
    
    Args:
        error_type (str): Type of error (e.g., "Camera", "Detection")
        message (str): Error message
        details (str, optional): Additional error details
        
    Returns:
        str: Formatted error message
    """
    error_msg = f"ERROR [{error_type}]: {message}"
    if details:
        error_msg += f" - {details}"
    return error_msg


# ===== File Operations =====

def clean_directory(directory_path, pattern=None, recursive=False, keep_directory=True):
    """
    Clean up files in a directory, optionally matching a pattern.
    
    Args:
        directory_path (str): Path to clean
        pattern (str, optional): Global pattern to match files
        recursive (bool, optional): Whether to clean subdirectories
        keep_directory (bool, optional): Whether to keep directory structure
    
    Returns:
        int: Number of files deleted
    """
    try:
        if not os.path.exists(directory_path):
            return 0
            
        count = 0
        
        if recursive:
            if pattern:
                for path in Path(directory_path).glob(f"**/{pattern}"):
                    if path.is_file():
                        path.unlink()
                        count += 1
            else:
                # Delete all files but keep directory structure
                if keep_directory:
                    for root, dirs, files in os.walk(directory_path, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                            count += 1
                # Delete entire directory and contents
                else:
                    if os.path.exists(directory_path):
                        shutil.rmtree(directory_path)
                        os.makedirs(directory_path)  # Recreate empty directory
                        count += 1  # Count as one operation
        else:
            # Non-recursive cleanup
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
    except Exception as e:
        print(f"Error cleaning directory {directory_path}: {e}")
        raise FileError(f"Failed to clean directory", str(e))

def safe_copy_file(src, dst, overwrite=False):
    """
    Safely copy a file with additional error handling.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
        overwrite (bool, optional): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, raises FileError otherwise
    """
    try:
        # Check if source exists
        if not os.path.exists(src):
            raise FileError(f"Source file does not exist: {src}")
            
        # Check if destination directory exists, create if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        # Check if destination exists and overwrite is disabled
        if os.path.exists(dst) and not overwrite:
            # Generate alternative filename with timestamp
            base, ext = os.path.splitext(dst)
            timestamp = int(time.time())
            dst = f"{base}_{timestamp}{ext}"
            
        # Copy file
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        raise FileError(f"Failed to copy file from {src} to {dst}", str(e))

def safe_move_file(src, dst, overwrite=False):
    """
    Safely move a file with additional error handling.
    
    Args:
        src (str): Source file path
        dst (str): Destination file path
        overwrite (bool, optional): Whether to overwrite existing files
        
    Returns:
        bool: True if successful, raises FileError otherwise
    """
    try:
        # Check if source exists
        if not os.path.exists(src):
            raise FileError(f"Source file does not exist: {src}")
            
        # Check if destination directory exists, create if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            
        # Check if destination exists and overwrite is disabled
        if os.path.exists(dst) and not overwrite:
            # Generate alternative filename with timestamp
            base, ext = os.path.splitext(dst)
            timestamp = int(time.time())
            dst = f"{base}_{timestamp}{ext}"
            
        # Move file
        shutil.move(src, dst)
        return True
    except Exception as e:
        raise FileError(f"Failed to move file from {src} to {dst}", str(e))


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
