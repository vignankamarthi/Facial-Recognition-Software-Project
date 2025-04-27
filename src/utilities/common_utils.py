"""
Common Utilities Module

This module provides centralized utility functions used across the facial recognition project.
It consolidates previously duplicated functionality for improved maintainability.

This module includes standardized functions for path management, window handling,
error handling, file operations, process management, and progress display.
"""

import os
import sys
import cv2
import time
import shutil
import subprocess
import traceback
from pathlib import Path

# Import our logging system
from .logger import get_logger, log_exception, log_method_call

# Initialize logger for this module
logger = get_logger(__name__)


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
    """
    Get the absolute path to the project root directory.
    
    Returns
    -------
    str
        Absolute path to the project root directory
    """
    return PROJECT_ROOT

def get_data_dir():
    """
    Get the absolute path to the data directory.
    
    Returns
    -------
    str
        Absolute path to the data directory
    """
    return DATA_DIR

def get_known_faces_dir():
    """
    Get the path to the known faces directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the known faces directory
    """
    path = ensure_dir_exists(KNOWN_FACES_DIR)
    logger.debug(f"Known faces directory: {path}")
    return path

def get_test_datasets_dir():
    """
    Get the path to the test datasets directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the test datasets directory
    """
    path = ensure_dir_exists(TEST_DATASETS_DIR)
    logger.debug(f"Test datasets directory: {path}")
    return path

def get_results_dir():
    """
    Get the path to the results directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the results directory
    """
    path = ensure_dir_exists(RESULTS_DIR)
    logger.debug(f"Results directory: {path}")
    return path
    
def get_datasets_dir():
    """
    Get the path to the datasets directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the datasets directory
    """
    path = ensure_dir_exists(DATASETS_DIR)
    logger.debug(f"Datasets directory: {path}")
    return path
    
def get_utkface_dir():
    """
    Get the path to the UTKFace dataset directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the UTKFace directory
    """
    path = ensure_dir_exists(UTKFACE_DIR)
    logger.debug(f"UTKFace directory: {path}")
    return path
    
def get_demographic_split_dir():
    """
    Get the path to the demographic split directory, ensuring it exists.
    
    Returns
    -------
    str
        Absolute path to the demographic split directory
    """
    path = ensure_dir_exists(DEMOGRAPHIC_SPLIT_SET_DIR)
    logger.debug(f"Demographic split directory: {path}")
    return path
    
def is_image_file(filename):
    """
    Check if a filename is an image file based on extension.
    
    Parameters
    ----------
    filename : str
        Filename to check
        
    Returns
    -------
    bool
        True if the file has an image extension, False otherwise
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")
    return filename.lower().endswith(image_extensions)

def get_image_files(directory_path):
    """
    Get all image files in a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory
        
    Returns
    -------
    list
        List of full paths to image files
    """
    if not os.path.exists(directory_path):
        logger.warning(f"Directory does not exist: {directory_path}")
        return []
        
    image_files = [
        os.path.join(directory_path, filename)
        for filename in os.listdir(directory_path)
        if is_image_file(filename)
    ]
    
    logger.debug(f"Found {len(image_files)} image files in {directory_path}")
    return image_files
    
def get_subdirectories(directory_path):
    """
    Get all subdirectories in a directory.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory
        
    Returns
    -------
    list
        List of subdirectory names
    """
    if not os.path.exists(directory_path):
        logger.warning(f"Directory does not exist: {directory_path}")
        return []
        
    subdirs = [
        name for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    ]
    
    logger.debug(f"Found {len(subdirs)} subdirectories in {directory_path}")
    return subdirs


# ===== OpenCV Window Management =====

def create_resizable_window(window_name, width=800, height=600):
    """
    Create a named OpenCV window and set it to be resizable with optional size.
    
    Parameters
    ----------
    window_name : str
        Name of the window to create
    width : int, optional
        Initial window width (default: 800)
    height : int, optional
        Initial window height (default: 600)
        
    Returns
    -------
    str
        The window name
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Ensure it gets focus
    cv2.resizeWindow(window_name, width, height)  # Set initial size
    logger.debug(f"Created resizable window: {window_name} ({width}x{height})")
    return window_name

def create_control_window(window_name, width=400, height=200):
    """
    Create a window for controls and UI elements.
    
    Parameters
    ----------
    window_name : str
        Name of the window to create
    width : int, optional
        Initial window width (default: 400)
    height : int, optional
        Initial window height (default: 200)
        
    Returns
    -------
    str
        The window name
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    logger.debug(f"Created control window: {window_name} ({width}x{height})")
    return window_name

@log_method_call(logger)
def safely_close_windows(window_name=None, video_capture=None, verbose=True):
    """
    Safely close OpenCV windows and release video capture resources with multiple attempts.
    
    Parameters
    ----------
    window_name : str, optional
        Specific window to close, or None to close all windows
    video_capture : cv2.VideoCapture, optional
        Video capture object to release
    verbose : bool, optional
        Whether to print status messages (default: True)
    """
    if verbose:
        logger.info("Cleaning up resources...")
    
    # Release video capture if provided
    if video_capture is not None:
        try:
            if video_capture.isOpened():
                video_capture.release()
                if verbose:
                    logger.info("Video capture released.")
        except Exception as e:
            logger.warning(f"Error releasing video capture: {e}")
    
    if verbose:
        logger.info("Closing OpenCV windows...")
    
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
    except Exception as e:
        logger.debug(f"Error in first window close attempt: {e}")
    
    time.sleep(0.2)  # Give time for windows to close
    
    # Second attempt with all windows
    try:
        cv2.destroyAllWindows()
    except Exception as e:
        logger.debug(f"Error in second window close attempt: {e}")
    
    time.sleep(0.2)
    
    # Third attempt with a loop and delays
    for i in range(3):
        cv2.waitKey(200)  # Longer wait
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.debug(f"Error in third window close attempt {i+1}: {e}")
        time.sleep(0.2)  # Direct sleep
    
    if verbose:
        logger.info("Windows closed.")
        
@log_method_call(logger)
def show_image_with_delay(image, window_name="Image", delay=0, resize=True, max_dim=800):
    """
    Display an image with optional resize and wait for a key press or delay.
    
    Parameters
    ----------
    image : numpy.ndarray
        Image to display
    window_name : str, optional
        Name of the window (default: "Image")
    delay : int, optional
        Delay in milliseconds, 0 means wait for key (default: 0)
    resize : bool, optional
        Whether to resize large images (default: True)
    max_dim : int, optional
        Maximum dimension for resizing (default: 800)
        
    Returns
    -------
    int
        Key code pressed, or -1 if no key was pressed
    """
    if image is None:
        logger.warning(f"Cannot display None image in window '{window_name}'")
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
            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h} for display")
    
    # Create window and show image
    create_resizable_window(window_name)
    cv2.imshow(window_name, display_img)
    
    # Wait for key press or delay
    key = cv2.waitKey(delay) & 0xFF
    if key != 255:  # If a key was pressed
        logger.debug(f"Key pressed in window '{window_name}': {key}")
    
    return key


# ===== Error Handling =====

class FaceRecognitionError(Exception):
    """Base exception class for facial recognition errors.
    
    Parameters
    ----------
    message : str
        The error message
    details : str, optional
        Additional error details
    source : Exception, optional
        The source exception, if this is wrapping another exception
    """
    def __init__(self, message, details=None, source=None):
        self.message = message
        self.details = details
        self.source = source
        
        # Format the message
        formatted_message = self.format_message()
        
        # Log the error with context information
        log_exception(logger, formatted_message, source)
        
        # Initialize the parent Exception class with the formatted message
        super().__init__(formatted_message)
        
        # Keep the original traceback if wrapping another exception
        if source and hasattr(source, '__traceback__'):
            self.__cause__ = source
        
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
    
    This decorator catches and logs exceptions that occur in the decorated function,
    properly cleans up OpenCV resources, and returns appropriate error values.
    
    Parameters
    ----------
    func : callable
        The function to decorate
    
    Returns
    -------
    callable
        The decorated function
    
    Examples
    --------
    >>> @handle_opencv_error
    >>> def process_frame(frame):
    ...     # Process the frame...
    ...     return processed_frame
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except cv2.error as e:
            error_msg = f"OpenCV error in {func.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            safely_close_windows()
            return None, {"error": error_msg, "type": "OpenCV"}
        except CameraError as e:
            error_msg = f"Camera error in {func.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            safely_close_windows()
            return None, {"error": str(e), "type": "Camera"}
        except (DetectionError, MatchingError, AnonymizationError) as e:
            error_msg = f"Facial recognition error in {func.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            safely_close_windows()
            return None, {"error": str(e), "type": "FacialRecognition"}
        except (DatasetError, FileError) as e:
            error_msg = f"Data error in {func.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            safely_close_windows()
            return None, {"error": str(e), "type": "Data"}
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            safely_close_windows()
            return None, {"error": str(e), "type": "Unexpected"}
    
    # Preserve the original function's metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    
    return wrapper

def format_error(error_type, message, details=None):
    """
    Format error messages consistently.
    
    Parameters
    ----------
    error_type : str
        Type of error (e.g., "Camera", "Detection")
    message : str
        Error message
    details : str, optional
        Additional error details
        
    Returns
    -------
    str
        Formatted error message
    """
    error_msg = f"ERROR [{error_type}]: {message}"
    if details:
        error_msg += f" - {details}"
    
    # Log the formatted error
    logger.error(error_msg)
    
    return error_msg


# ===== File Operations =====

@log_method_call(logger)
def clean_directory(directory_path, pattern=None, recursive=False, keep_directory=True):
    """
    Clean up files in a directory, optionally matching a pattern.
    
    Parameters
    ----------
    directory_path : str
        Path to clean
    pattern : str, optional
        Global pattern to match files
    recursive : bool, optional
        Whether to clean subdirectories
    keep_directory : bool, optional
        Whether to keep directory structure
    
    Returns
    -------
    int
        Number of files deleted
    
    Raises
    ------
    FileError
        If directory cleaning fails
    """
    try:
        if not os.path.exists(directory_path):
            logger.warning(f"Directory does not exist: {directory_path}")
            return 0
            
        count = 0
        logger.info(f"Cleaning directory: {directory_path} (pattern={pattern}, recursive={recursive})")
        
        if recursive:
            if pattern:
                for path in Path(directory_path).glob(f"**/{pattern}"):
                    if path.is_file():
                        path.unlink()
                        logger.debug(f"Deleted file: {path}")
                        count += 1
            else:
                # Delete all files but keep directory structure
                if keep_directory:
                    for root, dirs, files in os.walk(directory_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            logger.debug(f"Deleted file: {file_path}")
                            count += 1
                # Delete entire directory and contents
                else:
                    if os.path.exists(directory_path):
                        shutil.rmtree(directory_path)
                        os.makedirs(directory_path)  # Recreate empty directory
                        logger.info(f"Recreated directory: {directory_path}")
                        count += 1  # Count as one operation
        else:
            # Non-recursive cleanup
            if pattern:
                for path in Path(directory_path).glob(pattern):
                    if path.is_file():
                        path.unlink()
                        logger.debug(f"Deleted file: {path}")
                        count += 1
            else:
                for item in os.listdir(directory_path):
                    item_path = os.path.join(directory_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                        logger.debug(f"Deleted file: {item_path}")
                        count += 1
        
        logger.info(f"Cleaned {count} items from {directory_path}")
        return count
    except Exception as e:
        error_msg = f"Error cleaning directory {directory_path}"
        log_exception(logger, error_msg, e)
        raise FileError(error_msg, str(e), e)

@log_method_call(logger)
def safe_copy_file(src, dst, overwrite=False):
    """
    Safely copy a file with additional error handling.
    
    Parameters
    ----------
    src : str
        Source file path
    dst : str
        Destination file path
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns
    -------
    str
        Path to the copied file (might be different from dst if renamed)
    
    Raises
    ------
    FileError
        If file copying fails
    """
    try:
        # Check if source exists
        if not os.path.exists(src):
            error_msg = f"Source file does not exist: {src}"
            logger.error(error_msg)
            raise FileError(error_msg)
            
        # Check if destination directory exists, create if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            logger.debug(f"Created directory: {dst_dir}")
            
        # Check if destination exists and overwrite is disabled
        final_dst = dst
        if os.path.exists(dst) and not overwrite:
            # Generate alternative filename with timestamp
            base, ext = os.path.splitext(dst)
            timestamp = int(time.time())
            final_dst = f"{base}_{timestamp}{ext}"
            logger.info(f"Destination file exists, using alternative: {final_dst}")
            
        # Copy file
        shutil.copy2(src, final_dst)
        logger.info(f"Copied file from {src} to {final_dst}")
        return final_dst
    except Exception as e:
        error_msg = f"Failed to copy file from {src} to {dst}"
        log_exception(logger, error_msg, e)
        raise FileError(error_msg, str(e), e)

@log_method_call(logger)
def safe_move_file(src, dst, overwrite=False):
    """
    Safely move a file with additional error handling.
    
    Parameters
    ----------
    src : str
        Source file path
    dst : str
        Destination file path
    overwrite : bool, optional
        Whether to overwrite existing files
        
    Returns
    -------
    str
        Path to the moved file (might be different from dst if renamed)
    
    Raises
    ------
    FileError
        If file moving fails
    """
    try:
        # Check if source exists
        if not os.path.exists(src):
            error_msg = f"Source file does not exist: {src}"
            logger.error(error_msg)
            raise FileError(error_msg)
            
        # Check if destination directory exists, create if needed
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            logger.debug(f"Created directory: {dst_dir}")
            
        # Check if destination exists and overwrite is disabled
        final_dst = dst
        if os.path.exists(dst) and not overwrite:
            # Generate alternative filename with timestamp
            base, ext = os.path.splitext(dst)
            timestamp = int(time.time())
            final_dst = f"{base}_{timestamp}{ext}"
            logger.info(f"Destination file exists, using alternative: {final_dst}")
            
        # Move file
        shutil.move(src, final_dst)
        logger.info(f"Moved file from {src} to {final_dst}")
        return final_dst
    except Exception as e:
        error_msg = f"Failed to move file from {src} to {dst}"
        log_exception(logger, error_msg, e)
        raise FileError(error_msg, str(e), e)


# ===== Process Management =====

@log_method_call(logger)
def run_command(command):
    """
    Run a shell command and print its output.
    
    Parameters
    ----------
    command : str
        Command to run
        
    Returns
    -------
    tuple
        (bool, dict) tuple containing success status and command output
        
    Examples
    --------
    >>> success, result = run_command("ls -la")
    >>> if success:
    ...     print(f"Command executed successfully with output: {result['stdout']}")
    ... else:
    ...     print(f"Command failed with error: {result['stderr']}")
    """
    logger.info(f"Executing command: {command}")
    
    try:
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout_lines = []
        for line in process.stdout:
            line = line.strip()
            stdout_lines.append(line)
            logger.info(f"STDOUT: {line}")
        
        stderr_lines = []
        for line in process.stderr:
            line = line.strip()
            stderr_lines.append(line)
            logger.error(f"STDERR: {line}")
            
        process.wait()
        
        success = process.returncode == 0
        result = {
            "returncode": process.returncode,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines)
        }
        
        if success:
            logger.info(f"Command executed successfully (return code: {process.returncode})")
        else:
            logger.error(f"Command failed with return code: {process.returncode}")
            
        return success, result
    except Exception as e:
        error_msg = f"Error executing command: {command}"
        log_exception(logger, error_msg, e)
        return False, {"returncode": -1, "stdout": "", "stderr": str(e)}


# ===== Progress Display =====

class ProgressBar:
    """
    A simple progress bar for console output.
    
    Parameters
    ----------
    total : int
        Total number of items
    prefix : str, optional
        Prefix string (default: '')
    suffix : str, optional
        Suffix string (default: '')
    length : int, optional
        Character length of the bar (default: 50)
    fill : str, optional
        Bar fill character (default: '█')
    print_end : str, optional
        End character (default: '\\r')
    
    Examples
    --------
    >>> progress = ProgressBar(total=100, prefix='Processing:', suffix='Complete', length=50)
    >>> for i in range(100):
    ...     # Do work...
    ...     progress.update(i + 1)
    """
    def __init__(self, total, prefix='', suffix='', length=50, fill='█', print_end='\r', log_level=logging.INFO):
        """Initialize the progress bar."""
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.log_level = log_level
        self.start_time = time.time()
        self.last_log_time = 0
        self.logger = get_logger(__name__)
        
        # Initial update
        self.update(0)
    
    def update(self, iteration):
        """
        Update the progress bar.
        
        Parameters
        ----------
        iteration : int
            Current iteration (0 to total)
        """
        # Only log every second to avoid excessive logging
        current_time = time.time()
        should_log = current_time - self.last_log_time >= 1.0 or iteration == self.total
        
        percent = ("{0:.1f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = self.fill * filled_length + '░' * (self.length - filled_length)
        
        # Calculate speed and ETA
        elapsed = current_time - self.start_time
        items_per_sec = iteration / elapsed if elapsed > 0 else 0
        
        if items_per_sec > 0 and iteration < self.total:
            remaining = self.total - iteration
            eta_seconds = remaining / items_per_sec
            eta = f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
        else:
            elapsed_str = f"Time: {int(elapsed//60)}m {int(elapsed%60)}s"
            eta = elapsed_str
        
        # Construct the progress bar string
        progress_str = f'\r{self.prefix} |{bar}| {percent}% {self.suffix} | {iteration}/{self.total} | {items_per_sec:.1f} it/s | {eta}'
        
        # Print to console
        print(progress_str, end=self.print_end)
        
        # Log to file if enough time has passed
        if should_log:
            self.logger.log(self.log_level, progress_str.replace('\r', ''))
            self.last_log_time = current_time
        
        # Print a newline when complete
        if iteration == self.total:
            print()
            self.logger.info(f"Progress complete: {self.prefix} {self.suffix} - {iteration}/{self.total} in {elapsed:.2f}s")
