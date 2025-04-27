"""
Logging Module

This module provides a centralized logging system for the facial recognition project.
It configures different loggers for various components, handles log rotation,
and ensures consistent error reporting throughout the application.

All logs are automatically saved to the logs directory, organized by log level:
- error.log: Contains ERROR and CRITICAL level messages
- info.log: Contains INFO level and above messages
- debug.log: Contains DEBUG level and above messages

Functions and Classes
-------------------
get_logger : Get a logger for a specific module
log_exception : Log an exception with context information
log_method_call : Decorator to log method calls
set_log_level : Change the logging level for console output
get_all_logs : Get the contents of all log files
clear_logs : Clear all log files

Examples
--------
>>> from src.utils.logger import get_logger, log_exception, log_method_call
>>> logger = get_logger(__name__)
>>> logger.info("Application started")
>>>
>>> try:
...     result = process_image("invalid.jpg")
... except Exception as e:
...     log_exception(logger, "Failed to process image", e)
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import traceback
import inspect
from pathlib import Path
from datetime import datetime

# Determine project root and logs directory
def get_project_root():
    """Get the absolute path to the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))
        
PROJECT_ROOT = get_project_root()

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

# Log file paths
ERROR_LOG = os.path.join(LOGS_DIR, "error.log")
INFO_LOG = os.path.join(LOGS_DIR, "info.log")
DEBUG_LOG = os.path.join(LOGS_DIR, "debug.log")

# Maximum log file size (10 MB)
MAX_LOG_SIZE = 10 * 1024 * 1024 # 10 * 2^10 * 2^10 = 10 MB
# Number of backup log files to keep
BACKUP_COUNT = 5

# Log format with contextual information
LOG_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Configure the root logger
def configure_root_logger():
    """Configure the root logger with console and file handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers if any
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Console handler - only show INFO and above to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Make sure logs directory exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    
    # File handlers with rotation
    
    # Debug log - captures all messages
    debug_handler = RotatingFileHandler(
        DEBUG_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    debug_handler.setFormatter(debug_formatter)
    root_logger.addHandler(debug_handler)
    
    # Info log - captures INFO and above
    info_handler = RotatingFileHandler(
        INFO_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    info_handler.setLevel(logging.INFO)
    info_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    info_handler.setFormatter(info_formatter)
    root_logger.addHandler(info_handler)
    
    # Error log - captures ERROR and above
    error_handler = RotatingFileHandler(
        ERROR_LOG, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Log initialization 
    root_logger.info(f"Logging initialized. Log files will be saved to: {LOGS_DIR}")
    root_logger.info(f"Debug log: {DEBUG_LOG}")
    root_logger.info(f"Info log: {INFO_LOG}")
    root_logger.info(f"Error log: {ERROR_LOG}")
    
    return root_logger

# Configure the root logger at module import
root_logger = configure_root_logger()

def get_logger(name):
    """
    Get a logger configured for a specific module.
    
    Parameters
    ----------
    name : str
        The name of the module, typically __name__
        
    Returns
    -------
    logging.Logger
        A configured logger instance
    
    Examples
    --------
    >>> # In file src/core/face_detection.py
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Face detection started")
    """
    return logging.getLogger(name)

def log_exception(logger, message, exception=None, level=logging.ERROR):
    """
    Log an exception with full traceback and context information.
    
    Parameters
    ----------
    logger : logging.Logger
        The logger to use
    message : str
        The message to log
    exception : Exception, optional
        The exception to log. If None, the current exception is used.
    level : int, optional
        The logging level to use (default: logging.ERROR)
    
    Examples
    --------
    >>> try:
    ...     # Some code that might raise an exception
    ...     result = process_image("invalid_path.jpg")
    ... except Exception as e:
    ...     log_exception(logger, "Failed to process image", e)
    """
    if exception is None:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_type is None:
            # No current exception
            logger.log(level, message)
            return
        exception = exc_value
    
    # Get the stack frame where the exception occurred
    tb = traceback.extract_tb(exception.__traceback__)
    if tb:
        frame_info = tb[-1]  # Last frame in the traceback
        file_path = frame_info.filename
        line_number = frame_info.lineno
        function_name = frame_info.name
    else:
        # If traceback extraction fails, get current frame info
        frame = inspect.currentframe().f_back
        file_path = frame.f_code.co_filename
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
    
    # Format the error message with context
    context_msg = f"{message} in {function_name} at {Path(file_path).name}:{line_number}"
    
    # Log the error message with the exception details
    logger.log(level, context_msg, exc_info=True)

def log_method_call(logger, level=logging.DEBUG):
    """
    Decorator to log method calls with parameters and return values.
    
    Parameters
    ----------
    logger : logging.Logger
        The logger to use
    level : int, optional
        The logging level to use (default: logging.DEBUG)
    
    Returns
    -------
    callable
        A decorator function
    
    Examples
    --------
    >>> from src.utils.logger import get_logger, log_method_call
    >>> logger = get_logger(__name__)
    >>> 
    >>> class MyClass:
    ...     @log_method_call(logger)
    ...     def my_method(self, param1, param2=None):
    ...         return param1 + (param2 or 0)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get the class name if it's a method
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                args_str = ", ".join([repr(arg) for arg in args[1:]])
            else:
                class_name = ""
                args_str = ", ".join([repr(arg) for arg in args])
            
            # Format kwargs
            kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
            
            # Combine args and kwargs
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            # Function name with optional class
            if class_name:
                func_name = f"{class_name}.{func.__name__}"
            else:
                func_name = func.__name__
            
            # Log method call
            logger.log(level, f"Calling {func_name}({params_str})")
            
            try:
                # Call the function
                result = func(*args, **kwargs)
                
                # Log the result
                result_repr = repr(result)
                # Truncate if too long
                if len(result_repr) > 100:
                    result_repr = result_repr[:97] + "..."
                logger.log(level, f"{func_name} returned: {result_repr}")
                
                return result
            except Exception as e:
                # Log the exception and re-raise
                log_exception(logger, f"Exception in {func_name}", e)
                raise
        
        # Preserve the original function's metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator

def set_log_level(level):
    """
    Set the log level for the console handler.
    
    Parameters
    ----------
    level : int or str
        The logging level to set (e.g., logging.DEBUG, 'DEBUG')
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Set level for the console handler
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(level)
            root_logger.info(f"Console log level set to {logging.getLevelName(level)}")
            break

def get_all_logs():
    """
    Get the contents of all log files as a dictionary.
    
    Returns
    -------
    dict
        A dictionary with log names as keys and log contents as values
    """
    log_files = {
        "error": ERROR_LOG,
        "info": INFO_LOG,
        "debug": DEBUG_LOG
    }
    
    logs_content = {}
    for name, path in log_files.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    logs_content[name] = f.read()
            else:
                logs_content[name] = f"Log file {path} does not exist yet."
        except Exception as e:
            logs_content[name] = f"Error reading log file: {str(e)}"
    
    return logs_content

def clear_logs():
    """
    Clear all log files.
    
    Returns
    -------
    bool
        True if logs were cleared successfully, False otherwise
    """
    log_files = [ERROR_LOG, INFO_LOG, DEBUG_LOG]
    
    try:
        for path in log_files:
            if os.path.exists(path):
                with open(path, 'w') as f:
                    f.write(f"Log cleared at {datetime.now().strftime(DATE_FORMAT)}\n")
                    
        root_logger.info("All log files have been cleared")
        return True
    except Exception as e:
        root_logger.error(f"Error clearing logs: {e}")
        return False

# Example usage
if __name__ == "__main__":
    logger = get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test exception logging
    try:
        x = 1 / 0
    except Exception as e:
        log_exception(logger, "Division by zero", e)
    
    # Show where logs are stored
    print(f"Log files are stored in: {LOGS_DIR}")
    print(f"Debug log: {DEBUG_LOG}")
    print(f"Info log: {INFO_LOG}")
    print(f"Error log: {ERROR_LOG}")
