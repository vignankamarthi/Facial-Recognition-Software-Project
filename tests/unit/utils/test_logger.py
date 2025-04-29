"""
Unit tests for the logger module.
"""
import pytest
import os
import logging
import time
import sys
import traceback
import inspect
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from datetime import datetime

from src.utils.logger import (
    get_logger,
    log_exception,
    log_method_call,
    set_log_level,
    get_all_logs,
    clear_logs,
    configure_root_logger,
    LOGS_DIR,
    ERROR_LOG,
    INFO_LOG,
    DEBUG_LOG,
    LOG_FORMAT,
    DATE_FORMAT
)

class TestLogger:
    """Tests for the logger module."""
    
    def test_get_logger(self):
        """Test getting a logger for a specific module."""
        # Get a logger for a test module name
        logger = get_logger("test.module")
        
        # Verify logger has correct name
        assert logger.name == "test.module"
        
        # Verify logger is properly configured
        assert logger.level <= logging.DEBUG
        
        # Get another logger with same name - should be the same instance
        logger2 = get_logger("test.module")
        assert logger is logger2
        
        # Get a logger with different name - should be different instance
        logger3 = get_logger("test.another_module")
        assert logger is not logger3
    
    @patch('logging.Logger.log')
    def test_log_exception(self, mock_log):
        """Test logging an exception with context information."""
        # Create a logger for testing
        test_logger = logging.getLogger("test.exception")
        
        # Create a test exception with traceback
        try:
            # Raise an exception to get a real traceback
            raise ValueError("Test exception message")
        except ValueError as e:
            # Log the exception
            log_exception(test_logger, "Test error occurred", e)
            
            # Verify logger.log was called with correct parameters
            mock_log.assert_called_once()
            
            # Extract the call arguments
            args, kwargs = mock_log.call_args
            
            # Verify the log level is ERROR
            assert args[0] == logging.ERROR
            
            # Verify the message includes the provided message
            assert "Test error occurred" in args[1]
            
            # Verify the message includes context info (function name and file)
            assert "in test_log_exception" in args[1]
            assert "test_logger.py" in args[1]
            
            # Verify exc_info is True to include traceback
            assert kwargs.get('exc_info') is True
            
        # Test with message only, no exception
        mock_log.reset_mock()
        log_exception(test_logger, "Just a message", level=logging.WARNING)
        
        # Verify logger.log was called with different level
        mock_log.assert_called_once()
        args, _ = mock_log.call_args
        assert args[0] == logging.WARNING
        assert args[1] == "Just a message"
    
    def test_log_method_call(self):
        """Test the log_method_call decorator."""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Create a test class with decorated method
        class TestClass:
            @log_method_call(mock_logger)
            def test_method(self, arg1, arg2=None):
                """Test method docstring."""
                return arg1 + (arg2 or 0)
                
            @log_method_call(mock_logger)
            def failing_method(self):
                """Method that raises an exception."""
                raise ValueError("Test failure")
        
        # Create an instance and call the method
        test_instance = TestClass()
        result = test_instance.test_method(5, arg2=10)
        
        # Verify the result is correct
        assert result == 15
        
        # Verify logger.log was called twice (entry and exit)
        assert mock_logger.log.call_count == 2
        
        # First call should log the method call with args
        args1, _ = mock_logger.log.call_args_list[0]
        assert args1[0] == logging.DEBUG  # Default level
        assert "Calling TestClass.test_method" in args1[1]
        assert "5" in args1[1]  # First arg
        assert "arg2=10" in args1[1]  # Keyword arg
        
        # Second call should log the return value
        args2, _ = mock_logger.log.call_args_list[1]
        assert args2[0] == logging.DEBUG
        assert "TestClass.test_method returned:" in args2[1]
        assert "15" in args2[1]  # Return value
        
        # Test exception handling
        mock_logger.reset_mock()
        
        # Patch log_exception to prevent actual exception handling
        with patch('src.utils.logger.log_exception') as mock_log_exception:
            # Call the failing method and catch the exception
            with pytest.raises(ValueError) as excinfo:
                test_instance.failing_method()
                
            # Verify the exception is re-raised with correct message
            assert "Test failure" in str(excinfo.value)
            
            # Verify log_exception was called
            mock_log_exception.assert_called_once()
            
            # Verify args to log_exception
            args, _ = mock_log_exception.call_args
            assert args[0] == mock_logger
            assert "Exception in TestClass.failing_method" in args[1]
    
    @patch('logging.StreamHandler.setLevel')
    def test_set_log_level(self, mock_set_level):
        """Test setting the log level for the console handler."""
        # Test with integer level
        set_log_level(logging.WARNING)
        mock_set_level.assert_called_with(logging.WARNING)
        
        # Reset the mock and test with string level
        mock_set_level.reset_mock()
        set_log_level("INFO")
        mock_set_level.assert_called_with(logging.INFO)
        
        # Test with invalid string level (should use default)
        with patch('logging.getLevelName') as mock_get_level_name:
            mock_set_level.reset_mock()
            with pytest.raises(AttributeError):  # getattr will fail with invalid level name
                set_log_level("INVALID_LEVEL")
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    def test_get_all_logs(self, mock_exists, mock_open):
        """Test getting the contents of all log files."""
        # Configure the mocks
        mock_exists.return_value = True
        
        # Create mock file contents
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = "Test log content"
        mock_open.return_value = mock_file
        
        # Call get_all_logs
        logs = get_all_logs()
        
        # Verify the result contains all log files
        assert "error" in logs
        assert "info" in logs
        assert "debug" in logs
        
        # Verify all logs have content
        assert logs["error"] == "Test log content"
        assert logs["info"] == "Test log content"
        assert logs["debug"] == "Test log content"
        
        # Verify open was called for each log file
        assert mock_open.call_count == 3
        mock_open.assert_any_call(ERROR_LOG, "r")
        mock_open.assert_any_call(INFO_LOG, "r")
        mock_open.assert_any_call(DEBUG_LOG, "r")
        
        # Test with non-existent files
        mock_exists.return_value = False
        mock_open.reset_mock()
        
        logs = get_all_logs()
        
        # Verify open was not called
        assert mock_open.call_count == 0
        
        # Verify logs contain appropriate messages
        for name, content in logs.items():
            assert "does not exist yet" in content
        
        # Test with exception during file reading
        mock_exists.return_value = True
        mock_open.side_effect = IOError("Test IO error")
        
        logs = get_all_logs()
        
        # Verify logs contain error messages
        for name, content in logs.items():
            assert "Error reading log file" in content
            assert "Test IO error" in content
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.path.exists')
    def test_clear_logs(self, mock_exists, mock_open):
        """Test clearing all log files."""
        # Configure the mocks
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_open.return_value = mock_file
        
        # Mock datetime.now for consistent testing
        with patch('src.utils.logger.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "2025-04-27 12:00:00"
            mock_datetime.now.return_value = mock_now
            
            # Call clear_logs
            result = clear_logs()
            
            # Verify the result is True (success)
            assert result is True
            
            # Verify open was called for each log file in write mode
            assert mock_open.call_count == 3
            mock_open.assert_any_call(ERROR_LOG, "w")
            mock_open.assert_any_call(INFO_LOG, "w")
            mock_open.assert_any_call(DEBUG_LOG, "w")
            
            # Verify write was called with the correct timestamp message
            expected_content = f"Log cleared at 2025-04-27 12:00:00\n"
            assert mock_file.__enter__.return_value.write.call_count == 3
            mock_file.__enter__.return_value.write.assert_called_with(expected_content)
        
        # Test with exception during file clearing
        mock_open.reset_mock()
        mock_open.side_effect = IOError("Test IO error")
        
        with patch('src.utils.logger.root_logger') as mock_root_logger:
            # Call clear_logs
            result = clear_logs()
            
            # Verify the result is False (failure)
            assert result is False
            
            # Verify error was logged
            mock_root_logger.error.assert_called_once()
            args, _ = mock_root_logger.error.call_args
            assert "Error clearing logs" in args[0]
            assert "Test IO error" in args[0]
    
    @patch('logging.Logger.addHandler')
    @patch('logging.Handler.setFormatter')
    @patch('logging.Handler.setLevel')
    def test_configure_root_logger(self, mock_set_level, mock_set_formatter, mock_add_handler):
        """Test configuring the root logger."""
        # Patch getLogger to avoid affecting the actual root logger
        with patch('logging.getLogger') as mock_get_logger, \
             patch('os.makedirs') as mock_makedirs, \
             patch('os.path.exists') as mock_exists:
            
            # Configure mocks
            mock_exists.return_value = False
            mock_root_logger = MagicMock()
            mock_get_logger.return_value = mock_root_logger
            
            # Call configure_root_logger
            result = configure_root_logger()
            
            # Verify the result is the mock root logger
            assert result is mock_root_logger
            
            # Verify root logger is set to DEBUG level
            mock_root_logger.setLevel.assert_called_with(logging.DEBUG)
            
            # Verify handlers are cleared before adding new ones
            mock_root_logger.handlers.clear.assert_called_once()
            
            # Verify logs directory was created
            mock_makedirs.assert_called_with(LOGS_DIR)
            
            # Verify addHandler was called for each handler type
            assert mock_root_logger.addHandler.call_count == 4  # Console + 3 file handlers
            
            # Verify each handler had setLevel and setFormatter called
            assert mock_set_level.call_count >= 4
            assert mock_set_formatter.call_count >= 4
            
            # Verify log initialization messages
            assert mock_root_logger.info.call_count >= 4  # Several info messages logged
    
    def test_real_logging(self, tmpdir):
        """Test actual logging to file."""
        # Use a temporary directory for log files
        test_logs_dir = os.path.join(tmpdir, "logs")
        os.makedirs(test_logs_dir, exist_ok=True)
        
        # Define temporary log files
        test_error_log = os.path.join(test_logs_dir, "error.log")
        test_info_log = os.path.join(test_logs_dir, "info.log")
        test_debug_log = os.path.join(test_logs_dir, "debug.log")
        
        # Patch LOGS_DIR and log files to use temporary directory
        with patch('src.utils.logger.LOGS_DIR', test_logs_dir), \
             patch('src.utils.logger.ERROR_LOG', test_error_log), \
             patch('src.utils.logger.INFO_LOG', test_info_log), \
             patch('src.utils.logger.DEBUG_LOG', test_debug_log), \
             patch('src.utils.logger.configure_root_logger') as mock_configure:
            
            # Force re-configuration of root logger with our temporary paths
            root_logger = configure_root_logger()
            
            # Create a test logger
            logger = get_logger("test.real_logging")
            
            # Log messages at different levels
            logger.debug("Debug test message")
            logger.info("Info test message")
            logger.warning("Warning test message")
            logger.error("Error test message")
            
            # Log an exception
            try:
                raise ValueError("Test exception for real logging")
            except ValueError as e:
                log_exception(logger, "Exception test", e)
            
            # Verify log files were created and contain expected content
            with open(test_debug_log, 'r') as f:
                debug_content = f.read()
                assert "Debug test message" in debug_content
                assert "Info test message" in debug_content
                assert "Warning test message" in debug_content
                assert "Error test message" in debug_content
                assert "Exception test" in debug_content
                assert "Test exception for real logging" in debug_content
            
            with open(test_info_log, 'r') as f:
                info_content = f.read()
                assert "Debug test message" not in info_content
                assert "Info test message" in info_content
                assert "Warning test message" in info_content
                assert "Error test message" in info_content
                assert "Exception test" in info_content
            
            with open(test_error_log, 'r') as f:
                error_content = f.read()
                assert "Debug test message" not in error_content
                assert "Info test message" not in error_content
                assert "Warning test message" not in error_content
                assert "Error test message" in error_content
                assert "Exception test" in error_content
