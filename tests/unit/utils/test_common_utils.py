"""
Unit tests for the common utilities module.
"""

import pytest
import os
import cv2
import time
from unittest.mock import patch, MagicMock

from src.utils.common_utils import (
    safely_close_windows,
    handle_opencv_error,
    FaceRecognitionError,
    create_resizable_window,
    get_project_root,
    get_data_dir,
    get_known_faces_dir,
    get_image_files,
    is_image_file,
    clean_directory,
    safe_copy_file,
    run_command,
    ProgressBar,
    FileError,
)


class TestPathFunctions:
    """Tests for path utility functions."""

    def test_get_project_root(self):
        """Test getting the project root directory."""
        with patch("src.utils.config.get_config") as mock_config:
            # Configure mock
            mock_config_instance = MagicMock()
            mock_config_instance.paths.project_root = "/test/project/root"
            mock_config.return_value = mock_config_instance

            # Call the function
            result = get_project_root()

            # Verify result
            assert result == "/test/project/root"
            mock_config.assert_called_once()

    def test_get_data_dir(self):
        """Test getting the data directory."""
        with patch("src.utils.config.get_config") as mock_config:
            # Configure mock
            mock_config_instance = MagicMock()
            mock_config_instance.paths.data_dir = "/test/data/dir"
            mock_config.return_value = mock_config_instance

            # Call the function
            result = get_data_dir()

            # Verify result
            assert result == "/test/data/dir"
            mock_config.assert_called_once()

    def test_get_known_faces_dir(self):
        """Test getting the known faces directory."""
        with patch("src.utils.config.get_config") as mock_config, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:
            # Configure mock
            mock_config_instance = MagicMock()
            mock_config_instance.paths.known_faces_dir = "/test/known_faces/dir"
            mock_config.return_value = mock_config_instance

            # Call the function
            result = get_known_faces_dir()

            # Verify result
            assert result == "/test/known_faces/dir"
            mock_config.assert_called_once()

            # Verify logging
            mock_logger.debug.assert_called_once_with(
                f"Known faces directory: {'/test/known_faces/dir'}"
            )

    def test_is_image_file(self):
        """Test checking if a file is an image."""
        with patch("src.utils.config.get_config") as mock_config:
            # Configure mock
            mock_config_instance = MagicMock()
            mock_config_instance.detection.supported_image_extensions = [
                ".jpg",
                ".jpeg",
                ".png",
            ]
            mock_config.return_value = mock_config_instance

            # Test valid image files
            assert is_image_file("test.jpg") is True
            assert is_image_file("test.jpeg") is True
            assert is_image_file("test.png") is True
            assert is_image_file("TEST.JPG") is True  # Case insensitive

            # Test invalid image files
            assert is_image_file("test.txt") is False
            assert is_image_file("test.pdf") is False
            assert is_image_file("test") is False

    def test_get_image_files(self, test_data_dir):
        """Test getting all image files in a directory."""
        # Create test files
        os.makedirs(test_data_dir, exist_ok=True)
        test_files = [
            os.path.join(test_data_dir, "image1.jpg"),
            os.path.join(test_data_dir, "image2.png"),
            os.path.join(test_data_dir, "document.txt"),  # Not an image
        ]

        with patch("os.path.exists") as mock_exists, patch(
            "os.listdir"
        ) as mock_listdir, patch(
            "src.utils.common_utils.is_image_file"
        ) as mock_is_image, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["image1.jpg", "image2.png", "document.txt"]
            mock_is_image.side_effect = lambda f: f.endswith((".jpg", ".png"))

            # Call the function
            image_files = get_image_files(test_data_dir)

            # Verify results
            assert len(image_files) == 2
            assert os.path.join(test_data_dir, "image1.jpg") in image_files
            assert os.path.join(test_data_dir, "image2.png") in image_files
            assert os.path.join(test_data_dir, "document.txt") not in image_files

            # Verify logging
            mock_logger.debug.assert_called_once()

        # Test with nonexistent directory
        with patch("os.path.exists") as mock_exists, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            mock_exists.return_value = False
            nonexistent_dir = "/nonexistent/dir"

            # Call the function
            result = get_image_files(nonexistent_dir)

            # Verify empty list is returned
            assert result == []

            # Verify warning was logged
            mock_logger.warning.assert_called_once_with(
                f"Directory does not exist: {nonexistent_dir}"
            )


class TestWindowManagement:
    """Tests for window management utilities."""

    @patch("cv2.namedWindow")
    @patch("cv2.setWindowProperty")
    @patch("cv2.resizeWindow")
    def test_create_resizable_window(self, mock_resize, mock_set_prop, mock_named):
        """Test creating a resizable window."""
        # Call the function with default parameters
        window_name = create_resizable_window("Test Window")

        # Verify the window was created
        assert window_name == "Test Window"
        mock_named.assert_called_once_with("Test Window", cv2.WINDOW_NORMAL)
        mock_set_prop.assert_called_once_with("Test Window", cv2.WND_PROP_TOPMOST, 1)
        mock_resize.assert_called_once_with("Test Window", 800, 600)

        # Test with custom dimensions
        mock_named.reset_mock()
        mock_set_prop.reset_mock()
        mock_resize.reset_mock()

        window_name = create_resizable_window("Custom Window", 1024, 768)

        # Verify the window was created with custom dimensions
        assert window_name == "Custom Window"
        mock_named.assert_called_once_with("Custom Window", cv2.WINDOW_NORMAL)
        mock_set_prop.assert_called_once_with("Custom Window", cv2.WND_PROP_TOPMOST, 1)
        mock_resize.assert_called_once_with("Custom Window", 1024, 768)

    @patch("cv2.destroyWindow")
    @patch("cv2.destroyAllWindows")
    @patch("cv2.waitKey")
    @patch("time.sleep")
    @patch("src.utils.common_utils.logger")
    def test_safely_close_windows(
        self, mock_logger, mock_sleep, mock_wait, mock_destroy_all, mock_destroy
    ):
        """Test safely closing OpenCV windows."""
        # Test with window_name and video_capture
        window_name = "Test Window"
        mock_video_capture = MagicMock()
        mock_video_capture.isOpened.return_value = True

        # Call the function
        safely_close_windows(window_name, mock_video_capture)

        # Verify video_capture.release was called
        mock_video_capture.isOpened.assert_called_once()
        mock_video_capture.release.assert_called_once()

        # Verify window closing functions were called
        mock_destroy.assert_called_with(window_name)
        assert (
            mock_destroy_all.call_count >= 2
        )  # Called at least twice (first attempt and retry)
        assert mock_wait.call_count >= 1  # Called at least once
        assert mock_sleep.call_count >= 2  # Called at least twice

        # Test with no window_name
        mock_destroy.reset_mock()
        mock_destroy_all.reset_mock()
        mock_wait.reset_mock()
        mock_sleep.reset_mock()
        mock_video_capture.reset_mock()

        safely_close_windows(video_capture=mock_video_capture)

        # Verify video_capture.release was called
        mock_video_capture.isOpened.assert_called_once()
        mock_video_capture.release.assert_called_once()

        # Verify only destroyAllWindows was called (not destroyWindow)
        mock_destroy.assert_not_called()
        assert mock_destroy_all.call_count >= 2

        # Test with no parameters
        mock_destroy.reset_mock()
        mock_destroy_all.reset_mock()

        safely_close_windows()

        # Verify only destroyAllWindows was called
        mock_destroy.assert_not_called()
        assert mock_destroy_all.call_count >= 2


class TestErrorHandling:
    """Tests for error handling utilities."""

    def test_face_recognition_error(self):
        """Test the base FaceRecognitionError class."""
        # Test basic error with only message
        with patch("src.utils.common_utils.log_exception") as mock_log_exception:
            error_msg = "Test error message"
            error = FaceRecognitionError(error_msg)

            # Verify error properties
            assert error.message == error_msg
            assert error.details is None
            assert error.source is None
            assert str(error) == error_msg

            # Verify logging was called
            mock_log_exception.assert_called_once()

        # Test error with details
        with patch("src.utils.common_utils.log_exception") as mock_log_exception:
            error_msg = "Test error message"
            details = "Error details"
            error = FaceRecognitionError(error_msg, details)

            # Verify error properties
            assert error.message == error_msg
            assert error.details == details
            assert error.source is None
            assert str(error) == f"{error_msg} - {details}"

            # Verify logging was called
            mock_log_exception.assert_called_once()

        # Test error with source exception
        with patch("src.utils.common_utils.log_exception") as mock_log_exception:
            error_msg = "Test error message"
            source_error = ValueError("Source error")
            error = FaceRecognitionError(error_msg, source=source_error)

            # Verify error properties
            assert error.message == error_msg
            assert error.details is None
            assert error.source == source_error
            assert str(error) == error_msg

            # Verify logging was called with source
            mock_log_exception.assert_called_once_with(
                error.format_message(), source_error
            )

    def test_handle_opencv_error(self):
        """Test the handle_opencv_error decorator."""

        # Create a test function with the decorator
        @handle_opencv_error
        def test_func():
            raise cv2.error("Test OpenCV error")

        # Test with an OpenCV error
        with patch("src.utils.common_utils.safely_close_windows") as mock_close_windows:
            result = test_func()

            # Verify function returns None and error dict
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] is None
            assert isinstance(result[1], dict)
            assert "error" in result[1]
            assert "type" in result[1]
            assert result[1]["type"] == "OpenCV"

            # Verify windows were cleaned up
            mock_close_windows.assert_called_once()

        # Create a test function that raises a CameraError
        @handle_opencv_error
        def test_camera_func():
            from src.utils.common_utils import CameraError

            raise CameraError("Test camera error")

        # Test with a CameraError
        with patch("src.utils.common_utils.safely_close_windows") as mock_close_windows:
            result = test_camera_func()

            # Verify function returns None and error dict
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] is None
            assert isinstance(result[1], dict)
            assert "error" in result[1]
            assert "type" in result[1]
            assert result[1]["type"] == "Camera"

            # Verify windows were cleaned up
            mock_close_windows.assert_called_once()


class TestFileOperations:
    """Tests for file operation utilities."""

    def test_clean_directory(self, test_data_dir):
        """Test cleaning up files in a directory."""
        # Create a test directory structure
        os.makedirs(test_data_dir, exist_ok=True)
        test_files = [
            os.path.join(test_data_dir, "file1.txt"),
            os.path.join(test_data_dir, "file2.txt"),
            os.path.join(test_data_dir, "image.jpg"),
        ]

        # Test non-recursive cleanup
        with patch("os.path.exists") as mock_exists, patch(
            "os.listdir"
        ) as mock_listdir, patch("os.path.isfile") as mock_isfile, patch(
            "os.remove"
        ) as mock_remove, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["file1.txt", "file2.txt", "image.jpg"]
            mock_isfile.return_value = True
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            result = clean_directory(test_data_dir)

            # Verify all files were removed
            assert mock_remove.call_count == 3
            for file in ["file1.txt", "file2.txt", "image.jpg"]:
                mock_remove.assert_any_call(os.path.join(test_data_dir, file))

            # Verify count of deleted files is returned
            assert result == 3

            # Verify logging
            mock_logger.info.assert_any_call(f"Cleaned {3} items from {test_data_dir}")

        # Test with pattern
        with patch("os.path.exists") as mock_exists, patch(
            "src.utils.common_utils.Path.glob"
        ) as mock_glob, patch(
            "src.utils.common_utils.Path.is_file"
        ) as mock_is_file, patch(
            "src.utils.common_utils.Path.unlink"
        ) as mock_unlink, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.return_value = True
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Mock the glob results
            mock_path1 = MagicMock()
            mock_path2 = MagicMock()
            mock_glob.return_value = [mock_path1, mock_path2]
            mock_is_file.return_value = True

            # Call the function with pattern
            result = clean_directory(test_data_dir, pattern="*.txt")

            # Verify glob was called with the pattern
            mock_glob.assert_called_once_with("*.txt")

            # Verify files were deleted
            assert mock_unlink.call_count == 2

            # Verify count of deleted files is returned
            assert result == 2

        # Test with non-existent directory
        with patch("os.path.exists") as mock_exists, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch("src.utils.common_utils.logger") as mock_logger:

            # Configure mocks
            mock_exists.return_value = False
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            result = clean_directory("/nonexistent/dir")

            # Verify 0 is returned
            assert result == 0

            # Verify warning was logged
            mock_logger.warning.assert_called_once()

    def test_safe_copy_file(self, test_data_dir):
        """Test safely copying a file with error handling."""
        # Create test file paths
        src_path = os.path.join(test_data_dir, "source_file.txt")
        dst_path = os.path.join(test_data_dir, "destination_file.txt")

        # Test successful copy
        with patch("os.path.exists") as mock_exists, patch(
            "os.makedirs"
        ) as mock_makedirs, patch("shutil.copy2") as mock_copy2, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.side_effect = (
                lambda path: path == src_path
            )  # Source exists, destination doesn't
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            result_path = safe_copy_file(src_path, dst_path)

            # Verify destination directory was created
            mock_makedirs.assert_called_once()

            # Verify file was copied
            mock_copy2.assert_called_once_with(src_path, dst_path)

            # Verify original destination path is returned
            assert result_path == dst_path

            # Verify logging
            mock_logger.info.assert_called_once_with(
                f"Copied file from {src_path} to {dst_path}"
            )

        # Test copy with existing destination without overwrite
        with patch("os.path.exists") as mock_exists, patch(
            "os.path.splitext"
        ) as mock_splitext, patch("time.time") as mock_time, patch(
            "shutil.copy2"
        ) as mock_copy2, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.return_value = True  # Both source and destination exist
            mock_splitext.return_value = (
                os.path.join(test_data_dir, "destination_file"),
                ".txt",
            )
            mock_time.return_value = 12345
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function with overwrite=False
            result_path = safe_copy_file(src_path, dst_path, overwrite=False)

            # Verify alternative filename was used
            expected_alt_path = os.path.join(
                test_data_dir, "destination_file_12345.txt"
            )
            assert result_path == expected_alt_path
            mock_copy2.assert_called_once_with(src_path, expected_alt_path)

        # Test copy with existing destination with overwrite
        with patch("os.path.exists") as mock_exists, patch(
            "shutil.copy2"
        ) as mock_copy2, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.logger"
        ) as mock_logger:

            # Configure mocks
            mock_exists.return_value = True  # Both source and destination exist
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function with overwrite=True
            result_path = safe_copy_file(src_path, dst_path, overwrite=True)

            # Verify original destination path was used
            assert result_path == dst_path
            mock_copy2.assert_called_once_with(src_path, dst_path)

        # Test with non-existent source file
        with patch("os.path.exists") as mock_exists, patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.log_exception"
        ) as mock_log_exception:

            # Configure mocks
            mock_exists.return_value = False  # Source doesn't exist
            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            with pytest.raises(FileError) as excinfo:
                safe_copy_file(src_path, dst_path)

            # Verify error message
            assert "Source file does not exist" in str(excinfo.value)


class TestProcessManagement:
    """Tests for process management utilities."""

    @patch("subprocess.Popen")
    def test_run_command(self, mock_popen):
        """Test running a shell command and handling output."""
        # Configure the mock
        mock_process = MagicMock()
        mock_process.stdout = ["Output line 1\n", "Output line 2\n"]
        mock_process.stderr = ["Error line 1\n"]
        mock_process.wait.return_value = None
        mock_process.returncode = 0  # Success
        mock_popen.return_value = mock_process

        # Test successful command
        with patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch("src.utils.common_utils.logger") as mock_logger:

            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            success, result = run_command("test command")

            # Verify command was executed
            mock_popen.assert_called_once_with(
                "test command",
                shell=True,
                stdout=mock_popen.call_args[1]["stdout"],
                stderr=mock_popen.call_args[1]["stderr"],
                universal_newlines=True,
            )

            # Verify process was waited for
            mock_process.wait.assert_called_once()

            # Verify result is properly structured
            assert success is True
            assert result["returncode"] == 0
            assert "Output line 1" in result["stdout"]
            assert "Output line 2" in result["stdout"]
            assert "Error line 1" in result["stderr"]

            # Verify logging
            mock_logger.info.assert_any_call("Executing command: test command")
            mock_logger.info.assert_any_call("STDOUT: Output line 1")
            mock_logger.info.assert_any_call("STDOUT: Output line 2")
            mock_logger.error.assert_any_call("STDERR: Error line 1")
            mock_logger.info.assert_any_call(
                "Command executed successfully (return code: 0)"
            )

        # Test failed command
        mock_process.returncode = 1  # Failure

        with patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch("src.utils.common_utils.logger") as mock_logger:

            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            success, result = run_command("test command")

            # Verify result
            assert success is False
            assert result["returncode"] == 1

            # Verify logging
            mock_logger.error.assert_any_call("Command failed with return code: 1")

        # Test exception handling
        mock_popen.side_effect = Exception("Test exception")

        with patch(
            "src.utils.common_utils.log_method_call"
        ) as mock_log_decorator, patch(
            "src.utils.common_utils.log_exception"
        ) as mock_log_exception:

            mock_log_decorator.return_value = (
                lambda x: x
            )  # Return the function unchanged

            # Call the function
            success, result = run_command("test command")

            # Verify result
            assert success is False
            assert result["returncode"] == -1
            assert result["stdout"] == ""
            assert "Test exception" in result["stderr"]

            # Verify exception was logged
            mock_log_exception.assert_called_once()


class TestProgressDisplay:
    """Tests for progress display utilities."""

    @patch("builtins.print")
    @patch("time.time")
    def test_progress_bar(self, mock_time, mock_print):
        """Test the ProgressBar class."""
        # Configure time mock
        mock_time.side_effect = [
            0,
            0,
            5,
            5,
            10,
            10,
        ]  # Start time, then each update time

        # Set up test parameters
        total = 10
        prefix = "Processing:"
        suffix = "Complete"

        # Create a mock logger
        mock_logger = MagicMock()

        with patch("src.utils.common_utils.get_logger") as mock_get_logger:
            mock_get_logger.return_value = mock_logger

            # Create the progress bar
            progress_bar = ProgressBar(total, prefix, suffix)

            # Verify initial update
            mock_print.assert_called_once()
            mock_print.reset_mock()

            # Update progress to 5/10
            progress_bar.update(5)

            # Verify progress bar was updated
            mock_print.assert_called_once()
            # Extract the call args to check the progress string
            args, kwargs = mock_print.call_args
            progress_str = args[0]
            assert prefix in progress_str
            assert suffix in progress_str
            assert "5/10" in progress_str
            assert "50.0%" in progress_str
            assert "1.0 it/s" in progress_str  # 5 items / 5 seconds
            assert kwargs["end"] == "\r"

            # Log should be called since 5 seconds elapsed
            mock_logger.log.assert_called_once()
            mock_print.reset_mock()
            mock_logger.reset_mock()

            # Update to completion
            progress_bar.update(10)

            # Verify final progress bar
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            progress_str = args[0]
            assert "10/10" in progress_str
            assert "100.0%" in progress_str
            assert "1.0 it/s" in progress_str  # 10 items / 10 seconds

            # Second call to print newline at completion
            assert mock_print.call_count == 2

            # Log should be called for completion
            assert mock_logger.log.call_count == 1
            mock_logger.info.assert_called_once()
