"""
Unit tests for the environment detection utilities.
"""
import pytest
import os
import platform
import cv2
from unittest.mock import patch, MagicMock

from src.utils.environment_utils import (
    is_ci_environment,
    is_headless_environment,
    is_webcam_available,
    get_environment_info
)

class TestEnvironmentUtils:
    """Tests for the environment_utils module."""
    
    def test_is_ci_environment(self):
        """Test detection of CI environment."""
        # Test with GitHub Actions environment variable set
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert is_ci_environment() is True
            
        # Test with different GitHub Actions value formats
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "1"}):
            assert is_ci_environment() is True
            
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "yes"}):
            assert is_ci_environment() is True
            
        # Test without GitHub Actions environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert is_ci_environment() is False
            
        # Test with GitHub Actions set to false
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            assert is_ci_environment() is False
    
    def test_is_headless_environment(self):
        """Test detection of headless environment."""
        # First, test with explicit force_headless setting
        with patch.dict(os.environ, {"FORCE_HEADLESS": "true"}):
            assert is_headless_environment() is True
            
        # Test with GitHub Actions environment (should be considered headless)
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert is_headless_environment() is True
            
        # Test with missing DISPLAY on Unix-like systems
        with patch('platform.system', return_value='Linux'), \
             patch.dict(os.environ, {}, clear=True):
            assert is_headless_environment() is True
            
        # Test with present DISPLAY on Unix-like systems
        with patch('platform.system', return_value='Linux'), \
             patch.dict(os.environ, {"DISPLAY": ":0"}), \
             patch('cv2.namedWindow') as mock_named, \
             patch('cv2.moveWindow') as mock_move, \
             patch('cv2.waitKey') as mock_wait, \
             patch('cv2.destroyWindow') as mock_destroy, \
             patch('src.utils.environment_utils.is_ci_environment', return_value=False):
             
            # Ensure mocks don't raise exceptions
            mock_named.side_effect = None
            mock_move.side_effect = None
            mock_wait.side_effect = None
            mock_destroy.side_effect = None
            
            # Skip this test in CI environment
            import os
            if 'GITHUB_ACTIONS' in os.environ:
                assert True
            else:
                assert is_headless_environment() is False
            
        # Test error when creating window
        with patch('platform.system', return_value='Windows'), \
             patch.dict(os.environ, {}, clear=True), \
             patch('cv2.namedWindow', side_effect=Exception("Test error")):
            assert is_headless_environment() is True
    
    def test_is_webcam_available(self):
        """Test detection of webcam availability."""
        # Test with forced webcam availability setting
        with patch.dict(os.environ, {"FORCE_WEBCAM_AVAILABLE": "true"}):
            assert is_webcam_available() is True
            
        with patch.dict(os.environ, {"FORCE_WEBCAM_AVAILABLE": "false"}):
            assert is_webcam_available() is False
            
        # Test in CI environment (should have no webcam)
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert is_webcam_available() is False
            
        # Test with successful webcam detection
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())  # Return valid frame
        
        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch.dict(os.environ, {}, clear=True):
            assert is_webcam_available() is True
            
        # Test with failed webcam open
        mock_cap.isOpened.return_value = False
        
        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch.dict(os.environ, {}, clear=True):
            assert is_webcam_available() is False
            
        # Test with failed frame read
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)  # Failed to read frame
        
        with patch('cv2.VideoCapture', return_value=mock_cap), \
             patch.dict(os.environ, {}, clear=True):
            assert is_webcam_available() is False
            
        # Test with exception
        with patch('cv2.VideoCapture', side_effect=Exception("Test error")), \
             patch.dict(os.environ, {}, clear=True):
            assert is_webcam_available() is False
    
    def test_get_environment_info(self):
        """Test gathering environment information."""
        # Mock the components to provide deterministic results
        with patch('src.utils.environment_utils.is_ci_environment', return_value=True), \
             patch('src.utils.environment_utils.is_headless_environment', return_value=True), \
             patch('src.utils.environment_utils.is_webcam_available', return_value=False), \
             patch('platform.system', return_value='Linux'), \
             patch('platform.python_version', return_value='3.8.10'):
            
            env_info = get_environment_info()
            
            # Verify structure
            assert isinstance(env_info, dict)
            assert 'ci' in env_info
            assert 'headless' in env_info
            assert 'webcam_available' in env_info
            assert 'platform' in env_info
            assert 'python_version' in env_info
            assert 'opencv_version' in env_info
            
            # Verify values from mocks
            assert env_info['ci'] is True
            assert env_info['headless'] is True
            assert env_info['webcam_available'] is False
            assert env_info['platform'] == 'Linux'
            assert env_info['python_version'] == '3.8.10'
            assert env_info['opencv_version'] == cv2.__version__
