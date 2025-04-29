"""
Unit tests for the configuration module.
"""
import pytest
import os
import json
from unittest.mock import patch, mock_open

from src.utils.config import (
    Config,
    get_config,
    load_config,
    deep_merge,
    ensure_dir_exists,
    get_project_root,
    PathConfig,
    DetectionConfig,
    MatchingConfig,
    AnonymizationConfig,
    DemographicConfig,
    UIConfig
)

class TestConfig:
    """Tests for the Config class."""
    
    def test_singleton_pattern(self):
        """Test that Config implements the singleton pattern."""
        # Create two instances of Config
        config1 = Config()
        config2 = Config()
        
        # They should be the same instance
        assert config1 is config2
        
        # Modifying one should modify the other
        config1.env = "test_environment"
        assert config2.env == "test_environment"
    
    def test_initialization(self):
        """Test that Config initializes with default values."""
        # Reset the singleton instance for this test
        Config._instance = None
        
        # Create a fresh config instance
        with patch('src.utils.config.load_config') as mock_load_config:
            # Mock load_config to return an empty dict
            mock_load_config.return_value = {}
            
            # Initialize config
            config = Config()
            
            # Verify default environment is set
            assert config.env == 'development'
            
            # Verify config sections are initialized
            assert isinstance(config.paths, PathConfig)
            assert isinstance(config.detection, DetectionConfig)
            assert isinstance(config.matching, MatchingConfig)
            assert isinstance(config.anonymization, AnonymizationConfig)
            assert isinstance(config.demographic, DemographicConfig)
            assert isinstance(config.ui, UIConfig)
            
            # Verify load_config was called
            mock_load_config.assert_called_once_with('development')
    
    def test_reload(self):
        """Test reloading configuration from sources."""
        # Create a config instance
        config = Config()
        
        # Test reloading configuration
        with patch('src.utils.config.load_config') as mock_load_config:
            # Mock load_config to return a test configuration
            test_config = {'env': 'test_env', 'ui': {'window_name': 'Test Window'}}
            mock_load_config.return_value = test_config
            
            # Reload config
            config.reload()
            
            # Verify config was reloaded
            assert config.env == 'test_env'
            assert config.ui.window_name == 'Test Window'
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        # Create a config instance with some test values
        config = Config()
        config.env = 'test_env'
        config.paths.data_dir = '/test/data/dir'
        config.ui.window_name = 'Test Window'
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Verify dictionary structure
        assert isinstance(config_dict, dict)
        assert config_dict['env'] == 'test_env'
        assert 'paths' in config_dict
        assert 'detection' in config_dict
        assert 'matching' in config_dict
        assert 'anonymization' in config_dict
        assert 'demographic' in config_dict
        assert 'ui' in config_dict
        
        # Verify section values
        assert config_dict['paths']['data_dir'] == '/test/data/dir'
        assert config_dict['ui']['window_name'] == 'Test Window'
    
    def test_from_dict(self):
        """Test loading configuration from dictionary."""
        # Create a config instance
        config = Config()
        
        # Create a test config dictionary
        test_config = {
            'env': 'test_env',
            'paths': {'data_dir': '/test/data/dir', 'known_faces_dir': '/test/faces/dir'},
            'detection': {'confidence': 0.7},
            'matching': {'threshold': 0.8},
            'ui': {'window_name': 'Test Window'}
        }
        
        # Load config from dictionary
        config.from_dict(test_config)
        
        # Verify config was updated
        assert config.env == 'test_env'
        assert config.paths.data_dir == '/test/data/dir'
        assert config.paths.known_faces_dir == '/test/faces/dir'
        assert config.detection.confidence == 0.7
        assert config.matching.threshold == 0.8
        assert config.ui.window_name == 'Test Window'
        
        # Test with an empty dictionary
        config.env = 'original_env'  # Set a value to check it's not changed
        config.from_dict({})
        
        # Verify config was not changed
        assert config.env == 'original_env'
        
        # Test with unknown config key
        with patch('src.utils.config.logger.warning') as mock_warning:
            config.from_dict({'ui': {'unknown_key': 'value'}})
            
            # Verify warning was logged
            mock_warning.assert_called_once()


class TestConfigSections:
    """Tests for configuration section classes."""
    
    def test_path_config(self):
        """Test PathConfig initialization and directory creation."""
        # Initialize PathConfig with default values
        with patch('os.path.abspath') as mock_abspath, \
             patch('os.path.dirname') as mock_dirname, \
             patch('src.utils.config.ensure_dir_exists') as mock_ensure_dir:
            
            # Configure mock return values
            mock_abspath.return_value = '/test/project/root'
            mock_dirname.return_value = '/test/project'
            
            # Create PathConfig
            path_config = PathConfig()
            
            # Verify project root and paths are set correctly
            assert path_config.project_root == '/test/project/root'
            assert path_config.data_dir.startswith('/test/project/root')
            
            # Verify ensure_dir_exists was called for each directory
            # At minimum, these directories should be created
            assert mock_ensure_dir.call_count >= 4
            
        # Test custom initialization
        custom_paths = {
            'data_dir': '/custom/data',
            'known_faces_dir': '/custom/faces'
        }
        
        with patch('src.utils.config.ensure_dir_exists') as mock_ensure_dir:
            # Create PathConfig with custom paths
            path_config = PathConfig(**custom_paths)
            
            # Verify custom paths are set
            assert path_config.data_dir == '/custom/data'
            assert path_config.known_faces_dir == '/custom/faces'
            
            # Verify default paths are still set
            assert hasattr(path_config, 'logs_dir')
            
            # Verify ensure_essential_dirs was called
            assert mock_ensure_dir.called
    
    def test_detection_config(self):
        """Test DetectionConfig initialization and defaults."""
        # TODO: Implement this test
        pass
    
    def test_matching_config(self):
        """Test MatchingConfig initialization and defaults."""
        # TODO: Implement this test
        pass
    
    def test_anonymization_config(self):
        """Test AnonymizationConfig initialization and defaults."""
        # TODO: Implement this test
        pass
    
    def test_demographic_config(self):
        """Test DemographicConfig initialization and defaults."""
        # TODO: Implement this test
        pass
    
    def test_ui_config(self):
        """Test UIConfig initialization and defaults."""
        # TODO: Implement this test
        pass


class TestConfigUtilities:
    """Tests for configuration utility functions."""
    
    def test_get_config(self):
        """Test the get_config function."""
        # TODO: Implement this test
        pass
    
    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_config(self, mock_exists, mock_file, mock_json_load):
        """Test loading configuration from various sources."""
        # TODO: Implement this test
        pass
    
    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        # TODO: Implement this test
        pass
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_ensure_dir_exists(self, mock_exists, mock_makedirs):
        """Test ensuring a directory exists."""
        # TODO: Implement this test
        pass
