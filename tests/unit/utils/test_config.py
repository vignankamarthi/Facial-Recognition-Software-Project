"""
Unit tests for the configuration module.
"""
import pytest
import os
import json
from unittest.mock import patch, mock_open, MagicMock

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
        # Test default initialization
        detection_config = DetectionConfig()
        
        # Verify default values
        assert detection_config.confidence == 0.5
        assert isinstance(detection_config.supported_image_extensions, list)
        assert ".jpg" in detection_config.supported_image_extensions
        assert ".jpeg" in detection_config.supported_image_extensions
        assert ".png" in detection_config.supported_image_extensions
        
        # Test custom initialization
        custom_detection = {
            'confidence': 0.8,
            'supported_image_extensions': ['.custom', '.test']
        }
        
        detection_config = DetectionConfig(**custom_detection)
        
        # Verify custom values
        assert detection_config.confidence == 0.8
        assert detection_config.supported_image_extensions == ['.custom', '.test']
    
    def test_matching_config(self):
        """Test MatchingConfig initialization and defaults."""
        # Test default initialization
        matching_config = MatchingConfig()
        
        # Verify default values
        assert matching_config.threshold == 0.6
        
        # Test custom initialization
        custom_matching = {
            'threshold': 0.75
        }
        
        matching_config = MatchingConfig(**custom_matching)
        
        # Verify custom values
        assert matching_config.threshold == 0.75
    
    def test_anonymization_config(self):
        """Test AnonymizationConfig initialization and defaults."""
        # Test default initialization
        anonymization_config = AnonymizationConfig()
        
        # Verify default values
        assert anonymization_config.default_method == "blur"
        assert anonymization_config.default_intensity == 90
        
        # Test custom initialization
        custom_anonymization = {
            'default_method': 'pixelate',
            'default_intensity': 50
        }
        
        anonymization_config = AnonymizationConfig(**custom_anonymization)
        
        # Verify custom values
        assert anonymization_config.default_method == 'pixelate'
        assert anonymization_config.default_intensity == 50
    
    def test_demographic_config(self):
        """Test DemographicConfig initialization and defaults."""
        # Test default initialization
        demographic_config = DemographicConfig()
        
        # Verify default values
        assert isinstance(demographic_config.groups, list)
        assert len(demographic_config.groups) == 5  # white, black, asian, indian, others
        assert "white" in demographic_config.groups
        assert isinstance(demographic_config.ethnicity_codes, dict)
        assert isinstance(demographic_config.gender_codes, dict)
        assert demographic_config.ethnicity_codes[0] == "White"
        assert demographic_config.gender_codes[0] == "Male"
        
        # Test custom initialization
        custom_demographic = {
            'groups': ['group_a', 'group_b'],
            'ethnicity_codes': {0: 'Group A', 1: 'Group B'},
            'gender_codes': {0: 'One', 1: 'Two'}
        }
        
        demographic_config = DemographicConfig(**custom_demographic)
        
        # Verify custom values
        assert demographic_config.groups == ['group_a', 'group_b']
        assert demographic_config.ethnicity_codes == {0: 'Group A', 1: 'Group B'}
        assert demographic_config.gender_codes == {0: 'One', 1: 'Two'}
    
    def test_ui_config(self):
        """Test UIConfig initialization and defaults."""
        # Test default initialization
        ui_config = UIConfig()
        
        # Verify default values
        assert ui_config.window_name == "Video"
        assert ui_config.text_color == (255, 255, 255)
        assert ui_config.success_color == (0, 255, 0)
        assert ui_config.warning_color == (0, 255, 255)
        assert ui_config.error_color == (0, 0, 255)
        assert ui_config.wait_key_delay == 100
        assert ui_config.font_scale == 0.7
        assert ui_config.font_thickness == 2
        
        # Test custom initialization
        custom_ui = {
            'window_name': 'Custom Window',
            'text_color': (200, 200, 200),
            'wait_key_delay': 50,
            'font_scale': 1.0
        }
        
        ui_config = UIConfig(**custom_ui)
        
        # Verify custom values
        assert ui_config.window_name == 'Custom Window'
        assert ui_config.text_color == (200, 200, 200)
        assert ui_config.wait_key_delay == 50
        assert ui_config.font_scale == 1.0
        
        # Test initialize_opencv_constants method
        import cv2
        
        with patch.object(cv2, 'FONT_HERSHEY_SIMPLEX', 123):
            ui_config.initialize_opencv_constants()
            assert ui_config.font == 123


class TestConfigUtilities:
    """Tests for configuration utility functions."""
    
    def test_get_config(self):
        """Test the get_config function."""
        # Reset singleton for this test
        Config._instance = None
        
        # Call get_config
        config = get_config()
        
        # Verify it returns a Config instance
        assert isinstance(config, Config)
        
        # Call again and verify it returns the same instance
        config2 = get_config()
        assert config is config2
    
    @patch('json.load')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_load_config(self, mock_exists, mock_file, mock_json_load):
        """Test loading configuration from various sources."""
        # Mock the project root
        with patch('src.utils.config.get_project_root') as mock_get_root:
            mock_get_root.return_value = '/test/project/root'
            
            # Configure mocks
            # Config files exist
            mock_exists.return_value = True
            
            # Setup different config files
            default_config = {'ui': {'window_name': 'Default'}}
            dev_config = {'env': 'development', 'detection': {'confidence': 0.7}}
            user_config = {'ui': {'window_name': 'User'}, 'matching': {'threshold': 0.8}}
            
            # Mock json.load to return different configs based on file path
            def mock_json_load_side_effect(file_obj):
                filename = file_obj.name
                if 'default.json' in filename:
                    return default_config
                elif 'development.json' in filename:
                    return dev_config
                elif 'user.json' in filename:
                    return user_config
                return {}
                
            mock_json_load.side_effect = mock_json_load_side_effect
            
            # Mock os.environ for testing environment variables
            env_vars = {
                'FACIAL_RECOGNITION_PATHS_DATA_DIR': '/env/data',
                'FACIAL_RECOGNITION_UI_WINDOW_NAME': 'EnvWindow',
                'FACIAL_RECOGNITION_DETECTION_CONFIDENCE': '0.9',
                'FACIAL_RECOGNITION_MATCHING_THRESHOLD': '0.75',
                'FACIAL_RECOGNITION_ANONYMIZATION_DEFAULT_METHOD': 'pixelate'
            }
            
            with patch.dict('os.environ', env_vars):
                # Load config for development environment
                config = load_config('development')
                
                # Verify config was merged correctly
                assert isinstance(config, dict)
                
                # Check values from different sources with correct precedence
                # Environment variables should override config files
                assert config['paths']['data_dir'] == '/env/data'
                assert config['ui']['window_name'] == 'EnvWindow'
                assert config['detection']['confidence'] == 0.9
                assert config['matching']['threshold'] == 0.75
                assert config['anonymization']['default_method'] == 'pixelate'
                
                # Verify environment is set from dev config
                assert config['env'] == 'development'
                
                # Verify files were opened
                assert mock_file.call_count == 3
                mock_file.assert_any_call('/test/project/root/config/default.json', 'r')
                mock_file.assert_any_call('/test/project/root/config/development.json', 'r')
                mock_file.assert_any_call('/test/project/root/config/user.json', 'r')
    
    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        # Test with simple dictionaries
        base = {'a': 1, 'b': 2}
        update = {'b': 3, 'c': 4}
        
        result = deep_merge(base, update)
        
        # Verify result
        assert result is base  # Should modify base in-place
        assert base['a'] == 1
        assert base['b'] == 3  # Updated
        assert base['c'] == 4  # Added
        
        # Test with nested dictionaries
        base = {
            'a': 1,
            'nested': {
                'x': 100,
                'y': 200
            }
        }
        
        update = {
            'b': 2,
            'nested': {
                'y': 300,
                'z': 400
            }
        }
        
        result = deep_merge(base, update)
        
        # Verify result
        assert result is base  # Should modify base in-place
        assert base['a'] == 1
        assert base['b'] == 2
        assert base['nested']['x'] == 100  # Unchanged
        assert base['nested']['y'] == 300  # Updated
        assert base['nested']['z'] == 400  # Added
        
        # Test with empty dictionaries
        base = {}
        update = {'a': 1}
        
        result = deep_merge(base, update)
        
        # Verify result
        assert result is base
        assert base['a'] == 1
        
        base = {'a': 1}
        update = {}
        
        result = deep_merge(base, update)
        
        # Verify result
        assert result is base
        assert base['a'] == 1  # Unchanged
    
    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_ensure_dir_exists(self, mock_exists, mock_makedirs):
        """Test ensuring a directory exists."""
        # Test when directory already exists
        mock_exists.return_value = True
        
        result = ensure_dir_exists('/existing/dir')
        
        # Verify result
        assert result == '/existing/dir'
        
        # makedirs should not be called
        mock_makedirs.assert_not_called()
        
        # Test when directory doesn't exist
        mock_exists.return_value = False
        
        result = ensure_dir_exists('/new/dir')
        
        # Verify result
        assert result == '/new/dir'
        
        # makedirs should be called
        mock_makedirs.assert_called_once_with('/new/dir')
