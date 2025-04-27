"""
Enhanced Configuration Module

This module centralizes all configuration settings for the facial recognition project
using a class-based approach with validation, environment variable support, and
proper fallback mechanisms.

The Config class provides a central point for accessing all configuration settings,
organized by category (paths, detection, matching, UI, etc.) with validation and
default values.

Functions and Classes
-------------------
Config
    Main configuration class with validated settings
get_config
    Function to get the singleton Config instance
load_config
    Function to load configuration from files/environment
ensure_dir_exists
    Helper function to create directories if needed

Examples
--------
>>> from utils.config import get_config
>>> config = get_config()
>>> faces_dir = config.paths.known_faces_dir
>>> matching_threshold = config.matching.threshold
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigSection:
    """Base class for configuration sections"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class PathConfig(ConfigSection):
    """Configuration for file paths"""
    
    def __init__(self, **kwargs):
        # Set defaults first
        self.project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.data_dir = os.path.join(self.project_root, "data")
        self.known_faces_dir = os.path.join(self.data_dir, "known_faces")
        self.test_images_dir = os.path.join(self.data_dir, "test_images")
        self.test_datasets_dir = os.path.join(self.data_dir, "test_datasets")
        self.datasets_dir = os.path.join(self.data_dir, "datasets")
        self.results_dir = os.path.join(self.data_dir, "results")
        self.logs_dir = os.path.join(self.project_root, "logs")
        
        # UTKFace dataset paths
        self.utkface_dir = os.path.join(self.datasets_dir, "utkface")
        self.utkface_aligned_dir = os.path.join(self.utkface_dir, "utkface_aligned")
        self.demographic_split_dir = os.path.join(self.utkface_dir, "demographic_split")
        
        # Demographic bias testing paths
        self.demographic_split_set_dir = os.path.join(self.test_datasets_dir, "demographic_split_set")
        self.demographic_results_dir = os.path.join(self.test_datasets_dir, "results")
        
        # Override defaults with any provided values
        super().__init__(**kwargs)
        
        # Create essential directories
        self.ensure_essential_dirs()
    
    def ensure_essential_dirs(self):
        """Ensure all essential directories exist"""
        for attr_name in dir(self):
            if attr_name.endswith('_dir') and not attr_name.startswith('_'):
                dir_path = getattr(self, attr_name)
                ensure_dir_exists(dir_path)


class DetectionConfig(ConfigSection):
    """Configuration for face detection"""
    
    def __init__(self, **kwargs):
        self.confidence = 0.5  # Confidence threshold
        self.supported_image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        super().__init__(**kwargs)


class MatchingConfig(ConfigSection):
    """Configuration for face matching"""
    
    def __init__(self, **kwargs):
        self.threshold = 0.6  # Threshold for face matching (higher = stricter)
        super().__init__(**kwargs)


class AnonymizationConfig(ConfigSection):
    """Configuration for face anonymization"""
    
    def __init__(self, **kwargs):
        self.default_method = "blur"
        self.default_intensity = 90  # Range 1-100
        super().__init__(**kwargs)


class DemographicConfig(ConfigSection):
    """Configuration for demographic analysis"""
    
    def __init__(self, **kwargs):
        self.groups = ["white", "black", "asian", "indian", "others"]
        self.ethnicity_codes = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
        self.gender_codes = {0: "Male", 1: "Female"}
        super().__init__(**kwargs)


class UIConfig(ConfigSection):
    """Configuration for user interface"""
    
    def __init__(self, **kwargs):
        self.window_name = "Video"
        self.text_color = (255, 255, 255)
        self.success_color = (0, 255, 0)  # Green
        self.warning_color = (0, 255, 255)  # Yellow
        self.error_color = (0, 0, 255)  # Red
        self.wait_key_delay = 100  # Milliseconds to wait for key press
        
        # OpenCV parameters
        self.font = None  # Will be set dynamically
        self.font_scale = 0.7
        self.font_thickness = 2
        
        super().__init__(**kwargs)
    
    def initialize_opencv_constants(self):
        """Initialize constants that depend on OpenCV. Call after cv2 is imported."""
        import cv2
        self.font = cv2.FONT_HERSHEY_SIMPLEX


class Config:
    """
    Main configuration class for the facial recognition project.
    
    This class loads and validates configuration settings from various sources
    (environment variables, config files, defaults) and provides access to
    all settings organized by category.
    
    Attributes
    ----------
    paths : PathConfig
        Configuration for file paths
    detection : DetectionConfig
        Configuration for face detection
    matching : MatchingConfig
        Configuration for face matching
    anonymization : AnonymizationConfig
        Configuration for face anonymization
    demographic : DemographicConfig
        Configuration for demographic analysis
    ui : UIConfig
        Configuration for user interface
    env : str
        Current environment (development, production)
    
    Methods
    -------
    reload()
        Reload configuration from sources
    to_dict()
        Convert configuration to dictionary
    from_dict(config_dict)
        Load configuration from dictionary
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, env: str = None):
        # Only initialize once
        if hasattr(self, 'initialized'):
            return
        
        # Set environment (always development for portfolio project)
        self.env = 'development'
        
        # Initialize configuration sections
        self.paths = PathConfig()
        self.detection = DetectionConfig()
        self.matching = MatchingConfig()
        self.anonymization = AnonymizationConfig()
        self.demographic = DemographicConfig()
        self.ui = UIConfig()
        
        # Load configuration from environment variables and files
        self.reload()
        
        self.initialized = True
    
    def reload(self):
        """Reload configuration from all sources"""
        config_dict = load_config(self.env)
        self.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'env': self.env,
            'paths': {k: v for k, v in vars(self.paths).items() if not k.startswith('_')},
            'detection': {k: v for k, v in vars(self.detection).items() if not k.startswith('_')},
            'matching': {k: v for k, v in vars(self.matching).items() if not k.startswith('_')},
            'anonymization': {k: v for k, v in vars(self.anonymization).items() if not k.startswith('_')},
            'demographic': {k: v for k, v in vars(self.demographic).items() if not k.startswith('_')},
            'ui': {k: v for k, v in vars(self.ui).items() if not k.startswith('_') and k != 'font'},
        }
    
    def from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary"""
        if not config_dict:
            return
        
        # Update environment if provided
        if 'env' in config_dict:
            self.env = config_dict['env']
        
        # Update each section
        sections = [
            ('paths', self.paths),
            ('detection', self.detection),
            ('matching', self.matching),
            ('anonymization', self.anonymization),
            ('demographic', self.demographic),
            ('ui', self.ui)
        ]
        
        for section_name, section_obj in sections:
            if section_name in config_dict:
                for key, value in config_dict[section_name].items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown config key: {section_name}.{key}")


def get_config() -> Config:
    """
    Get the singleton Config instance.
    
    Returns
    -------
    Config
        The singleton Config instance
    
    Examples
    --------
    >>> config = get_config()
    >>> known_faces_dir = config.paths.known_faces_dir
    """
    return Config()


def load_config(env: str = 'development') -> Dict[str, Any]:
    """
    Load configuration from various sources.
    
    This function loads configuration from environment variables and config files,
    with a specific precedence order:
    1. Environment variables
    2. User-specific config file
    3. Development-specific config file
    4. Default config file
    
    Parameters
    ----------
    env : str, optional
        Environment name (always 'development' for this portfolio project)
    
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary
    """
    config = {}
    
    # Load from config files (development environment only for portfolio project)
    config_paths = [
        os.path.join(get_project_root(), 'config', 'default.json'),
        os.path.join(get_project_root(), 'config', 'development.json'),
        os.path.join(get_project_root(), 'config', 'user.json'),
    ]
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Deep merge
                    deep_merge(config, file_config)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {e}")
    
    # Load from environment variables
    # We look for variables like FACIAL_RECOGNITION_PATHS_DATA_DIR
    prefix = 'FACIAL_RECOGNITION_'
    for key, value in os.environ.items():
        if key.startswith(prefix):
            parts = key[len(prefix):].lower().split('_')
            if len(parts) >= 2:
                section = parts[0]
                param = '_'.join(parts[1:])
                
                if section not in config:
                    config[section] = {}
                
                # Try to parse as int, float, or boolean
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value.lower() in ('true', 'false'):
                    value = (value.lower() == 'true')
                
                config[section][param] = value
    
    return config


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]):
    """
    Deep merge two dictionaries.
    
    Parameters
    ----------
    base : Dict[str, Any]
        Base dictionary (modified in place)
    update : Dict[str, Any]
        Dictionary with updates
    
    Returns
    -------
    Dict[str, Any]
        Updated base dictionary
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def ensure_dir_exists(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    directory_path : str
        Path to the directory
    
    Returns
    -------
    str
        Path to the directory (same as input)
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns
    -------
    str
        Path to the project root directory
    """
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Create the config directory (for future use)
config_dir = os.path.join(get_project_root(), 'config')
ensure_dir_exists(config_dir)

# Create default config files if they don't exist
default_config_path = os.path.join(config_dir, 'default.json')
if not os.path.exists(default_config_path):
    with open(default_config_path, 'w') as f:
        json.dump({
            "env": "development",
            "ui": {
                "window_name": "Facial Recognition Demo"
            }
        }, f, indent=4)
