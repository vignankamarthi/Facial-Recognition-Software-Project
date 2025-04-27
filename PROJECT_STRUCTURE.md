# Project Structure

This document explains the organization of the Facial Recognition Software Project codebase, including the separation of concerns between core functionality and user interfaces.

## Directory Structure

```
Facial-Recognition-Software-Project/
├── config/                         # Configuration files
│   ├── default.json                # Default configuration
│   └── development.json            # Development environment overrides
├── data/                           # Data directory
│   ├── datasets/                   # Raw datasets
│   ├── known_faces/                # Reference faces for matching
│   ├── results/                    # Output from processing
│   └── test_datasets/              # Prepared test data
├── docs/                           # Documentation
│   └── quick_guides/               # Feature-specific guides
├── logs/                           # Log files
│   ├── debug.log                   # Debug level messages
│   ├── info.log                    # Info level messages
│   └── error.log                   # Error level messages
├── src/                            # Source code
│   ├── core/                       # Core functionality
│   │   ├── __init__.py
│   │   ├── face_detection.py       # Face detection implementation
│   │   ├── face_matching.py        # Face matching implementation
│   │   ├── anonymization.py        # Face anonymization implementation
│   │   └── bias_testing.py         # Bias testing implementation
│   ├── utils/                      # Utility modules
│   │   ├── __init__.py
│   │   ├── api_patch.py            # Patching for face_recognition
│   │   ├── common_utils.py         # Shared utility functions
│   │   ├── config.py               # Enhanced configuration management
│   │   ├── face_recognition_patch.py # Additional patches
│   │   ├── image_processing.py     # Image handling utilities
│   │   └── logger.py               # Centralized logging system
│   ├── ui/                         # User interfaces
│   │   ├── __init__.py
│   │   └── streamlit/              # Streamlit web interface
│   │       ├── __init__.py
│   │       ├── app.py              # Main Streamlit app
│   │       └── pages/              # Streamlit pages
│   └── __init__.py
├── tests/                          # Test suite
├── .gitignore
├── PROJECT_STRUCTURE.md            # This file
├── README.md
├── requirements.txt
└── run_demo.py                     # Launcher script
```

## Separation of Concerns

The project has been restructured to clearly separate different concerns:

1. **Core Functionality** (`src/core/`):
   - Contains the algorithmic implementation of facial recognition features
   - Independent of any specific UI implementation
   - Focused on the business logic and computational aspects

2. **Utilities** (`src/utils/`):
   - Shared functionality used across the project
   - Enhanced configuration management with environment support
   - Centralized logging system
   - Common utility functions

3. **User Interface** (`src/ui/`):
   - Streamlit-based web interface
   - Uses the core functionality
   - Provides an interactive, user-friendly experience

## Configuration System

The new configuration system provides:

1. **Environment-Specific Settings**:
   - `default.json`: Base configuration for all settings
   - `development.json`: Development-specific overrides
   - `user.json`: User-specific overrides (gitignored)

2. **Environment Variable Support**:
   - Override any configuration value with an environment variable
   - Format: `FACIAL_RECOGNITION_SECTION_PARAM`
   - Example: `FACIAL_RECOGNITION_MATCHING_THRESHOLD=0.7`

3. **Configuration Access**:
   - Centralized access with `get_config()` function
   - Organized by section (paths, detection, matching, etc.)
   - Example: `config.paths.known_faces_dir`

## Logging System

The project uses a comprehensive logging system:

1. **Log Files**:
   - `logs/debug.log`: Contains all log messages (DEBUG level and above)
   - `logs/info.log`: Contains INFO level messages and above
   - `logs/error.log`: Contains only ERROR and CRITICAL level messages

2. **Log Features**:
   - Automatic log rotation (10MB file size limit with 5 backups)
   - Contextual information (timestamp, log level, file, line number)
   - Exception tracking with stack traces
   - Method call tracing

3. **Usage in Code**:
   ```python
   from src.utils.logger import get_logger
   
   logger = get_logger(__name__)
   logger.info("Operation started")
   logger.error("An error occurred")
   ```

## Running the Application

### Streamlit Interface

```bash
# Launch the Streamlit interface
python run_demo.py

# Specify a custom port
python run_demo.py --port 8502

# Specify server address
python run_demo.py --server-address 0.0.0.0
```

## Development Guidelines

1. **Core Logic Changes**:
   - Update files in `src/core/` when changing algorithmic functionality
   - Ensure changes are UI-independent
   - Add appropriate tests in `tests/`

2. **UI Changes**:
   - Update Streamlit components in `src/ui/streamlit/`
   - Keep UI logic separate from core logic

3. **Configuration Changes**:
   - Update `config/default.json` for universal changes
   - Use environment-specific files for targeted overrides
   - Document new configuration parameters

4. **Testing**:
   - Test core functionality independently of UI
   - Test UI components with mock core components when possible
   - Ensure complete test coverage of core functionality
