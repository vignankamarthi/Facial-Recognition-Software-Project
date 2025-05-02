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
│   │   ├── lfw/                    # Labeled Faces in the Wild dataset (Not Compatible Anymore)
│   │   └── utkface/                # UTKFace dataset
│   │       ├── archives/           # Original downloaded archives
│   │       ├── demographic_split/  # Images organized by demographic groups
│   │       ├── utkface_aligned/    # Aligned face images
│   │       └── utkface_data/       # Raw extracted data
│   ├── known_faces/                # Reference faces for matching
│   ├── results/                    # Output from processing
│   │   └── test_datasets/          # Results from dataset testing
│   ├── test_datasets/              # Prepared test data
│   │   ├── demographic_split_set/  # Data for bias testing
│   │   └── results/                # Result visualizations
│   └── test_images/                # Images for testing
│       ├── known/                  # Known face test images
│       └── unknown/                # Unknown face test images
├── docker/                         # Docker configuration
│   ├── docker-compose.yml          # Docker Compose configuration
│   ├── Dockerfile                  # Docker image definition
│   ├── entrypoint.sh               # Container entry point script
│   ├── init_demo_data.py           # Demo data initialization script
│   ├── README.md                   # Docker usage instructions
│   └── webcam/                     # Webcam integration for Docker
├── docs/                           # Documentation
│   ├── docstring_template.md       # Template for code documentation
│   ├── ethical_discussion_draft.md # Notes on ethical considerations
│   └── troubleshooting.md          # Troubleshooting guide
├── logs/                           # Log files
│   ├── debug.log                   # Debug level messages
│   ├── info.log                    # Info level messages
│   └── error.log                   # Error level messages
├── src/                            # Source code
│   ├── backend/                    # Backend functionality
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
│   │   ├── environment_utils.py    # Environment variable handling
│   │   ├── face_recognition_patch.py # Additional patches
│   │   ├── image_processing.py     # Image handling utilities
│   │   └── logger.py               # Centralized logging system
│   ├── ui/                         # User interfaces
│   │   ├── __init__.py
│   │   └── streamlit/              # Streamlit web interface
│   │       ├── __init__.py
│   │       ├── app.py              # Main Streamlit app
│   │       ├── components/         # Reusable UI components
│   │       ├── pages_modules/      # Individual feature pages
│   │       ├── static/             # Static assets
│   │       └── style.css           # Custom CSS styling
│   └── __init__.py
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Test configuration
│   ├── data/                       # Test data
│   ├── functional/                 # Functional tests
│   ├── integration/                # Integration tests
│   ├── test_logging.py             # Logging tests
│   └── unit/                       # Unit tests
├── PROJECT_STRUCTURE.md            # This file
├── README.md                       # Project overview
├── pytest.ini                      # pytest configuration
├── requirements.txt                # Python dependencies
├── run_demo.py                     # Launcher script
└── run_tests.py                    # Test runner script
```

## Separation of Concerns

The project has been restructured to clearly separate different concerns:

1. **Backend Functionality** (`src/backend/`):

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

4. **Container Support** (`docker/`):
   - Docker configuration for containerized deployment
   - Special handling for webcam functionality in containers
   - Initialization scripts for demo data

## Testing Structure

The testing framework is organized into multiple levels:

1. **Unit Tests** (`tests/unit/`):

   - Tests for individual components in isolation
   - Backend modules, utility functions, etc.

2. **Integration Tests** (`tests/integration/`):

   - Tests interactions between components
   - E.g., detection → anonymization, detection → matching

3. **Functional Tests** (`tests/functional/`):
   - Tests complete workflows from end to end
   - Mimics actual user interactions

## Dataset Organization

The project uses several datasets organized for different purposes:

1. **Raw Datasets** (`data/datasets/`):

   - UTKFace: Images with demographic annotations
   - LFW (Labeled Faces in the Wild): Additional face dataset (Not Compatible Anymore)

2. **Processed Datasets** (`data/test_datasets/`):

   - Demographic splits for bias testing
   - Test images for evaluating recognition performance

3. **Known Faces** (`data/known_faces/`):
   - Reference faces for face matching feature

## Configuration System

The configuration system provides:

1. **Environment-Specific Settings**:

   - `default.json`: Base configuration for all settings
   - `development.json`: Development-specific overrides

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

### Docker Deployment

```bash
# Build and start using Docker Compose
cd docker
docker-compose up -d

# Access the application at http://localhost:8501
```

## Development Guidelines

1. **Backend Logic Changes**:

   - Update files in `src/backend/` when changing algorithmic functionality
   - Ensure changes are UI-independent
   - Add appropriate tests in `tests/`

2. **UI Changes**:

   - Update Streamlit components in `src/ui/streamlit/`
   - Keep UI logic separate from backend logic

3. **Configuration Changes**:

   - Update `config/default.json` for universal changes
   - Use environment-specific files for targeted overrides
   - Document new configuration parameters

4. **Testing**:
   - Test backend functionality independently of UI
   - Test UI components with mock backend components when possible
   - Ensure complete test coverage of backend functionality
