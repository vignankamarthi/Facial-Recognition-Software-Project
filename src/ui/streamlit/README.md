# Streamlit UI for Facial Recognition Project

## Directory Structure

- `app.py`: Main entry point for the Streamlit application
- `style.css`: Custom CSS styles for the Streamlit UI
- `pages_modules/`: Contains the individual page modules (used by app.py)
- `pages/`: Empty directory (prevents Streamlit auto-navigation)
- `main.py`: Deprecated file (kept for reference)

## Navigation

The application uses a custom single-page navigation system rather than Streamlit's multi-page app structure.
This provides a more consistent user experience and prevents duplicate navigation menus.

## Page Modules

Each feature of the application has its own module in the `pages_modules/` directory:

- `face_detection.py`: Face detection functionality
- `face_matching.py`: Face matching against known references
- `face_anonymization.py`: Face anonymization methods
- `bias_testing.py`: Demographic bias testing
- `dataset_management.py`: Dataset setup and management

## Styling

Custom styles are defined in:

1. `style.css`: External CSS file
2. Inline styles in `app.py`: Additional styles defined in the `apply_custom_css()` function

## Configuration

Streamlit configuration is controlled by:

1. `.streamlit/config.toml` in the project root
2. Runtime arguments passed to Streamlit via the `run_demo.py` script
