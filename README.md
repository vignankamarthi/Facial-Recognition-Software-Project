# Facial Recognition Software Project
A facial recognition system demonstrating technical capabilities and ethical considerations in AI-based recognition technologies.

[ ] TODO: Refactor README after project is complete
[ ] TODO: Go through ALL documentation/md files


## Quick Start Guide

### Installation
```bash
git clone https://github.com/vignankamarthi/Facial-Recognition-Software-Project.git
cd Facial-Recognition-Software-Project
pip install -r requirements.txt
pip install streamlit  # For the web interface
```

### Running the Demo
```bash
python run_demo.py
```

This will launch the Streamlit web interface that provides access to all features.

### Features Overview

| Feature | Webcam Required | Description |
|---------|:--------------:|-------------|
| Face Detection | ✅ | Detect faces in real-time or in uploaded images |
| Face Anonymization | ✅ | Blur, pixelate, or mask detected faces |
| Face Matching | ✅ | Match detected faces against reference images |
| Demographic Bias Testing | ❌ | Test recognition accuracy across different ethnicities using UTKFace dataset |
| Dataset Management | ❌ | Download and prepare datasets for testing |

## UTKFace Dataset Integration

This project uses the UTKFace (University of Tennessee, Knoxville Face) dataset for ethical bias testing, providing:

- **Demographic Labels**: Age, gender, and ethnicity annotations for over 20,000 face images
- **Ethnicity Categories**: White, Black, Asian, Indian, and Others (including Hispanic, Latino, Middle Eastern)
- **Statistical Analysis**: Variance, standard deviation, and bias level calculations
- **Visualization**: Color-coded charts showing accuracy differences across demographic groups

The dataset allows for realistic demonstration of potential algorithmic bias in facial recognition systems and serves as an educational tool for understanding ethical concerns in AI.

## Streamlit Interface Options

```bash
# Run on default port (8501)
python run_demo.py

# Run on a specific port
python run_demo.py --port 8502

# Make available on network
python run_demo.py --server-address 0.0.0.0
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed information about the codebase organization.

```
Facial-Recognition-Software-Project/
├── config/                 # Configuration files
├── data/                   # Data directory
│   ├── datasets/           # Raw datasets
│   │   └── utkface/        # UTKFace dataset
│   ├── test_datasets/      # Processed test data
│   │   └── demographic_split_set/ # Ethnicity-organized images
├── docs/                   # Documentation
│   ├── quick_guides/       # Feature-specific guides
│   └── ethical_discussion.md
├── logs/                   # System logs
│   ├── debug.log
│   ├── info.log
│   └── error.log
├── src/                    # Source code
│   ├── backend/            # Backend functionality
│   ├── utils/              # Utility modules
│   └── ui/                 # Streamlit user interface
├── PROJECT_STRUCTURE.md    # Detailed structure documentation
├── run_demo.py             # Demo launcher
└── requirements.txt        # Dependencies
```

For detailed usage instructions, see [docs/quick_guides/](docs/quick_guides/).

For ethical considerations, see [docs/ethical_discussion.md](docs/ethical_discussion_draft.md).

## Recent Improvements

### 1. Project Restructuring
- Separated backend functionality from user interface
- Implemented Streamlit web interface for better user experience
- Created clear directory structure for improved maintainability

### 2. Enhanced Configuration System
- Added centralized configuration management
- Support for environment-specific configuration
- Configuration via JSON files and environment variables

### 3. Robust Logging System
- Comprehensive logging with different severity levels
- Log rotation to prevent log files from growing too large
- Detailed error tracking with stack traces
- Context information for better debugging

### 4. Standardized Method Signatures
- Consistent parameter naming across all modules
- Improved docstrings with NumPy-style formatting
- Better return type consistency across the codebase
- Added input validation to prevent errors

## Logging System

All application logs are automatically saved to the `logs` directory:
- `logs/debug.log`: Contains all log messages (DEBUG level and above)
- `logs/info.log`: Contains INFO level messages and above
- `logs/error.log`: Contains only ERROR and CRITICAL level messages

If you encounter issues, check these logs for detailed error information.

## Troubleshooting

If you encounter issues:

1. Check logs: Look in the `logs` directory for detailed error information
2. Verify Streamlit installation: `pip install streamlit`
3. Check dependencies: `pip install -r requirements.txt`
4. For UTKFace download issues, install gdown: `pip install gdown`
5. If Streamlit interface doesn't launch, try running directly: `streamlit run src/ui/streamlit/app.py`

## Project Goals

This project serves as both a functional demonstration and an ethical case study exploring:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias across demographic groups
- Methods to detect and measure bias in AI systems
- Balancing security benefits with individual rights
