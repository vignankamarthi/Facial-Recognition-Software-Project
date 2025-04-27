# Facial Recognition Software Project
A facial recognition system demonstrating technical capabilities and ethical considerations in AI-based recognition technologies.

## Quick Start Guide

### Installation
```bash
git clone https://github.com/vignankamarthi/Facial-Recognition-Software-Project.git
cd Facial-Recognition-Software-Project
pip install -r requirements.txt
python src/utilities/quick_setup.py
```

### Running the Demo
```bash
python run_demo.py
```

### Features Overview

| Feature | Webcam Required | Description |
|---------|:--------------:|-------------|
| Face Detection | ✅ | Detect faces in real-time |
| Face Anonymization | ✅ | Blur, pixelate, or mask detected faces |
| Face Matching | ✅ | Match detected faces against reference images |
| Static Image Processing | ❌ | Analyze saved images instead of webcam feed |
| Dataset Management | ❌ | Download and prepare datasets for testing |
| Demographic Bias Testing | ❌ | Test recognition accuracy across different ethnicities using UTKFace dataset |

## Running Without a Webcam

If you don't have a webcam, you can still use these features:

1. **Set up the UTKFace dataset**:
   ```bash
   python run_demo.py --setup-dataset
   # Select option 1 to download UTKFace dataset
   # Select option 2 to set up bias testing
   ```

2. **Process static images**:
   ```bash
   python run_demo.py --image data/test_images/sample.jpg
   ```

3. **Run demographic bias testing**:
   ```bash
   python run_demo.py --bias
   ```
   
   This will demonstrate how facial recognition accuracy can vary across different ethnic groups, providing statistical analysis and visualizations of potential algorithmic bias.

## Command Line Shortcuts

| Command | Description |
|---------|-------------|
| `--detect` | Run face detection demo |
| `--anonymize` | Run face anonymization demo |
| `--match` | Run face matching demo |
| `--bias` | Run bias testing demo with UTKFace dataset |
| `--image PATH` | Process a single image |
| `--dir PATH` | Process all images in a directory |
| `--setup-dataset` | Download and prepare sample datasets |

## UTKFace Dataset Integration

This project uses the UTKFace (University of Tennessee, Knoxville Face) dataset for ethical bias testing, providing:

- **Demographic Labels**: Age, gender, and ethnicity annotations for over 20,000 face images
- **Ethnicity Categories**: White, Black, Asian, Indian, and Others (including Hispanic, Latino, Middle Eastern)
- **Statistical Analysis**: Variance, standard deviation, and bias level calculations
- **Visualization**: Color-coded charts showing accuracy differences across demographic groups

The dataset allows for realistic demonstration of potential algorithmic bias in facial recognition systems and serves as an educational tool for understanding ethical concerns in AI.

## Project Structure

```
Facial-Recognition-Software-Project/
├── data/                   # Data directory
│   ├── datasets/           # Raw datasets
│   │   └── utkface/        # UTKFace dataset
│   ├── test_datasets/      # Processed test data
│   │   └── demographic_split_set/ # Ethnicity-organized images
│   └── logs/               # System logs for error tracking
├── docs/                   # Documentation
│   ├── quick_guides/       # Feature-specific guides
│   └── ethical_discussion.md
├── src/                    # Source code
│   ├── facial_recognition_software/  # Core modules
│   ├── utilities/          # Utility modules
│   └── main.py             # Main application
├── run_demo.py             # Demo launcher
└── requirements.txt        # Dependencies
```

For detailed usage instructions, see [docs/quick_guides/](docs/quick_guides/).

For ethical considerations, see [docs/ethical_discussion.md](docs/ethical_discussion_draft.md).

## Recent Improvements

### 1. Robust Logging System
- Added comprehensive logging with different severity levels
- Created log rotation to prevent log files from growing too large
- Implemented detailed error tracking with full stack traces
- Added context information for better debugging

### 2. Standardized Method Signatures
- Consistent parameter naming across all modules
- Improved docstrings with NumPy-style formatting
- Better return type consistency across the codebase
- Added input validation to prevent errors

### 3. Enhanced Error Handling
- Implemented proper exception hierarchy
- Added error context information
- Improved error recovery mechanisms
- Added fallback behaviors for common error conditions

## Troubleshooting

If you encounter issues:

1. Run the setup script: `python src/utilities/quick_setup.py`
2. Check dependencies: `pip install -r requirements.txt`
3. Clean up temporary files: `python src/utilities/cleanup.py`
4. Fix import issues: `python src/utilities/fix_imports.py`
5. Check logs: Look in the `logs` directory for detailed error information
6. For UTKFace download issues, install gdown: `pip install gdown`

## Project Goals

This project serves as both a functional demonstration and an ethical case study exploring:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias across demographic groups
- Methods to detect and measure bias in AI systems
- Balancing security benefits with individual rights
