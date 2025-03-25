# Facial Recognition Software Project

A facial recognition system that demonstrates both technical capabilities and ethical considerations in AI-based recognition technologies.

## Project Overview

This project implements a facial recognition system with the following features:

- Real-time face detection using webcam feed
- Face matching against stored reference images
- Anonymization mode to protect privacy (blur/pixelate/mask)
- Bias testing to demonstrate accuracy variations across demographics
- Static image processing for analyzing photos

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Facial-Recognition-Software-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script to ensure all directories are created:
```bash
python src/utilities/quick_setup.py
```

## Running the Demo

The easiest way to run the demo is using our launcher script:

```bash
python run_demo.py
```

This will open an interactive menu with the following options:

1. **Face Detection (Webcam)** - Detect faces in real-time using your webcam
2. **Face Detection with Anonymization (Webcam)** - Detect and anonymize faces in real-time
3. **Face Matching (Webcam)** - Match detected faces against known faces
4. **Static Image Processing** - Process images from files or directories
5. **Dataset Setup & Management** - Tools for working with face datasets
6. **Bias Testing Demonstration** - Test recognition accuracy across different groups
7. **Exit** - Exit the application

## Command Line Arguments

You can also run specific demos directly using command line arguments:

```bash
# Run face detection demo
python run_demo.py --detect

# Run face detection with anonymization
python run_demo.py --anonymize

# Run face matching demo
python run_demo.py --match

# Run bias testing demo
python run_demo.py --bias

# Process a single image file
python run_demo.py --image path/to/image.jpg

# Process a directory of images
python run_demo.py --dir path/to/directory

# Run dataset setup tools
python run_demo.py --setup-dataset
```

## Project Structure

```
Facial-Recognition-Software-Project/
├── data/                   # Data directory for faces and datasets
│   ├── sample_faces/       # Reference faces for matching
│   ├── test_datasets/      # Datasets for bias testing
│   ├── test_images/        # Test images for static processing
│   └── results/            # Processed images output
├── docs/                   # Documentation
│   └── ethical_discussion.md
├── src/                    # Source code
│   ├── facial_recognition_software/  # Core recognition modules
│   │   ├── face_detection.py        # Face detection functionality
│   │   ├── face_matching.py         # Face matching/identification
│   │   ├── anonymization.py         # Face anonymization features
│   │   └── bias_testing.py          # Bias analysis tools
│   ├── utilities/          # Utility modules
│   │   ├── image_processing.py      # Static image processing
│   │   ├── fix_imports.py           # Import fixer utility
│   │   └── quick_setup.py           # Setup script
│   └── main.py             # Main application entry point
├── run_demo.py             # Demo launcher script
└── requirements.txt        # Required dependencies
```

## Ethical Considerations

This project serves as both a functional demonstration and an ethical case study. Key ethical aspects explored:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias in recognition systems
- Balancing security benefits with individual rights

For more detailed discussion, see the [ethical_discussion.md](docs/ethical_discussion.md) document.

## Working with Datasets

For bias testing and face matching demonstrations, you can use the LFW (Labeled Faces in the Wild) dataset:

1. Run the dataset setup:
```bash
python run_demo.py --setup-dataset
```

2. Follow the on-screen prompts to:
   - Download a sample of the LFW dataset
   - Prepare known faces for matching
   - Create test datasets with varying demographics

## Troubleshooting

If you encounter import errors or other issues:

1. Make sure you're using the `run_demo.py` launcher script
2. Check that all dependencies are installed correctly
3. Run the setup script to create necessary directories:
```bash
python src/utilities/quick_setup.py
```

4. Fix import issues by running:
```bash
python src/utilities/fix_imports.py
```

## Demo Preparation Checklist

Before presenting a demo:

- [ ] Run the setup script: `python src/utilities/quick_setup.py`
- [ ] Test webcam availability: `python run_demo.py --detect`
- [ ] Download a sample dataset: `python run_demo.py --setup-dataset`
- [ ] Prepare reference faces for matching
- [ ] Test all anonymization modes
- [ ] Run a bias test to ensure analysis works
