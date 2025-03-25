# Facial Recognition Software Project

 [ ] Make sure the directions are IRON CLAD for a demonstration

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

## Running Demos Without a Webcam

This project supports both webcam-based demos and static image processing using sample data:

### Using Sample Data (No Webcam Required)
1. Set up the sample dataset:
   ```bash
   python run_demo.py --setup-dataset
   ```
   Follow the prompts to download and prepare sample faces.

2. Run the bias testing demo (uses static images):
   ```bash
   python run_demo.py --bias
   ```
   Results will be displayed in the terminal and saved as PNG visualizations in `./data/test_datasets/results/`.

3. Process individual static images:
   ```bash
   python run_demo.py --image data/test_images/known/[image_name].jpg --match
   ```

### Using the Interactive Menu
Start the application with `python run_demo.py` and:
- Options 1-3 require a webcam
- Options 4-6 work with static images (no webcam required)

## Detailed Menu Navigation

### Main Menu Options

#### Webcam Required Options

1. **Face Detection (Webcam)**
   - Activates your webcam and draws boxes around detected faces
   - Displays the number of faces detected in real-time
   - Controls:
     - Press 'q' to quit and return to the main menu

2. **Face Detection with Anonymization (Webcam)**
   - Activates your webcam and anonymizes detected faces
   - Prompts you to select an anonymization method before starting:
     - **blur**: Applies Gaussian blur to faces
     - **pixelate**: Creates a pixelated effect over faces
     - **mask**: Replaces faces with a black mask and simple face icon
   - Controls during operation:
     - Press 'b' to switch to blur mode
     - Press 'p' to switch to pixelate mode
     - Press 'm' to switch to mask mode
     - Press 'q' to quit and return to the main menu

3. **Face Matching (Webcam)**
   - Activates your webcam and tries to identify faces against known references
   - Before starting, checks if sample faces are available in data/sample_faces
   - Displays a box around each face:
     - Green box: Known face (with name and confidence score)
     - Red box: Unknown face
   - Controls:
     - Press 'q' to quit and return to the main menu

#### No Webcam Required Options

4. **Static Image Processing**
   - Sub-menu for processing image files:
     1. **Process a single image (detection only)**
        - Prompts for an image file path
        - Detects and highlights faces in the image
        - Displays the processed image in a window
        - Saves the result to data/results directory
        - Press any key to close the image window
     
     2. **Process a single image (with face matching)**
        - Prompts for an image file path
        - Detects faces and attempts to match them against known faces
        - Displays the processed image with names for identified faces
        - Shows confidence scores for each match
        - Saves the result to data/results directory
        - Press any key to close the image window
     
     3. **Process a single image (with anonymization)**
        - Prompts for an image file path
        - Detects and anonymizes faces in the image (using blur method)
        - Displays the processed image with anonymized faces
        - Saves the result to data/results directory
        - Press any key to close the image window
     
     4. **Process a directory of images**
        - Prompts for a directory path containing images
        - Processes all images in the directory (detection and matching)
        - Displays each processed image briefly
        - Saves all results to data/results directory
        - Press any key to proceed to the next image
     
     5. **Return to main menu**
        - Returns to the main menu without processing images

5. **Dataset Setup & Management**
   - Tools for downloading and preparing face datasets:
     1. **Download LFW dataset sample**
        - Downloads a sample of the Labeled Faces in the Wild dataset
        - Prompts for number of people to include (recommended: 10-100)
        - Downloads and extracts the dataset to data/datasets/lfw
        - Creates a random sample in data/datasets/lfw/lfw_sample
        - This operation may take several minutes depending on your internet connection
     
     2. **Prepare known faces from LFW**
        - Creates reference faces for the face matching feature
        - Prompts for number of people to include as known faces
        - Selects random people from the LFW dataset with multiple images
        - Copies one image per person to data/sample_faces
        - These faces will be used for matching in options 3 and 4.2
     
     3. **Prepare test dataset from LFW**
        - Creates test images for face recognition evaluation
        - Prompts for:
          - Number of known people to include in test set
          - Number of test images per person
        - Copies images to:
          - data/test_images/known: Additional images of known people
          - data/test_images/unknown: Images of people not in the known set
        - These images can be used with option 4.4 to test recognition
     
     4. **Return to main menu**
        - Returns to the main menu without further dataset operations

6. **Bias Testing Demonstration**
   - Runs an automated demonstration of potential bias in face recognition
   - Steps executed automatically:
     1. Checks if sample test data exists in data/test_datasets/sample_dataset
     2. If not found, creates sample dataset structure with demographic groups:
        - group_a: First demographic group
        - group_b: Second demographic group
        - group_c: Third demographic group
     3. If no images are in these groups, attempts to copy sample images from LFW
     4. Runs face detection on all groups and measures accuracy
     5. Displays results showing detection accuracy for each demographic group
     6. Creates a bar chart visualization and saves it to data/test_datasets/results
     7. Checks for significant accuracy differences between groups
     8. Returns to the main menu when complete
   - Note: For a meaningful bias test, you should manually populate the demographic
     groups with appropriate images representing different demographics

7. **Exit**
   - Exits the application

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
│   │   ├── face_recognition_patch.py # Shared face recognition patching utility
│   │   ├── api_patch.py             # Face recognition API patching wrapper
│   │   ├── cleanup.py               # Project cleanup utility
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

## Cleaning Up the Project

To clean unnecessary files from your project directory, use the cleanup script:

```bash
# Basic cleanup - removes Python cache files, backups, and temporary data
python src/utilities/cleanup.py

# Dry run - shows what would be deleted without actually deleting anything
python src/utilities/cleanup.py --dry-run

# Reset all datasets - cleans up and resets dataset directories to initial state
python src/utilities/cleanup.py --reset-datasets
```

This script helps maintain a clean project by removing:
- Python cache files (`__pycache__`, `.pyc`, etc.)
- Backup files (`.bak`, `.backup`, etc.)
- Temporary dataset files (`.tgz`, `.zip`, etc.)
- Generated test images and results
- Log files

Run this script periodically during development or when experiencing unexpected behavior.

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

## Advanced Features

### Face Recognition Patching System

This project includes a robust system for handling face recognition library compatibility issues:

- `face_recognition_patch.py`: A centralized utility that applies monkey patches to the face_recognition library when needed
- This prevents crashes when the face_recognition_models package is not properly installed
- The patching system is used by both api_patch.py and run.py, ensuring consistent behavior across the application

## Demo Preparation Checklist

Before presenting a demo:

- Clean up unnecessary files: `python src/utilities/cleanup.py`
- Run the setup script: `python src/utilities/quick_setup.py`
- Test webcam availability: `python run_demo.py --detect`
- Download a sample dataset: `python run_demo.py --setup-dataset`
- Prepare reference faces for matching
- Test all anonymization modes
- Run a bias test to ensure analysis works
