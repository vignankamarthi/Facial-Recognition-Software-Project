# Facial Recognition Software Project

A facial recognition system developed for a tech-philosophy class that demonstrates both technical capabilities and ethical considerations in AI-based recognition technologies.

## Project Overview

This project implements a facial recognition system with the following features:

- Real-time face detection using webcam feed
- Face matching against stored reference images
- Anonymization mode to protect privacy (face blurring/masking)
- Bias testing to demonstrate accuracy variations across different demographics
- Static image processing for analyzing photos without webcam
- Dataset management tools for working with public face datasets

## Quick Start

**IMPORTANT**: To run the demos, use the launcher script:

```bash
# Fix imports (one-time setup)
python fix_imports_now.py

# Run the demo launcher
python run_demo.py
```

The demo launcher adds the project to the Python path so that imports work correctly. You can pass any arguments to the launcher:

```bash
# Launch with specific demo
python run_demo.py --detect
python run_demo.py --match
python run_demo.py --anonymize
python run_demo.py --bias
```

## Ethical Considerations

This project serves as both a functional demonstration and an ethical case study. Key ethical aspects explored:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias in recognition systems
- Balancing security benefits with individual rights

For more detailed discussion, see [Ethical Discussion](docs/ethical_discussion.md).

## Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/Facial-Recognition-Software-Project.git
cd Facial-Recognition-Software-Project
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Fix imports and prepare your environment:

```bash
python fix_imports_now.py
```

## Project Structure

The project is organized as follows:

```
Facial-Recognition-Software-Project/
├── data/                   # Data directory for faces and datasets
│   ├── datasets/           # Downloaded datasets (LFW, etc.)
│   ├── sample_faces/       # Reference faces for matching
│   ├── test_datasets/      # Datasets for bias testing
│   ├── test_images/        # Test images for matching
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
│   │   └── api_patch.py             # Dependency fixes
│   └── main.py             # Main application entry point
├── run_demo.py             # Demo launcher script
├── fix_imports_now.py      # Import fixer script
└── requirements.txt        # Required dependencies
```

## Usage

Run the main application using the launcher:

```bash
python run_demo.py
```

Follow the on-screen instructions to:

- Enable/disable anonymization mode
- Select reference images for matching
- Run bias testing demonstrations
- Process static images or directories of images
- Download and manage datasets

### Command-line Arguments

The application supports command-line arguments through the launcher:

```bash
# Process a single image file
python run_demo.py --image path/to/image.jpg

# Process all images in a directory
python run_demo.py --dir path/to/directory

# Process images with face matching
python run_demo.py --image path/to/image.jpg --match

# Process images with anonymization
python run_demo.py --image path/to/image.jpg --anonymize

# Run dataset setup and management tools
python run_demo.py --setup-dataset

# Run original webcam-based demos
python run_demo.py --detect
python run_demo.py --anonymize
python run_demo.py --match
python run_demo.py --bias
```

## Working with Datasets

This project includes command-line tools to work with the LFW (Labeled Faces in the Wild) dataset:

1. **Download a sample** of the LFW dataset
2. **Prepare reference faces** for face matching
3. **Create test datasets** with known and unknown faces

These features allow for testing without capturing many faces via webcam.

### Running an LFW Dataset Demo

1. **Set up the dataset**:
   ```bash
   # Download and set up LFW dataset samples
   python run_demo.py --setup-dataset
   ```
   When prompted:
   - Enter a number (e.g., 20) when asked for people to include in the dataset
   - Enter a number (e.g., 5) when asked for people to include as known faces
   - Enter numbers when asked about test images

2. **Run face matching with the dataset**:
   ```bash
   # Test with webcam against LFW known faces
   python run_demo.py --match
   ```

3. **Process test images from the dataset**:
   ```bash
   # Process test images with face matching
   python run_demo.py --dir data/test_images --match
   ```

4. **Test bias with different demographic groups**:
   ```bash
   # Run bias testing demonstration
   python run_demo.py --bias
   ```

## Demo Preparation Checklist

Before presenting your demo:

- [x] Run `python fix_imports_now.py` to fix import statements
- [ ] Test webcam availability with `python run_demo.py --detect`
- [ ] Download a sample dataset with `python run_demo.py --setup-dataset`
- [ ] Prepare a few test images in `data/test_images` folder
- [ ] Practice the demo flow to ensure smooth transitions
- [ ] Check that all ethical points from the project plan are demonstrated

## License

This project is created for educational purposes.
