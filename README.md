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

## Ethical Considerations

This project serves as both a functional demonstration and an ethical case study. Key ethical aspects explored:

- Privacy concerns in facial recognition
- Issues of consent when capturing biometric data
- Algorithmic bias in recognition systems
- Balancing security benefits with individual rights

For more detailed discussion, see [Ethical Discussion](docs/ethical_discussion.md).

## Installation

1. Clone this repository:

```
git clone https://github.com/your-username/Facial-Recognition-Software-Project.git
cd Facial-Recognition-Software-Project
```

2. Install required dependencies:

```
pip install -r requirements.txt
```

**Note:** If you encounter an error about missing `face_recognition_models`, run the provided patch:

```
python src/api_patch.py
```

Alternatively, use the run script which handles all dependencies:

```
python src/run.py
```

## Usage

Run the main application:

```
python src/main.py
```

Follow the on-screen instructions to:

- Enable/disable anonymization mode
- Select reference images for matching
- Run bias testing demonstrations
- Process static images or directories of images
- Download and manage datasets

### Command-line Arguments

The application also supports command-line arguments:

```
# Process a single image file
python src/main.py --image path/to/image.jpg

# Process all images in a directory
python src/main.py --dir path/to/directory

# Process images with face matching
python src/main.py --image path/to/image.jpg --match

# Process images with anonymization
python src/main.py --image path/to/image.jpg --anonymize

# Run dataset setup and management tools
python src/main.py --setup-dataset

# Run original webcam-based demos
python src/main.py --detect
python src/main.py --anonymize
python src/main.py --match
python src/main.py --bias
```

## Working with Datasets

This project includes command-line tools to work with the LFW (Labeled Faces in the Wild) dataset:

1. **Download a sample** of the LFW dataset
2. **Prepare reference faces** for face matching
3. **Create test datasets** with known and unknown faces

These features allow for testing without capturing many faces via webcam.

### Running an LFW Dataset Demo

1. **Set up the dataset**:
   ```
   # Download and set up LFW dataset samples
   python src/main.py --setup-dataset
   ```
   When prompted:
   - Enter a number (e.g., 20) when asked for people to include in the dataset
   - Enter a number (e.g., 5) when asked for people to include as known faces
   - Enter numbers when asked about test images

2. **Run face matching with the dataset**:
   ```
   # Test with webcam against LFW known faces
   python src/main.py --match
   ```

3. **Process test images from the dataset**:
   ```
   # Process test images with face matching
   python src/main.py --dir data/test_images --match
   ```

4. **Test bias with different demographic groups**:
   ```
   # Run bias testing demonstration
   python src/main.py --bias
   ```

5. **Other useful command combinations**:
   ```
   # Process images with anonymization
   python src/main.py --dir data/test_images --anonymize
   
   # Combine matching and anonymization
   python src/main.py --dir data/test_images --match --anonymize
   ```

## Project Structure

- `src/`: Source code for the facial recognition system
- `data/`: Sample faces and test datasets
- `docs/`: Documentation including ethical discussions

## License

This project is created for educational purposes.
