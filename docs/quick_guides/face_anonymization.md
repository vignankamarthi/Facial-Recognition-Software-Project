# Face Anonymization Guide

## With Webcam
```bash
python run_demo.py --anonymize
```

- **What it does**: Anonymizes faces detected via webcam in real-time
- **Setup**:
  1. Activates your webcam
  2. Prompts you to select anonymization method:
     - **blur**: Applies Gaussian blur to faces
     - **pixelate**: Creates a pixelated effect over faces
     - **mask**: Replaces faces with a black mask and simple face icon
- **Controls during operation**:
  - 'b' - Switch to blur mode
  - 'p' - Switch to pixelate mode
  - 'm' - Switch to mask mode
  - 'q' - Quit and return to the main menu

## With Static Images
```bash
python run_demo.py --image path/to/image.jpg --anonymize
```

- **What it does**: Anonymizes faces in a static image file
- **Process**:
  1. Loads the specified image
  2. Detects all faces in the image
  3. Applies anonymization (blur by default)
  4. Adds "Anonymized" label on each face
- **Output**: Displays and saves anonymized image to data/results
- **Controls**: Press any key to close the image window

## Anonymization Methods

### Blur
- Applies a Gaussian blur filter to the face area
- Higher intensity values create a more thorough anonymization
- Best for preserving overall facial structure while hiding identity

### Pixelate
- Downsamples and then upsamples the face area to create a pixelated effect
- Scale factor determines the size of the pixels
- Useful for creating a "censored" appearance

### Mask
- Replaces the face with a solid black rectangle
- Adds a simple cartoon face icon
- Provides complete anonymization

## Use Cases

- Privacy protection in demonstration or educational videos
- Ethics demonstrations for data protection
- Testing facial recognition system limitations

## Technical Implementation

The anonymization system:
1. Gets face locations from the face detector
2. For each face, applies the selected anonymization method
3. Adds visual indicators for anonymized regions
4. Displays the anonymization mode in the corner of the frame

For more details, see `src/facial_recognition_software/anonymization.py`.
