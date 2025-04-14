# Face Matching Guide

## Prerequisites

Before using face matching, you need reference faces:

1. Create a directory for reference faces:
   ```bash
   mkdir -p data/known_faces
   ```

2. Add reference images to this directory:
   - Each image should contain one clearly visible face
   - Name files with the person's name (e.g., `john_smith.jpg`)
   - Supported formats: JPG, JPEG, PNG

3. Alternatively, use the dataset setup to create sample faces automatically:
   ```bash
   python run_demo.py --setup-dataset
   ```
   Then select "Prepare known faces from UTKFace"

## With Webcam
```bash
python run_demo.py --match
```

- **What it does**: Identifies faces against known reference images
- **Setup**: Automatically loads reference faces from data/known_faces
- **Display**:
  - Green box: Known face (with name and confidence score)
  - Red box: Unknown face
  - Shows face count in top-left corner
- **Controls**: Press 'q' to quit and return to the main menu

## With Static Images
```bash
python run_demo.py --image path/to/image.jpg --match
```

- **What it does**: Tries to identify faces in a static image
- **Process**:
  1. Loads reference faces from data/known_faces
  2. Detects faces in the specified image
  3. Compares each face against known references
  4. Labels faces with names and confidence scores
- **Output**: Displays and saves image with identified faces
- **Controls**: Press any key to close the image window

## How Face Matching Works

The system uses facial encoding to identify people:

1. **Face Encoding**: Converts facial features into a 128-dimension vector
2. **Comparison**: Calculates distance between the detected face encoding and reference encodings
3. **Matching**: Identifies the closest match if the distance is below threshold
4. **Confidence Score**: Shows how closely the face matches (higher is better)

## Tips for Best Results

- Use clear, well-lit reference photos with the face centered
- For best accuracy, use multiple reference images per person
- Higher-quality images improve matching accuracy
- Match threshold is set to 0.6 (60% similarity) by default

## Common Issues

- **"No known faces found"**: Add reference images to data/known_faces
- **False positives**: Try using better quality reference images
- **Low confidence scores**: Improve lighting and face positioning

For technical details, see `src/facial_recognition_software/face_matching.py`.
