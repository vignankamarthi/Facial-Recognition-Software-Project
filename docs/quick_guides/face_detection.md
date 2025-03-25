# Face Detection Guide

## With Webcam
```bash
python run_demo.py --detect
```

- **What it does**: Activates your webcam and draws boxes around detected faces
- **Display**: Shows the number of faces detected in real-time
- **Controls**: Press 'q' to quit and return to the main menu

## With Static Images
```bash
python run_demo.py --image path/to/image.jpg
```

- **What it does**: Detects faces in a static image file
- **Process**:
  1. Loads the specified image
  2. Detects all faces in the image
  3. Draws boxes around each face
  4. Displays the number of faces found
- **Output**: Displays image with boxes around faces and saves to data/results
- **Controls**: Press any key to close the image window

## Tips for Best Results

- Ensure adequate lighting for better face detection
- Position faces clearly in the frame
- For webcam usage, a front-facing camera works best
- For static images, higher resolution improves detection accuracy

## Common Issues

- **No faces detected**: Check lighting and positioning
- **Webcam not found**: Verify your camera is properly connected
- **Import errors**: Run `python src/utilities/fix_imports.py`

## How It Works

The face detection module uses the `face_recognition` library which employs a pre-trained convolutional neural network to identify facial features in images. The system:

1. Converts image data to RGB format (from BGR if using OpenCV)
2. Locates face positions using histogram of oriented gradients
3. Creates bounding boxes around detected faces
4. Displays or saves the processed image

For more technical details, see the source code in `src/facial_recognition_software/face_detection.py`.
