# Docker Camera Support for Facial Recognition

This guide explains the limitations and approach for using camera capabilities with facial recognition features when running in Docker containers.

## Solution Overview

Due to fundamental limitations with hardware access in containers, the facial recognition features in Docker have important restrictions:

1. **Static Photo Capture Only**: Uses Streamlit's native `camera_input` component for individual photo capture
2. **No Live Video Processing**: External OpenCV windows and real-time video streaming are not supported
3. **Browser-Based Approach**: All camera interactions happen through the browser, not directly from the container

## Implementation Details

When running facial recognition features in Docker:

1. **Container Detection**: The application detects it's running in Docker and disables external OpenCV windows
2. **Streamlit-Only Mode**: Forces the use of Streamlit's `camera_input` component as the only camera option
3. **Static Processing Only**: Each frame must be manually captured using the "Take Photo" button
4. **Limited Functionality**: While all recognition algorithms work, they only process static images

Important: The live webcam features with external OpenCV windows are completely disabled in Docker.

## Testing Camera Availability

You can verify that your camera is accessible to the Docker container:

```bash
# Run the camera test script in the container
docker-compose exec facerec python /app/docker/webcam/test_webcam.py
```

This test only validates basic camera accessibility, but does not enable live streaming:
1. Checks if the camera can be detected by the container
2. Attempts to capture a single frame
3. Reports whether the camera could be accessed

A successful test only means photo capture will work - it does not enable real-time processing.

## Using Camera-Based Facial Recognition in Docker

1. Launch the application in your browser: http://localhost:8501
2. Navigate to one of the facial recognition features:
   - **Face Detection**: Identifies and locates faces in captured photos
   - **Face Matching**: Compares detected faces against known references
   - **Face Anonymization**: Applies privacy filters to detected faces
3. Select the "Use Webcam" tab within the feature
4. Allow camera access when prompted by your browser
5. For static photo processing:
   - Position your face properly in the frame (center, well-lit)
   - Click the "Take Photo" button to capture a single frame
   - Wait for processing to complete on that frame
   - Click "Take Photo" again for each new frame you want to process

Note: The external window-based live webcam features shown in non-Docker demos are not available in the containerized version.

## Troubleshooting Docker Camera Integration

### Why Live Webcam Features Don't Work

Live streaming with external OpenCV windows doesn't work in Docker because:

1. **Container Isolation**: Containers can't directly access the host's display server
2. **X11 Forwarding Issues**: GUI windows from containers require a complex setup due to security, latency, and display configuration restraints
3. **OpenCV Limitations**: OpenCV's display functions require direct display access

### Camera Access Issues

If the "Take Photo" feature isn't working:

1. **Browser Permissions**: Check camera settings in your browser (chrome://settings/content/camera in Chrome)
2. **Container Configuration**: Verify `USE_STREAMLIT_CAMERA=1` is set (required for Docker)
3. **Browser Compatibility**: Try Chrome or Firefox for better compatibility
4. **Error Messages**: Check browser console and Docker logs for camera access errors

### Improving Photo Capture Quality

1. **Lighting**: Ensure your face is well-lit from the front
2. **Position**: Center your face and maintain appropriate distance
3. **Background**: Use a plain background for better detection accuracy
4. **Resolution**: Higher resolution may improve recognition but can impact performance

## Working Around Docker Camera Limitations

### Using Streamlit for Photo Processing

While live processing isn't available, you can still effectively use the facial recognition features:

```bash
# Start the container optimized for photo processing
docker-compose up -d

# Configure Streamlit camera settings (if needed)
STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false docker-compose restart
```

### Testing Recognition Components in Docker

You can validate that recognition algorithms work properly in Docker:

```bash
# Test face detection module (algorithmic part only)
docker-compose exec facerec python -c "from src.backend.face_detection import FaceDetector; detector = FaceDetector(); print('Detector available')"

# Test with a sample image instead of live camera
docker-compose exec facerec python -c "import cv2; from src.backend.face_detection import FaceDetector; img = cv2.imread('/app/data/test_images/sample.jpg'); detector = FaceDetector(); detector.detect_faces(img); print('Detection works on static images')"
```
