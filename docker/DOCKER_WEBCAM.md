# Docker Webcam Support

This guide explains how to use webcam features in the Facial Recognition Software Project when running in Docker.

## Solution Overview

Since Docker containers have limited access to hardware devices like webcams, and GUI applications face challenges in containerized environments, we've implemented a browser-based approach:

1. **Browser-Based Webcam**: Uses Streamlit's native `camera_input` component
2. **Headless Operation**: Configures OpenCV to run without GUI windows
3. **Environment Variables**: Sets proper configuration for container operation

## How It Works

When running in Docker, the application automatically:

1. Detects it's running in a container environment
2. Uses the `streamlit_webcam_component` instead of external OpenCV windows
3. Processes webcam frames entirely within the browser and Streamlit

## Testing Webcam Access

You can test webcam access in the Docker container without the Streamlit UI:

```bash
# Run the webcam test script in the container
docker-compose exec facerec python /app/docker/test_webcam.py
```

This script will:
1. Try various camera access methods
2. Report successful frame captures
3. Print diagnostic information

## Using the Webcam in Streamlit

1. Open the Streamlit application in your browser: http://localhost:8501
2. Navigate to any feature with webcam functionality (Face Detection, Face Matching, etc.)
3. Click the "Use Webcam" tab
4. Allow camera access when prompted by your browser
5. Use the "Take Photo" button to capture frames for processing

## Troubleshooting

### Browser Camera Access

If your browser doesn't prompt for camera access:

1. Check browser settings (chrome://settings/content/camera in Chrome)
2. Make sure camera permissions are enabled for localhost
3. Try a different browser (Chrome tends to work best)

### Permission Issues

If you see permission errors in the Docker logs:

```bash
# View container logs
docker-compose logs
```

Look for messages like:
- `Camera detection failed`
- `NotReadableError: Could not start video source`
- `Permission denied` messages

These typically indicate browser permission issues rather than Docker configuration problems.

### Performance Considerations

The browser-based approach may be slightly slower than native webcam access, but it offers:

1. Better compatibility across platforms
2. Simpler configuration in containerized environments
3. Works on systems where direct hardware access is restricted

## Manual Testing

If you want to manually test different camera configurations, you can:

1. Edit the `.env` file to try different settings
2. Rebuild and restart the container
3. Run the test script to verify camera access

```bash
# Edit .env file
echo "USE_STREAMLIT_CAMERA=0" >> .env  # Try external windows (not recommended in Docker)

# Restart container
docker-compose down
docker-compose up -d

# Test camera access
docker-compose exec facerec python /app/docker/test_webcam.py
```
