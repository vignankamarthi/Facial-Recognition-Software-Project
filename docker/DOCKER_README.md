# Docker Setup for Facial Recognition Software Project

This directory contains Docker configuration for containerizing the Facial Recognition Software Project. The Docker setup provides a consistent environment for running the application across different platforms, with particular attention to webcam functionality for facial recognition features.

## Directory Structure

```
docker/
├── Dockerfile             # Multi-stage build definition
├── docker-compose.yml     # Container orchestration config
├── entrypoint.sh          # Container initialization script
├── init_demo_data.py      # Script to set up demo data
├── DOCKER_README.md       # This file
└── webcam/                # Webcam-related utilities
    ├── DOCKER_WEBCAM.md   # Browser-based webcam integration guide
    ├── test_webcam.py     # Script to test webcam access
    └── webcam_setup.sh    # Script to configure webcam
```

## Quick Start

To run the application in Docker:

```bash
# Build and start the container in the background
docker-compose up -d

# View the logs (useful for troubleshooting)
docker-compose logs -f

# Stop the container when finished
docker-compose down
```

The Streamlit interface will be available at http://localhost:8501. For facial recognition features requiring camera access, the application only works with Streamlit's browser-based camera component in Docker. Note that external OpenCV windows and live webcam streaming are not supported in containerized environments - only static photo capture is available.

## Key Features

1. **Multi-stage Build**: Optimized container size with separate build and runtime stages
2. **Volume Mapping**: Real-time code updates without rebuilding
3. **Static Photo Capture**: Facial recognition through Streamlit's native camera component (photo capture only, no live streaming)
4. **Cross-platform Compatibility**: Works on macOS, Windows, and Linux systems
5. **Environment Configuration**: Easily customizable via environment variables

## Configuration Options

You can configure the application using environment variables in the `docker-compose.yml` file or by passing them directly to the container:

```yaml
# Core application settings
UI_PORT: 8501                     # Streamlit interface port
WEBCAM_ENABLED: true              # Enable facial recognition with webcam
FORCE_HEADLESS: true              # Run OpenCV in headless mode (required for Docker)
LOG_LEVEL: INFO                   # Logging verbosity

# Facial recognition performance settings
OPENCV_VIDEOIO_DEBUG: 1           # Enable detailed camera debugging
WEBCAM_DEVICE_INDEX: 0            # Default camera device index
USE_STREAMLIT_CAMERA: 1           # Use browser-based camera (recommended for Docker)
```

The `USE_STREAMLIT_CAMERA=1` setting is required in Docker as only Streamlit's browser-based camera component with static photo capture works in containerized environments. External OpenCV windows and real-time video processing are not supported in Docker.

## Volume Mappings

The Docker setup mounts several volumes to enable persistence and development:

- **Source Code**: `../src:/app/src` - Enables live code changes for facial recognition modules
- **Configuration**: `../config:/app/config` - Feature configuration and thresholds
- **Data**: `../data:/app/data` - Datasets, known faces, and processing results
- **Logs**: `../logs:/app/logs` - Detailed execution logs

These mappings ensure that your facial recognition datasets, known faces, and detection results persist between container restarts, while also allowing you to modify code or configuration files without rebuilding the container.

## Camera Support for Facial Recognition in Docker

The project implements a specialized approach for providing camera functionality in Docker:

- **Browser-based Photo Capture Only**: Uses Streamlit's native `camera_input` component for static image capture
- **Automatic Environment Detection**: Detects Docker environment and enforces appropriate camera mode
- **OpenCV Headless Mode**: Ensures compatibility with containerized environments

Important limitations:
- Only static photo capture is supported (no live video streaming)
- External OpenCV windows do not work in Docker environments
- Real-time processing requires the non-Docker installation
- The "Take Photo" button must be used to process each frame individually

For detailed implementation specifics and troubleshooting, see [DOCKER_WEBCAM.md](webcam/DOCKER_WEBCAM.md)

## Development Workflow

1. **Start the container**: `docker-compose up -d`
2. **Edit face recognition code**: Make changes to files in `src/backend/` or UI components in `src/ui/streamlit/`
3. **Test recognition features**: Refresh the browser at http://localhost:8501
4. **Monitor performance**: `docker-compose logs -f` shows processing times and detection results
5. **Run tests**: `docker-compose exec facerec python run_tests.py`
6. **Test photo capture**: Use the Face Detection, Face Matching, or Face Anonymization features with the "Take Photo" button (live streaming is not available)

The container automatically reloads when you make changes to the Python code, allowing for rapid development of recognition algorithms and UI components.

## Troubleshooting

### Common Issues with Facial Recognition Features

1. **Photo Capture Not Working**:
   - Ensure browser camera permissions are enabled
   - Verify `USE_STREAMLIT_CAMERA=1` is set (required in Docker)
   - Note that only static photo capture is supported (not live streaming)
   - You must click "Take Photo" for each frame you want to process
   ```bash
   # Run webcam test to verify camera access
   docker-compose exec facerec python /app/docker/webcam/test_webcam.py
   ```

2. **Permission Issues with Dataset Directories**:
   ```bash
   # Fix permissions for datasets and results
   sudo chown -R $(id -u):$(id -g) data logs
   ```

3. **Slow Face Processing Performance**:
   - Check container resource allocation
   - Consider reducing frame size or detection frequency
   - Monitor resource usage: `docker stats facerec-app`

4. **UTKFace Dataset Not Loading**:
   - Verify dataset is properly mounted in the Docker volume
   - Check logs for specific dataset access errors
   ```bash
   # Check dataset access logs
   docker-compose logs | grep "dataset"
   ```

## Advanced Usage

### Optimizing Face Recognition Performance

For better facial recognition performance in Docker:

```bash
# Rebuild with optimized settings
docker-compose build --no-cache --build-arg OPTIMIZE_RECOGNITION=1
docker-compose up -d
```

### Camera Configuration

Configure the Streamlit camera component:

```bash
# Adjust camera settings for Streamlit's photo capture
STREAMLIT_CAMERA_RESOLUTION=640x480 docker-compose up -d
```

Remember that external OpenCV windows and real-time video processing are not available in Docker. Only the static photo capture with the "Take Photo" button will work.

### Testing Different Recognition Features

Test specific facial recognition components:

```bash
# Run inside the container to test face detection directly
docker-compose exec facerec python -c "from src.backend.face_detection import FaceDetector; detector = FaceDetector(); print('Face detection module loaded successfully')"

# Test face matching
docker-compose exec facerec python -c "from src.backend.face_matching import FaceMatcher; matcher = FaceMatcher(); print('Known faces loaded:', len(matcher.known_face_names))"
```

### Running Without Docker

If you need to test the facial recognition on your host directly:

```bash
pip install -r requirements.txt
python run_demo.py
```
