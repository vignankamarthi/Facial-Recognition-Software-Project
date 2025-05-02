# Troubleshooting Guide

This guide helps resolve common issues with the Facial Recognition Software Project. For Docker-specific issues, see the [Docker-Specific Issues](#docker-specific-issues) section.

## Installation Issues

> **Note**: If you're using Docker, you can skip this section as dependencies are pre-installed in the container. See the [Docker-Specific Issues](#docker-specific-issues) section instead.

### Missing Dependencies

**Problem**: Error messages about missing packages or modules.

**Solution**:
1. Ensure you've installed all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. For face_recognition issues, try installing directly from GitHub:
   ```bash
   pip install git+https://github.com/ageitgey/face_recognition_models
   pip install face_recognition
   ```

### Import Errors

**Problem**: "ImportError: No module named..." or "ModuleNotFoundError" messages.

**Solution**:
1. Run the import fixer script:
   ```bash
   python src/utils/api_patch.py
   ```
2. Make sure you're running from the project's root directory.
3. Check if Python path is set correctly in run_demo.py.

## Streamlit Interface Issues

### UI Not Loading

**Problem**: Streamlit interface doesn't load or shows errors.

**Solution**:
1. Check if Streamlit is installed:
   ```bash
   pip install streamlit
   ```
2. Make sure you're using the launcher script:
   ```bash
   python run_demo.py
   ```
3. Check for port conflicts and try a different port:
   ```bash
   python run_demo.py --port 8502
   ```
4. Look for Streamlit-specific errors in the console output

### Interface Elements Not Working

**Problem**: UI components are missing or not functioning correctly.

**Solution**:
1. Clear the Streamlit cache:
   ```bash
   rm -rf ~/.streamlit
   ```
2. Try refreshing the browser with a hard refresh (Ctrl+Shift+R or Cmd+Shift+R)
3. Try with a different browser
4. Check browser console for JavaScript errors

### Session State Loss

**Problem**: Settings or app state resets unexpectedly.

**Solution**:
1. Avoid refreshing the page during active operations
2. Check if you're using the latest version of Streamlit
3. For persistent settings, use the configuration files in `config/`

### Streamlit Camera Issues

**Problem**: 'Take Photo' button doesn't work or camera feed not showing.

**Solution**:
1. Ensure browser permissions are allowed for camera access
2. Try a different browser (Chrome or Firefox recommended)
3. Check if camera is in use by another application
4. For Docker, confirm `USE_STREAMLIT_CAMERA=1` is set
5. Try running directly on host if Docker camera access isn't working

## Webcam Issues

### Camera Not Found

**Problem**: "Error: Could not open webcam" message.

**Solution**:
1. Check if your webcam is properly connected
2. Verify webcam works in other applications
3. Try using static image options instead:
   ```bash
   python run_demo.py --image data/test_images/sample.jpg
   ```

### Performance Issues

**Problem**: Webcam detection is slow or laggy.

**Solution**:
1. Ensure your machine meets minimum system requirements
2. Close other resource-intensive applications
3. Try reducing the webcam resolution in your camera settings
4. Use static image processing for smoother performance

## Face Detection Issues

> **Note**: For Docker-specific camera issues, see the [Docker Camera Issues](#docker-camera-issues) section.

### No Faces Detected

**Problem**: The system doesn't detect visible faces.

**Solution**:
1. Improve lighting (even, front-facing light works best)
2. Make sure faces are clearly visible and not too small in the frame
3. Check that the face isn't at an extreme angle
4. Run the patching script if models aren't loading:
   ```bash
   python src/utils/api_patch.py
   ```

### False Detections

**Problem**: The system detects faces where there are none.

**Solution**:
1. Improve lighting to reduce shadows
2. Remove objects that resemble facial features
3. Clean the camera lens if using a webcam

## Face Matching Issues

### "No known faces found"

**Problem**: Face matching finds no matches even with known faces.

**Solution**:
1. Ensure you've added reference images to data/known_faces/
2. Check that reference images contain clear, well-lit faces
3. Set up sample faces using the Dataset Management feature in the Streamlit UI:
   - Go to "Dataset Management" in the sidebar
   - Select "Prepare Known Faces from UTKFace"
   - Click "Execute Action"

### Poor Match Accuracy

**Problem**: The system makes incorrect matches or has low confidence.

**Solution**:
1. Use higher quality reference images
2. Add more reference images per person (from different angles)
3. Improve lighting conditions
4. Position faces more directly toward the camera
5. Adjust the matching threshold in settings (lower values = more permissive matching)

## Face Anonymization Issues

### Anonymization Not Applied

**Problem**: Face anonymization doesn't apply effects to detected faces.

**Solution**:
1. Verify faces are being detected (try the Face Detection feature first)
2. Check anonymization intensity settings (higher values = stronger effect)
3. Try different anonymization methods (blur, pixelate, mask)
4. Ensure the input image has sufficient resolution for face detection

### Artifacts in Anonymized Images

**Problem**: Strange visual artifacts appear in anonymized faces.

**Solution**:
1. Try a different anonymization method
2. Adjust intensity settings to a moderate level (30-70)
3. Make sure the face is well-lit and clearly visible
4. For pixelation issues, use the blur method instead

## Dataset Issues

### Download Failures

**Problem**: UTKFace dataset fails to download or extract.

**Solution**:
1. Check your internet connection
2. Ensure you have sufficient disk space
3. Use the Dataset Management feature in the Streamlit UI:
   - Go to "Dataset Management" in the sidebar
   - Select "Download UTKFace Dataset"
   - Click "Execute Action"
4. If download still fails, install gdown for better Google Drive support:
   ```bash
   pip install gdown
   ```
   Then try the download again

### Missing Directories

**Problem**: "Directory not found" errors when accessing data folders.

**Solution**:
1. Use the Dataset Management feature to initialize directory structure
2. Check file system permissions in your data directory
3. Verify that the Docker volume mounts are correct if using Docker

## Bias Testing Issues

### "No demographic groups found"

**Problem**: Bias testing cannot find demographic groups.

**Solution**:
1. Create or check the sample dataset structure:
   ```
   data/test_datasets/demographic_split_set/
   ├── white/    # White ethnicity group
   ├── black/    # Black ethnicity group
   ├── asian/    # Asian ethnicity group
   ├── indian/   # Indian ethnicity group
   └── others/   # Other ethnicities
   ```
2. Add images to at least one group directory (each group should have 10+ images)
3. Run the dataset setup using the Dataset Management feature in Streamlit UI:
   - Go to "Dataset Management" in the sidebar
   - Select "Set up Bias Testing with UTKFace" 
   - Click "Execute Action"

### Visualization Errors

**Problem**: Bias testing visualization fails to generate.

**Solution**:
1. Ensure matplotlib, pandas, and numpy are properly installed:
   ```bash
   pip install matplotlib pandas numpy
   ```
2. Check if you have proper permissions to write to the results directory
3. Make sure at least one demographic group has images
4. Try with a smaller dataset first to verify functionality

### Incomplete Test Results

**Problem**: Bias testing shows incomplete or unexpected results.

**Solution**:
1. Ensure all demographic groups have sufficient images (at least 10 per group recommended)
2. Check the console for processing errors during testing
3. Verify demographic group directories contain only valid face images
4. Try running the detailed analysis option in the bias testing settings

## Docker-Specific Issues

When running the application in Docker, you may encounter different issues than with a direct installation. For comprehensive Docker troubleshooting, refer to:

- [Docker README](../docker/DOCKER_README.md): General Docker setup and configuration
- [Docker Webcam Guide](../docker/webcam/DOCKER_WEBCAM.md): Camera integration in Docker

### Docker Installation Issues

**Problem**: Docker container fails to build or start.

**Solution**:
1. Check Docker and Docker Compose installations:
   ```bash
   docker --version
   docker-compose --version
   ```
2. Ensure Docker service is running:
   ```bash
   # Linux
   sudo systemctl status docker
   # macOS/Windows
   docker info
   ```
3. Check container logs for specific errors:
   ```bash
   docker-compose logs
   ```

### Docker Camera Issues

**Problem**: Camera doesn't work in Docker environment.

**Solution**:
1. Remember that in Docker, only static photo capture via browser is supported (not live streaming)
2. Ensure `USE_STREAMLIT_CAMERA=1` is set in docker-compose.yml
3. Use the Streamlit "Take Photo" button rather than expecting real-time processing
4. Verify browser permissions for camera access
5. For more detailed troubleshooting, run the webcam test script from inside the container:
   ```bash
   docker-compose exec facerec python /app/docker/webcam/test_webcam.py
   ```

### Docker Volume Issues

**Problem**: Data doesn't persist between container restarts or datasets aren't visible.

**Solution**:
1. Check volume mapping in docker-compose.yml
2. Verify host directory permissions:
   ```bash
   # Fix permissions (Linux/macOS)
   sudo chown -R $(id -u):$(id -g) ./data ./logs
   ```
3. Place datasets in the correct host directory so they're visible in the container
4. For UTKFace dataset, follow the instructions in Dataset Management and place files in `data/datasets/`

### Docker Memory Issues

**Problem**: Docker container crashes or restarts unexpectedly.

**Solution**:
1. Check container memory usage:
   ```bash
   docker stats facerec-app
   ```
2. Allocate more memory to Docker in Docker Desktop Settings
3. Reduce workload by using smaller datasets or processing fewer images at once
4. For large dataset processing, run directly on the host rather than in Docker

## Log Analysis

The project maintains detailed logs that can help diagnose issues. Check these files for specific error information:

- `logs/debug.log`: Contains all log messages (DEBUG level and above)
- `logs/info.log`: Contains INFO level messages and above
- `logs/error.log`: Contains only ERROR and CRITICAL level messages

Logs contain timestamps and context information to help pinpoint when and where errors occurred. Example log analysis commands:

```bash
# Find all error messages
grep "ERROR" logs/error.log

# Find specific module errors
grep "face_detection" logs/error.log

# Look for the most recent errors
tail -n 50 logs/error.log
```

When running in Docker, you can also view logs directly from the container:

```bash
docker-compose logs facerec
```

## General Troubleshooting Steps

If you encounter any other issues:

1. **Clean up temporary files**:
   ```bash
   # Delete cache files
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```

2. **Verify your Python version**:
   ```bash
   python --version
   ```
   This project works best with Python 3.6-3.9.

3. **Check opencv-python installation**:
   ```bash
   pip show opencv-python
   ```
   If there are conflicts, try:
   ```bash
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-python
   ```

4. **Restart with a clean environment**:
   ```bash
   # Reset project by removing cache files
   find . -name "__pycache__" -type d -exec rm -rf {} +
   find . -name "*.pyc" -delete
   # Then set up datasets again using the Dataset Management feature
   ```

If problems persist, check the console output for specific error messages and review the logs.
