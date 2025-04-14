# Troubleshooting Guide

This guide helps resolve common issues with the Facial Recognition Software Project.

## Installation Issues

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
   python src/utilities/fix_imports.py
   ```
2. Make sure you're running from the project's root directory.
3. Check if Python path is set correctly in run_demo.py.

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

### No Faces Detected

**Problem**: The system doesn't detect visible faces.

**Solution**:
1. Improve lighting (even, front-facing light works best)
2. Make sure faces are clearly visible and not too small in the frame
3. Check that the face isn't at an extreme angle
4. Run the patching script if models aren't loading:
   ```bash
   python src/utilities/api_patch.py
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
3. Try setting up sample faces from UTKFace dataset:
   ```bash
   python run_demo.py --setup-dataset
   ```
   Then select option 3 "Prepare known faces from UTKFace"

### Poor Match Accuracy

**Problem**: The system makes incorrect matches or has low confidence.

**Solution**:
1. Use higher quality reference images
2. Add more reference images per person (from different angles)
3. Improve lighting conditions
4. Position faces more directly toward the camera

## Dataset Issues

### Download Failures

**Problem**: UTKFace dataset fails to download or extract.

**Solution**:
1. Check your internet connection
2. Ensure you have sufficient disk space
3. Try running with the `--setup-dataset` flag:
   ```bash
   python run_demo.py --setup-dataset
   ```
4. If download still fails, install gdown for better Google Drive support:
   ```bash
   pip install gdown
   ```
   Then try the download again

### Missing Directories

**Problem**: "Directory not found" errors when accessing data folders.

**Solution**:
1. Run the quick setup script to create all necessary directories:
   ```bash
   python src/utilities/quick_setup.py
   ```
2. Check file system permissions in your data directory

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
2. Add images to at least one group directory
3. Run the bias testing setup process:
   ```bash
   python run_demo.py --bias --utkface
   ```

### Visualization Errors

**Problem**: Bias testing visualization fails to generate.

**Solution**:
1. Ensure matplotlib is properly installed:
   ```bash
   pip install matplotlib
   ```
2. Check if you have proper permissions to write to the results directory
3. Make sure at least one demographic group has images

## General Troubleshooting Steps

If you encounter any other issues:

1. **Clean up temporary files**:
   ```bash
   python src/utilities/cleanup.py
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
   python src/utilities/cleanup.py --reset-datasets
   python src/utilities/quick_setup.py
   ```

If problems persist, check the console output for specific error messages.
