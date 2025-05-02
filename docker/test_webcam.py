#!/usr/bin/env python3
"""
Webcam test script for Docker environment.

This script tests webcam access directly in the Docker container.
It attempts to open the webcam using various methods and captures frames.

Usage:
    python test_webcam.py
"""

import cv2
import os
import sys
import time

def test_webcam():
    """Test webcam access in Docker."""
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Platform: {sys.platform}")
    print(f"Python version: {sys.version}")
    
    # Set environment variables for better compatibility
    if sys.platform == "darwin":  # macOS
        print("macOS detected - optimizing for AVFoundation")
        os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
        os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1000"
        os.environ["OPENCV_VIDEOIO_PRIORITY_QT"] = "0"
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["OPENCV_VIDEOIO_PRIORITY_V4L"] = "0"
        os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "0"
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        os.environ["OPENCV_VIDEOIO_DEBUG"] = "1"
    elif sys.platform == "win32":  # Windows
        print("Windows detected - optimizing for DirectShow")
        os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "900"
    else:  # Linux
        print("Linux detected - optimizing for V4L2")
        os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "1000"
        os.environ["OPENCV_VIDEOIO_PRIORITY_V4L"] = "900"
    
    # Setup for headless operation - skip window creation
    print("Running in headless mode - no GUI windows will be created")
    window_name = None  # No windows in headless mode
    
    # Create list of camera sources to try
    sources = []
    
    if sys.platform == "darwin":  # macOS
        sources = [
            (0, cv2.CAP_AVFOUNDATION),  # Default camera with AVFoundation
            (0, None),  # Default camera with default backend
            (1, cv2.CAP_AVFOUNDATION),  # Secondary camera with AVFoundation
            "avfoundation://0",  # AVFoundation URL format
            "avfoundation://0:0",  # AVFoundation with audio
            "avfoundation://1",  # Secondary camera AVFoundation URL 
            (-1, None)  # Any camera
        ]
    elif sys.platform == "win32":  # Windows
        sources = [
            (0, cv2.CAP_DSHOW),  # Default camera with DirectShow
            (0, None),  # Default camera with default backend
            "dshow://0",  # DirectShow URL
            "msmf://0",  # Media Foundation URL
            (1, None),  # Secondary camera
            (-1, None)  # Any camera
        ]
    else:  # Linux
        sources = [
            (0, cv2.CAP_V4L2),  # Default camera with V4L2
            (0, None),  # Default camera with default backend
            "v4l2:///dev/video0",  # V4L2 URL
            (1, None),  # Secondary camera
            (-1, None)  # Any camera
        ]
    
    # Add shared fallbacks
    sources.extend([
        "camera:0",
        "camera",
        0,  # Simple index as fallback
        1,
    ])
    
    # Try each source
    success = False
    
    for source in sources:
        print(f"\nTrying source: {source}")
        cap = None
        
        try:
            if isinstance(source, tuple):
                # Source with explicit backend
                index, backend = source
                if backend is not None:
                    print(f"Opening camera index {index} with specific backend")
                    cap = cv2.VideoCapture(index, backend)
                else:
                    print(f"Opening camera index {index} with default backend")
                    cap = cv2.VideoCapture(index)
            else:
                # String-based source or simple index
                print(f"Opening camera source: {source}")
                cap = cv2.VideoCapture(source)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                print(f"Failed to open {source}")
                if cap:
                    cap.release()
                continue
            
            print(f"Camera opened successfully with source: {source}")
            
            # Try to read a frame to verify
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Camera opened but failed to capture frame")
                cap.release()
                continue
            
            print(f"Successfully captured frame with dimensions: {frame.shape}")
            
            # Camera is working - show video feed for 5 seconds
            success = True
            start_time = time.time()
            frame_count = 0
            
            print("Capturing video for 5 seconds... Press 'q' to quit early.")
            
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    print("Frame capture failed")
                    break
                
                frame_count += 1
                
                # In headless mode, we don't display frames
                # Just report frame properties
                if frame_count % 10 == 0:  # Only report every 10 frames to reduce output
                    print(f"Frame {frame_count}: shape={frame.shape}, type={frame.dtype}")
                
                # Simulate delay without GUI
                time.sleep(0.03)  # ~30 FPS
                
                # Check if we should exit
                if frame_count > 100:  # Capture 100 frames max
                    print("Reached maximum frame count")
                    break
            
            print(f"Captured {frame_count} frames in {time.time() - start_time:.2f} seconds")
            print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
            
            # Success! Break out of the loop
            break
            
        except Exception as e:
            print(f"Error with camera source {source}: {e}")
            if cap:
                cap.release()
            continue
        
        finally:
            # Always release the camera
            if cap:
                cap.release()
    
    # Clean up - no windows to destroy in headless mode
    print("Test completed, no cleanup needed in headless mode")
    
    if success:
        print("\nWebcam test SUCCESSFUL! Found working camera configuration.")
        print(f"Working camera source: {source}")
        return True
    else:
        print("\nWebcam test FAILED! Could not access any camera.")
        print("Try checking camera permissions and connections.")
        return False

if __name__ == "__main__":
    try:
        result = test_webcam()
        if result:
            print("\nExit status: SUCCESS")
            sys.exit(0)
        else:
            print("\nExit status: FAILURE")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(3)
