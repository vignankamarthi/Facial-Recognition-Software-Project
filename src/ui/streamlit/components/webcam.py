"""
Webcam Component for Streamlit Interface

This module provides a reusable webcam component that can be used across different pages
of the Streamlit interface for capturing and processing video frames.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
import sys
from PIL import Image
from typing import Tuple, Dict, Callable, Any, List, Optional


def webcam_component(
    callback_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    use_column: bool = True,
    fps_limit: int = 24,
    show_fps: bool = True,
    key_prefix: str = "",
) -> None:
    """
    Reusable webcam component that launches an external OpenCV window for real-time processing.

    Parameters
    ----------
    callback_func : Callable
        Function to process each frame, should return (processed_frame, metadata_dict)
    use_column : bool, optional
        Whether to use columns for layout (default: True)
    fps_limit : int, optional
        Maximum FPS to process (default: 24)
    show_fps : bool, optional
        Whether to show FPS counter (default: True)
    key_prefix : str, optional
        Prefix for session state keys to avoid conflicts (default: "")
    """
    # Import necessary utilities for external window
    try:
        # Import from src.utils with error handling
        from src.utils.common_utils import (
            safely_close_windows,
            handle_opencv_error,
            CameraError,
            format_error,
            create_resizable_window,
        )
        from src.utils.config import get_config

        config = get_config()
        WINDOW_NAME = config.ui.window_name
        WAIT_KEY_DELAY = config.ui.wait_key_delay
    except ImportError as e:
        # Fallback constants if imports fail
        WINDOW_NAME = "Facial Recognition Demo"
        WAIT_KEY_DELAY = 100

        # Minimal implementations
        def safely_close_windows(window_name=None, video_capture=None):
            if video_capture is not None and video_capture.isOpened():
                video_capture.release()
            cv2.destroyAllWindows()

        def create_resizable_window(window_name):
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            return window_name

        class CameraError(Exception):
            """Fallback exception for camera errors."""

            pass

        def format_error(error_type, message):
            return f"ERROR: {message}"

    # Set up session state for webcam
    if f"{key_prefix}webcam_running" not in st.session_state:
        st.session_state[f"{key_prefix}webcam_running"] = False
        st.session_state[f"{key_prefix}external_window_active"] = False
        st.session_state[f"{key_prefix}frame_count"] = 0
        st.session_state[f"{key_prefix}start_time"] = time.time()
        st.session_state[f"{key_prefix}camera_error"] = None

    # Create info placeholders for feedback
    if use_column:
        col_info1, col_info2 = st.columns([1, 1])
        status_placeholder = col_info1.empty()
        info_placeholder = col_info2.empty()
    else:
        status_placeholder = st.empty()
        info_placeholder = st.empty()

    # Control buttons
    col1, col2 = st.columns([1, 1])

    with col1:
        if not st.session_state[f"{key_prefix}webcam_running"]:
            if st.button(
                "Launch Webcam Window", key=f"{key_prefix}start_webcam", type="primary"
            ):
                st.session_state[f"{key_prefix}webcam_running"] = True
                st.session_state[f"{key_prefix}external_window_active"] = True
                st.session_state[f"{key_prefix}frame_count"] = 0
                st.session_state[f"{key_prefix}start_time"] = time.time()

                # Launch external window in a separate thread with Streamlit context
                try:
                    # Store the current streamlit context to allow cross-thread updates
                    from streamlit.runtime.scriptrunner import get_script_run_ctx, add_script_run_ctx
                    
                    # Get current context
                    ctx = get_script_run_ctx()
                    
                    # Create a wrapper function that restores the context
                    def thread_with_context():
                        try:
                            # Restore the Streamlit context in this thread
                            add_script_run_ctx(ctx)
                            # Now run the actual function
                            launch_external_window(callback_func, info_placeholder, key_prefix, show_fps)
                        except Exception as e:
                            print(f"Error in thread: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Start thread with context
                    import threading
                    thread = threading.Thread(target=thread_with_context)
                    thread.daemon = True
                    thread.start()
                except Exception as e:
                    # If threading with context fails, fallback to direct launch
                    print(f"Could not create thread with Streamlit context: {e}. Launching directly.")
                    # If the threading setup failed, run directly
                    import threading
                    thread = threading.Thread(
                        target=launch_external_window,
                        args=(callback_func, info_placeholder, key_prefix, show_fps),
                    )
                    thread.daemon = True
                    thread.start()

                # Update UI
                status_placeholder.success("External webcam window launched!")
        else:
            if st.button(
                "Close Webcam Window", key=f"{key_prefix}stop_webcam", type="secondary"
            ):
                st.session_state[f"{key_prefix}webcam_running"] = False
                st.session_state[f"{key_prefix}external_window_active"] = False
                # Force update
                st.rerun()

    # Provide instructions for external window
    if not st.session_state[f"{key_prefix}webcam_running"]:
        # Show any previous camera errors if they exist
        if (
            f"{key_prefix}camera_error" in st.session_state
            and st.session_state[f"{key_prefix}camera_error"]
        ):
            status_placeholder.error(st.session_state[f"{key_prefix}camera_error"])

            # Add OS-specific troubleshooting tips
            if sys.platform == "darwin":  # macOS
                info_placeholder.warning(
                    "**macOS Camera Tips:** Check System Preferences > Security & Privacy > Camera permissions"
                )
            elif sys.platform == "win32":  # Windows
                info_placeholder.warning(
                    "**Windows Camera Tips:** Check Settings > Privacy > Camera access settings"
                )
            elif sys.platform.startswith("linux"):  # Linux
                info_placeholder.warning(
                    "**Linux Camera Tips:** You may need to add your user to the video group"
                )
        else:
            info_placeholder.info(
                """
            Click "Launch Webcam Window" to open a separate window with real-time webcam feed.\n\n
            **Controls in external window:**\n
            - Press 'q' to close the window\n
            - The window can be resized by dragging its corners
            """
            )
    else:
        info_placeholder.info(
            """
        **External window is active.**\n\n
        - The webcam processing is happening in a separate window.\n
        - If you can't see it, check if it's behind other windows or minimized.\n
        - Press 'q' in that window to close it, or click 'Close Webcam Window' here.
        """
        )

        # Check if button to close all OpenCV windows should be shown
        if st.session_state[f"{key_prefix}external_window_active"]:
            with col2:
                if st.button(
                    "Force Close All Windows",
                    key=f"{key_prefix}force_close",
                    type="secondary",
                ):
                    # Force destroy all OpenCV windows
                    cv2.destroyAllWindows()
                    st.session_state[f"{key_prefix}webcam_running"] = False
                    st.session_state[f"{key_prefix}external_window_active"] = False
                    status_placeholder.warning("Forced all OpenCV windows to close!")
                    # Force update
                    st.rerun()


# Function to launch external OpenCV window
def launch_external_window(
    callback_func, info_placeholder, key_prefix="", show_fps=True
):
    """Launch external OpenCV window with webcam feed."""
    try:
        # Import necessary utilities
        try:
            from src.utils.common_utils import (
                safely_close_windows,
                create_resizable_window,
            )
            from src.utils.config import get_config

            config = get_config()
            WINDOW_NAME = config.ui.window_name
            WAIT_KEY_DELAY = config.ui.wait_key_delay
        except ImportError:
            WINDOW_NAME = "Facial Recognition Demo"
            WAIT_KEY_DELAY = 100

            def safely_close_windows(window_name=None, video_capture=None):
                if video_capture is not None and video_capture.isOpened():
                    video_capture.release()
                cv2.destroyAllWindows()

            def create_resizable_window(window_name):
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                return window_name

        # Initialize webcam with multi-platform support
        cap = None

        # Try multiple camera sources with OS-specific optimizations
        camera_sources = []
        
        # For MacOS, we use a specialized approach with AVFoundation
        if sys.platform == "darwin":
            # These sources work best for macOS
            camera_sources = [
                0,  # Default camera - usually works with AVFoundation
                "avfoundation://0",  # Primary camera explicit
                "avfoundation://1",  # Secondary camera explicit
                1,  # Secondary camera index
                "avfoundation://0:0",  # Alternative syntax with audio channel
                -1,  # Any available camera
            ]
            print("macOS detected, using AVFoundation sources:")
            print(camera_sources)
        elif sys.platform == "win32":
            # Windows-specific sources
            camera_sources = [
                0,  # Default camera
                1,  # Secondary camera
                "dshow://0",  # DirectShow primary
                "msmf://0",  # Media Foundation primary
                "dshow://video=0",  # Alternative DirectShow syntax
                -1,  # Any available camera
            ]
        elif sys.platform.startswith("linux"):
            # Linux-specific sources
            camera_sources = [
                0,  # Default camera
                1,  # Secondary camera
                "v4l2:///dev/video0",  # Video4Linux primary
                "v4l2:///dev/video1",  # Video4Linux secondary
                -1,  # Any available camera
            ]
        else:
            # Generic fallback for other platforms
            camera_sources = [
                0,  # Default camera
                1,  # Secondary camera (external/USB cameras on many systems)
                -1,  # Any available camera (works on some systems)
                "camera:0",
                "camera:1",  # Alternative camera syntax
                "camera",  # Generic camera API
            ]

        # Set OpenCV backend preferences for better compatibility
        # This helps with macOS and Linux compatibility
        cv2.setUseOptimized(True)

        # Set common environment variables for better camera access
        os.environ["OPENCV_VIDEOIO_DEBUG"] = (
            "1"  # Enable debug output for camera issues
        )

        # On macOS, prioritize AVFoundation backend and optimize settings
        if sys.platform == "darwin":
            # Set environment variables for macOS camera access
            os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
            os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "1000"
            os.environ["OPENCV_VIDEOIO_PRIORITY_QT"] = "0"
            os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
            os.environ["OPENCV_VIDEOIO_PRIORITY_V4L"] = "0"
            os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "0"
            os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
            
            print("Applied macOS-specific OpenCV backend settings")
            
            # Create directory for macOS camera access
            os.makedirs("/tmp/webcam", exist_ok=True)
            
            # Try to reset QuickTime/AVFoundation
            try:
                print("Attempting to reset VDCAssistant (camera service)...")
                os.system("killall VDCAssistant 2>/dev/null")
                time.sleep(1)  # Give time for the service to restart
            except Exception as e:
                print(f"Error resetting VDCAssistant: {e}")

        # On Windows, prioritize DirectShow/MediaFoundation
        elif sys.platform == "win32":
            os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"
            os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "900"
            camera_sources = [0, 1, "dshow://0", "msmf://0"] + camera_sources

        # On Linux, prioritize V4L2
        elif sys.platform.startswith("linux"):
            os.environ["OPENCV_VIDEOIO_PRIORITY_V4L2"] = "1000"
            os.environ["OPENCV_VIDEOIO_PRIORITY_V4L"] = "900"
            camera_sources = [0, 1, "v4l2:///dev/video0"] + camera_sources

            # Try setting video device permissions
            try:
                os.system("chmod 777 /dev/video* 2>/dev/null")
            except:
                pass

        # Try each source until one works
        success = False
        error_messages = []

        for idx, source in enumerate(camera_sources):
            try:
                # Try with default parameters first
                print(f"Trying camera source: {source}")
                
                # Apply specific settings for different source types
                if sys.platform == "darwin" and isinstance(source, int):
                    # On macOS with numeric index, explicitly use AVFoundation backend
                    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
                    print(f"Using explicit AVFoundation backend for source {source}")
                elif sys.platform == "darwin" and isinstance(source, str) and "avfoundation" in source:
                    # For AVFoundation URL format
                    cap = cv2.VideoCapture(source)
                    print(f"Using AVFoundation URL: {source}")
                else:
                    # Default approach for other platforms/sources
                    cap = cv2.VideoCapture(source)

                # Check if opened successfully
                if cap and cap.isOpened():
                    # Try multiple reads to make sure the camera is stable
                    for _ in range(3):
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            success = True
                            print(f"Successfully opened camera with source: {source}")
                            # Configure camera properties for better performance
                            cap.set(
                                cv2.CAP_PROP_BUFFERSIZE, 1
                            )  # Reduce buffer size for fresher frames

                            # Try setting common resolution (640x480) which works on most webcams
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                            # Set lower frame rate for more stable processing
                            cap.set(cv2.CAP_PROP_FPS, 15)
                            break

                    if success:
                        break
                else:
                    # Try to release and retry with explicit API preferences
                    if cap:
                        cap.release()

                    # On macOS, try with AVFoundation backend explicitly
                    if sys.platform == "darwin":
                        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
                    # On Windows, try with DirectShow backend explicitly
                    elif sys.platform == "win32":
                        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                    # On Linux, try with V4L2 backend explicitly
                    elif sys.platform.startswith("linux"):
                        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
                    else:
                        cap = cv2.VideoCapture(source)

                    # Check again with backend selection
                    if cap and cap.isOpened():
                        # Try multiple reads to make sure the camera is stable
                        for _ in range(3):
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                success = True
                                print(
                                    f"Successfully opened camera with source: {source} (backend specified)"
                                )
                                # Configure camera properties for better performance
                                cap.set(
                                    cv2.CAP_PROP_BUFFERSIZE, 1
                                )  # Reduce buffer size for fresher frames

                                # Try setting common resolution (640x480) which works on most webcams
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                                # Set lower frame rate for more stable processing
                                cap.set(cv2.CAP_PROP_FPS, 15)
                                break

                        if success:
                            break
                    else:
                        if cap:
                            cap.release()
                        error_messages.append(f"Could not open camera source: {source}")
            except Exception as e:
                error_messages.append(f"Error with camera source {source}: {str(e)}")
                if cap:
                    cap.release()
                cap = None

        if not success:
            # All camera sources failed
            error_str = "\n".join(error_messages)
            print(f"Failed to open any camera source. Errors: {error_str}")
            st.session_state[f"{key_prefix}camera_error"] = "\n".join(
                [
                    "Camera detection failed. Try these steps:",
                    "1. Close other applications using your camera",
                    "2. Check camera permissions in your OS settings",
                    "3. Try disconnecting and reconnecting your webcam",
                    "4. Restart your computer if problems persist",
                ]
            )
            # Don't update UI from thread - could cause NoSessionContext error
            # Main thread will read error from session state
            print("Camera error stored in session state - main thread will display it")
            return

        # Create a resizable window
        create_resizable_window(WINDOW_NAME)

        # Start processing loop
        start_time = time.time()
        frame_count = 0

        # Process frames until window is closed or 'q' is pressed
        frame_retry_count = 0
        max_retries = 5
        last_frame_time = time.time()
        frame_interval = 1.0 / 30.0  # Cap at 30 FPS to avoid overloading

        while st.session_state[f"{key_prefix}webcam_running"] and cap.isOpened():
            # Control frame rate to avoid excessive CPU usage
            current_time = time.time()
            time_since_last_frame = current_time - last_frame_time

            if time_since_last_frame < frame_interval:
                # Sleep to maintain frame rate limit
                time.sleep(frame_interval - time_since_last_frame)

            # Capture frame
            ret, frame = cap.read()
            last_frame_time = time.time()

            if not ret or frame is None or frame.size == 0:
                frame_retry_count += 1
                print(
                    f"Failed to capture frame from webcam (attempt {frame_retry_count}/{max_retries})"
                )

                if frame_retry_count >= max_retries:
                    print("Max retries reached for frame capture. Exiting.")
                    break

                # Try to recover
                time.sleep(0.5)  # Wait before retry
                continue  # Skip to next iteration

            # Update timing info
            frame_count += 1
            elapsed_time = time.time() - start_time

            # Calculate FPS
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Process frame with callback function
            try:
                processed_frame, metadata = callback_func(frame)

                # Add FPS counter if enabled
                if show_fps:
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(
                        processed_frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                # Display the processed frame in external window
                cv2.imshow(WINDOW_NAME, processed_frame)

            except Exception as e:
                print(f"Error processing frame: {e}")
                # If frame processing fails, still try to show original frame
                cv2.imshow(WINDOW_NAME, frame)

            # Check for key press with wait delay
            key = cv2.waitKey(WAIT_KEY_DELAY) & 0xFF

            # Handle key presses - 'q', 'Q', or ESC to quit
            if key in [ord("q"), ord("Q"), 27]:
                print("User pressed quit key. Closing webcam window...")
                break

            # Check if we should still be running (button may have changed state)
            if not st.session_state[f"{key_prefix}webcam_running"]:
                print("External flag set to stop webcam. Closing window...")
                break

    except Exception as e:
        # Log any errors
        print(f"Error in external webcam window: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up resources
        print("Cleaning up webcam resources...")

        try:
            # Try to release the camera capture
            if cap is not None:
                if cap.isOpened():
                    # On some platforms, reading one last frame can help with clean release
                    try:
                        cap.read()
                    except:
                        pass
                    cap.release()
                else:
                    # For already closed captures, try explicit deletion
                    del cap
        except Exception as e:
            print(f"Error releasing camera: {e}")

        # Close all OpenCV windows with multiple attempts to ensure closure
        try:
            # First destroy attempt
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Wait for windows to actually close (OpenCV quirk)

            # Second attempt, sometimes needed on Windows
            time.sleep(0.1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            # Final attempt - try to destroy any window with our specific name
            try:
                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)
            except:
                pass
        except Exception as e:
            print(f"Error closing OpenCV windows: {e}")

        # Update session state to reflect window closure
        st.session_state[f"{key_prefix}webcam_running"] = False
        st.session_state[f"{key_prefix}external_window_active"] = False

        print("External webcam window closed and resources released.")


def image_upload_component(
    callback_func: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    allowed_types: List[str] = ["jpg", "jpeg", "png"],
    key_prefix: str = "",
) -> None:
    """
    Reusable image upload component that handles image upload and processing.

    Parameters
    ----------
    callback_func : Callable
        Function to process the image, should return (processed_image, metadata_dict)
    allowed_types : List[str], optional
        List of allowed file extensions (default: ["jpg", "jpeg", "png"])
    key_prefix : str, optional
        Prefix for session state keys to avoid conflicts (default: "")
    """
    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose an image...", type=allowed_types, key=f"{key_prefix}image_uploader"
    )

    if uploaded_file is not None:
        # Create columns for display
        col_img, col_info = st.columns([3, 1])

        try:
            # Read and process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV format) if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Process image with callback function
            processed_image, metadata = callback_func(image_np)

            # Convert back to RGB for display
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3:
                display_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            else:
                display_img = processed_image

            # Display processed image
            col_img.image(
                display_img,
                caption=f"Processed image: {uploaded_file.name}",
                use_column_width=True,
            )

            # Display metadata
            if metadata:
                info_text = ""
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        info_text += f"**{key}**: {value}\n\n"
                    elif (
                        isinstance(value, list)
                        and key == "face_locations"
                        and len(value) > 0
                    ):
                        info_text += f"**{key}**: {len(value)} faces found\n\n"

                        # Show first few faces' locations
                        max_faces_to_show = min(5, len(value))
                        for i in range(max_faces_to_show):
                            face = value[i]
                            if isinstance(face, tuple) and len(face) == 4:
                                top, right, bottom, left = face
                                info_text += f"Face {i+1}: (T={top}, R={right}, B={bottom}, L={left})\n\n"

                if info_text:
                    col_info.markdown(info_text)

            # Add download button for processed image
            if st.button("Save Processed Image", key=f"{key_prefix}save_button"):
                # Save image to temp file
                filename = f"processed_{uploaded_file.name}"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=f".{filename.split('.')[-1]}"
                ) as tmp:
                    if (
                        len(processed_image.shape) == 3
                        and processed_image.shape[2] == 3
                    ):
                        save_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    else:
                        save_img = processed_image

                    pil_img = Image.fromarray(save_img)
                    pil_img.save(tmp.name)

                # Offer download
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="Download Processed Image",
                        data=f.read(),
                        file_name=filename,
                        mime=f"image/{filename.split('.')[-1]}",
                    )

                # Clean up temp file
                os.unlink(tmp.name)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
