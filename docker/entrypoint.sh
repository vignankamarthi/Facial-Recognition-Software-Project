#!/bin/bash
set -e

# Function to check if a directory is writable
check_writable() {
    local dir=$1
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory $dir is not writable. This may cause problems."
        echo "Try running the container with appropriate volume permissions."
        return 1
    fi
    return 0
}

# Create and check essential directories
echo "Checking essential directories..."

# Create and setup Streamlit home directory
mkdir -p /app/src/ui/streamlit/.streamlit
chmod -R 777 /app/src/ui/streamlit/.streamlit

# Create and setup directories with proper permissions
mkdir -p /app/logs /app/data /app/data/known_faces /app/data/results /app/data/test_datasets
chmod -R 777 /app/logs /app/data

# Initialize log files if they don't exist
touch /app/logs/debug.log /app/logs/info.log /app/logs/error.log
chmod 666 /app/logs/*.log

# Fix potential permissions in data subdirectories
find /app/data -type d -exec chmod 777 {} \;
find /app/data -type f -exec chmod 666 {} \;

# Check data directory is writable
if [ -d "/app/data" ]; then
    check_writable "/app/data" || true
fi

# Setup for webcam access
if [ "$WEBCAM_ENABLED" = "true" ]; then
    echo "Setting up webcam access..."
    
    # Create video device nodes if they don't exist (mainly for macOS compatibility)
    if [ ! -e "/dev/video0" ] && [ -e "/dev" ]; then
        echo "Creating virtual video device nodes..."
        # Try to create a symlink to the actual camera device if available
        for cam_device in /dev/video* /dev/avfoundation* /dev/dshow*; do
            if [ -e "$cam_device" ]; then
                ln -sf "$cam_device" /dev/video0 2>/dev/null || true
                echo "Linked $cam_device to /dev/video0"
                break
            fi
        done
    fi
    
    # Set permissions for any video devices
    for dev in /dev/video*; do
        if [ -e "$dev" ]; then
            echo "Setting permissions for $dev"
            chmod 777 "$dev" 2>/dev/null || true
        fi
    done
    
    # Create X11 socket directory for GUI windows if needed
    mkdir -p /tmp/.X11-unix
    chmod 1777 /tmp/.X11-unix
    
    echo "Webcam setup complete."
fi

# Set environment variable to avoid Streamlit trying to create files in /home/facerec
export HOME=/app

# Check if we're in demo mode
if [ "$DEMO_MODE" = "true" ]; then
    echo "Initializing demo data..."
    python /app/init_demo_data.py || {
        echo "ERROR: Failed to initialize demo data."
        echo "Check permissions and available disk space."
    }
    
    # Verify key directories exist after initialization
    echo "Verifying data directories..."
    key_dirs=(
        "/app/data/datasets/utkface/utkface_aligned"
        "/app/data/datasets/utkface/demographic_split"
        "/app/data/known_faces"
        "/app/data/test_datasets/demographic_split_set"
    )
    
    for dir in "${key_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "WARNING: Directory $dir was not created. Some features may not work correctly."
        fi
    done
fi

# Display startup message
echo "==================================================="
echo "Facial Recognition Software Project - Local Demo Edition"
echo "==================================================="
echo "Environment setup complete."
echo "Streamlit UI port: $UI_PORT"
echo "Headless mode: $FORCE_HEADLESS"
echo "Log level: $LOG_LEVEL"
echo "Note: This application is configured for LOCAL USE ONLY"
echo "==================================================="

# Execute passed command
exec "$@"
