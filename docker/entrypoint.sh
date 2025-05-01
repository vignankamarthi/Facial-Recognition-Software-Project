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

# Initialize log files if they don't exist
mkdir -p /app/logs
touch /app/logs/debug.log || echo "WARNING: Cannot create debug.log file"
touch /app/logs/info.log || echo "WARNING: Cannot create info.log file"
touch /app/logs/error.log || echo "WARNING: Cannot create error.log file"
chmod 666 /app/logs/*.log || echo "WARNING: Cannot set permissions on log files"

# Check data directory is writable
if [ -d "/app/data" ]; then
    check_writable "/app/data" || true
fi

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
