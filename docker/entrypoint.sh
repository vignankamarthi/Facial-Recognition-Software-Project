#!/bin/bash
set -e

# Initialize log files if they don't exist
mkdir -p /app/logs
touch /app/logs/debug.log
touch /app/logs/info.log
touch /app/logs/error.log
chmod 666 /app/logs/*.log

# Check if we're in demo mode
if [ "$DEMO_MODE" = "true" ]; then
    echo "Initializing demo data..."
    python /app/init_demo_data.py
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
