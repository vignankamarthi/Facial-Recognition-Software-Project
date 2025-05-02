#!/bin/bash
# This file has been moved to the webcam/ directory
# Please use the new location: docker/webcam/webcam_setup.sh

echo "WARNING: This file has been moved to docker/webcam/webcam_setup.sh"
echo "Please update your references to use the new location."
echo ""
echo "Redirecting to new location..."

NEW_SCRIPT="$(dirname "$0")/webcam/webcam_setup.sh"

if [ -f "$NEW_SCRIPT" ]; then
    echo "Executing: $NEW_SCRIPT"
    bash "$NEW_SCRIPT" "$@"
    exit $?
else
    echo "ERROR: New script location not found: $NEW_SCRIPT"
    echo "The webcam directory may have been moved or deleted."
    exit 1
fi
