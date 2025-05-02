#!/bin/bash
# Make this script executable with: chmod +x webcam_setup.sh
# Docker webcam setup script for browser-based camera access

# Make sure the script is executable
chmod +x ./test_webcam.py 

# Stop and remove any existing containers
docker-compose down

# Build a fresh container
docker-compose build --no-cache

# Start the container in the background
docker-compose up -d

# Wait for container to initialize
echo "Waiting for container to initialize..."
sleep 5

# Test webcam access directly
echo "Testing webcam access..."
docker-compose exec facerec python /app/docker/test_webcam.py

# Show container status
docker-compose ps

echo ""
echo "Setup complete! Container is running in the background."
echo "Visit http://localhost:8501 to access the application."
echo ""
echo "THE NEW BROWSER-BASED CAMERA APPROACH:"
echo "1. Open http://localhost:8501 in your browser"
echo "2. Navigate to any feature with webcam functionality"
echo "3. Click the 'Use Webcam' tab"
echo "4. Allow camera access when prompted by your browser"
echo "5. Use the 'Take Photo' button to capture frames for processing"
echo ""
echo "If issues persist, try these commands to troubleshoot:"
echo "1. Check container logs: docker-compose logs"
echo "2. View the documentation: open DOCKER_WEBCAM.md"
echo "3. Stop the container: docker-compose down"
