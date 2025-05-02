#!/bin/bash
# Facial Recognition Docker Static Camera Setup Script
# Configures and validates static photo capture for facial recognition in Docker

# Make test script executable
chmod +x ./webcam/test_webcam.py 

# Reset container state for clean setup
echo "Stopping any existing containers..."
docker-compose down

# Build optimized container for facial recognition
echo "Building facial recognition container with static photo capture support..."
docker-compose build --no-cache

# Start container with Streamlit camera configuration
echo "Starting container with browser-based static photo capture..."
docker-compose up -d

# Allow time for facial recognition systems to initialize
echo "Initializing facial recognition modules (static photo mode only)..."
sleep 5

# Validate basic camera access (not live streaming)
echo "Testing camera accessibility for static photo capture..."
docker-compose exec facerec python /app/docker/webcam/test_webcam.py

# Display container status with resource allocation
docker-compose ps

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║ Facial Recognition Docker Photo Capture Setup Complete      ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""
echo "The application is running at: http://localhost:8501"
echo ""
echo "To use facial recognition features with static photo capture:"
echo ""
echo "1. Open http://localhost:8501 in your browser"
echo "2. Navigate to one of the facial recognition features (static photo only):"
echo "   - Face Detection: Locate and identify faces in photos"
echo "   - Face Matching: Compare photos against known references"
echo "   - Face Anonymization: Apply privacy filters to photos"
echo "3. Select the 'Use Webcam' tab in the feature"
echo "4. Allow camera access when prompted"
echo "5. Position your face for optimal photo capture:"
echo "   - Center in the frame"
echo "   - Well-lit from the front"
echo "   - Limited background complexity"
echo "6. Click the 'Take Photo' button to capture and process a single frame"
echo "7. Note that live webcam features are NOT available in Docker"
echo "8. You must click 'Take Photo' for each new frame you want to process"
echo ""
echo "Troubleshooting photo capture issues:"
echo "1. Check camera detection: docker-compose exec facerec python /app/docker/webcam/test_webcam.py"
echo "2. Verify browser permissions: chrome://settings/content/camera"
echo "3. View recognition logs: docker-compose logs | grep 'detector'"
echo "4. See detailed documentation: docker/webcam/DOCKER_WEBCAM.md"
echo "5. For live webcam features: exit Docker and run 'python run_demo.py' directly on your host"
