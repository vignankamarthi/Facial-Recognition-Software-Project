#!/bin/bash
# Script to run the Facial Recognition Software Project in Docker
# Usage: ./run_docker_demo.sh [--detached]

# Print banner
echo "=================================================="
echo "Facial Recognition Software Project - Docker Demo"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create required directories if they don't exist
mkdir -p data/known_faces
mkdir -p data/test_images
mkdir -p data/datasets
mkdir -p data/results
mkdir -p data/test_datasets/demographic_split_set
mkdir -p logs

# Check if detached mode is requested
if [[ "$1" == "--detached" ]]; then
    echo "Starting in detached mode..."
    echo "- Running as daemon in background"
    echo "- Streamlit UI available at http://localhost:8501"
    
    # Run in detached mode
    docker-compose up -d
    
    # Show logs for initial startup
    echo "Container starting... showing initial logs:"
    docker-compose logs -f --tail=20
    
    echo ""
    echo "Container is running in the background"
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "Starting in interactive mode..."
    echo "- Code changes will be reflected immediately"
    echo "- Data will persist between restarts"
    echo "- Logs will be saved to ./logs directory"
    echo "- Press Ctrl+C to stop"
    
    # Run in interactive mode
    docker-compose up
fi

echo ""
echo "Access the Streamlit UI at: http://localhost:8501"
echo "=================================================="
