# Docker Setup for Facial Recognition Software Project

This directory contains Docker configuration for containerizing the Facial Recognition Software Project. The Docker setup provides a consistent environment for running the application across different platforms.

## Directory Structure

```
docker/
├── .env                   # Environment variables for Docker
├── Dockerfile             # Multi-stage build definition
├── docker-compose.yml     # Container orchestration config
├── entrypoint.sh          # Container initialization script
├── init_demo_data.py      # Script to set up demo data
├── README.md              # This file
└── webcam/                # Webcam-related utilities
    ├── DOCKER_WEBCAM.md   # General webcam usage guide
    ├── test_webcam.py     # Script to test webcam access
    ├── WEBCAM_MACOS.md    # macOS-specific webcam guide
    └── webcam_setup.sh    # Script to configure webcam
```

## Quick Start

To run the application in Docker:

```bash
# Build and start the container in the background
docker-compose up -d

# View the logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The Streamlit interface will be available at http://localhost:8501

## Key Features

1. **Multi-stage Build**: Optimized container size with separate build and runtime stages
2. **Volume Mapping**: Real-time code updates without rebuilding
3. **Cross-platform Webcam Support**: Browser-based webcam integration for all platforms
4. **Environment Configuration**: Easily customizable via .env file

## Configuration Options

You can configure the application by editing the `.env` file:

```
# Core settings
UI_PORT=8501                    # Web UI port
WEBCAM_ENABLED=true             # Enable webcam features
FORCE_HEADLESS=true             # Run in headless mode
LOG_LEVEL=INFO                  # Logging verbosity

# Webcam configuration 
WEBCAM_DEVICE_INDEX=0           # Default camera index
USE_STREAMLIT_CAMERA=1          # Use browser-based camera
```

## Volume Mappings

The Docker setup mounts several volumes to enable persistence and development:

- **Source Code**: `../src:/app/src` - Enables live code changes
- **Configuration**: `../config:/app/config` - Application configuration
- **Data**: `../data:/app/data` - Persistent data storage
- **Logs**: `../logs:/app/logs` - Application logs

## Webcam Support

The project includes special handling for webcam access in containerized environments:

- **Browser-based Webcam**: Uses Streamlit's native camera component
- **Platform-specific Optimizations**: Special handling for macOS, Windows, and Linux
- **Diagnostic Tools**: Scripts for testing and troubleshooting webcam access

For detailed webcam setup instructions:
- General webcam usage: [DOCKER_WEBCAM.md](webcam/DOCKER_WEBCAM.md)
- macOS-specific guidance: [WEBCAM_MACOS.md](webcam/WEBCAM_MACOS.md)

## Development Workflow

1. **Start the container**: `docker-compose up -d`
2. **Edit code**: Make changes to files in the `src` directory
3. **View changes**: Refresh the browser at http://localhost:8501
4. **Check logs**: `docker-compose logs -f`
5. **Run tests**: `docker-compose exec facerec python run_tests.py`

## Troubleshooting

### Common Issues

1. **Permission errors**: 
   ```bash
   # Fix permissions on host
   sudo chown -R $(id -u):$(id -g) data logs
   ```

2. **Webcam not working**:
   ```bash
   # Test webcam access
   docker-compose exec facerec python /app/docker/webcam/test_webcam.py
   ```

3. **Container won't start**:
   ```bash
   # Check for errors
   docker-compose logs
   ```

## Advanced Usage

### Rebuilding the Container

After making changes to the Dockerfile:

```bash
docker-compose build --no-cache
docker-compose up -d
```

### Custom Environment Variables

Temporarily override environment variables:

```bash
UI_PORT=8502 docker-compose up -d
```

### Running Without Docker

If you encounter issues with Docker, you can run directly on your host:

```bash
pip install -r requirements.txt
python run_demo.py
```
