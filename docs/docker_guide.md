# Docker Configuration Guide

This guide explains how to use Docker with the Facial Recognition Software Project.

## Quick Reference

```bash
# Start in interactive mode (default)
./run_docker_demo.sh

# Start in detached mode (background)
./run_docker_demo.sh --detached

# View logs when running in production mode
docker-compose logs -f

# Stop the container
docker-compose down

# Rebuild after Dockerfile changes
docker-compose build --no-cache
```

## Docker Setup Benefits

Running the Facial Recognition Software Project in Docker provides several benefits:

1. **Consistent Environment**: Eliminates "it works on my machine" problems
2. **No Local Dependencies**: No need to install Python, OpenCV, or other libraries locally
3. **Easy Deployment**: Share the project with others without complex setup instructions
4. **Development Flexibility**: Edit code in real-time with volume mounts
5. **Resource Isolation**: Container sandboxing prevents system conflicts

## Container Architecture

The Docker setup uses a single-container architecture with:

- **Python 3.9 Base**: Optimized for facial recognition libraries
- **OpenCV and face_recognition**: Pre-installed with all dependencies
- **Multi-stage Build**: Smaller final image size
- **Non-root User**: Enhanced security with minimal permissions
- **Volume Mounts**: Persistent data and real-time code development

## Configuration Options

The following environment variables can be configured:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | `true` | Initialize demo data on startup |
| `WEBCAM_ENABLED` | `false` | Enable/disable webcam features |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `UI_PORT` | `8501` | Streamlit UI port |
| `FORCE_HEADLESS` | `true` | Run in headless mode (no GUI windows) |

### Setting Environment Variables

You can set environment variables in several ways:

1. **Command Line**:
   ```bash
   UI_PORT=8502 docker-compose up
   ```

2. **In docker-compose.yml**:
   ```yaml
   environment:
     - DEMO_MODE=true
     - UI_PORT=8502
   ```

3. **.env File**: Create a file named `.env` in the project root:
   ```
   DEMO_MODE=true
   UI_PORT=8502
   ```

## Working with Data

Data persists between container restarts through volume mounts:

- **./data** → **/app/data**: Datasets, known faces, test images, results
- **./logs** → **/app/logs**: Application logs

### Sample Data

The container automatically initializes directory structure on startup if `DEMO_MODE=true`. To use with the UTKFace dataset:

1. Download the dataset manually following the instructions in the main README
2. Place the dataset files in the appropriate local directories
3. The container will automatically access them through volume mounts

## Interactive Development

The Docker setup is optimized for demonstration with volume mounts that update in real-time:

1. **Code Changes**: Edit files in `./src` and see changes immediately 
2. **Configuration**: Modify files in `./config` to adjust application settings
3. **Documentation**: Update files in `./docs` to improve project documentation

When you edit mapped files, the changes will be reflected in the container without rebuilding. This is especially useful for portfolio demonstration and educational purposes.

## Troubleshooting

Common Docker-related issues:

1. **Port Already In Use**:
   - Error: `Error starting userland proxy: port is already allocated`
   - Solution: Change the UI_PORT environment variable

2. **Permission Denied on Volumes**:
   - Error: `permission denied` when accessing volumes
   - Solution: Check file permissions on host directories

3. **Container Fails Health Check**:
   - Error: `Container ... is unhealthy`
   - Solution: Check logs with `docker-compose logs -f`

4. **Missing Dataset Directory**:
   - Error: Messages about missing datasets
   - Solution: Create the directory structure or follow README instructions to download datasets

## Performance Considerations

- The Docker container runs in headless mode by default
- Webcam features are disabled by default since containers typically don't have camera access
- For resource-intensive operations like demographic bias testing, ensure your host has adequate CPU and memory
