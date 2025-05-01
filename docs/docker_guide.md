# Docker Configuration Guide

This guide explains how to use Docker with the Facial Recognition Software Project.

## Quick Reference

```bash
# Navigate to docker directory
cd docker

# Start in interactive mode (default)
docker-compose up

# Start in detached mode (background)
docker-compose up -d

# View logs when running in production mode
docker-compose logs -f

# Stop the container
docker-compose down

# Rebuild after Dockerfile changes
docker-compose build --no-cache
```

## Project Organization

All Docker-related files are organized in the `docker/` directory:

- `docker/Dockerfile`: Container definition
- `docker/docker-compose.yml`: Service configuration
- `docker/entrypoint.sh`: Container startup script
- `docker/init_demo_data.py`: Demo data initialization 
- `docker/.env`: Environment variable defaults

To use Docker, simply navigate to this directory and use standard Docker Compose commands.

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

2. **In docker/docker-compose.yml**:
   ```yaml
   environment:
     - DEMO_MODE=true
     - UI_PORT=8502
   ```

3. **docker/.env File**: Edit the file in the docker directory:
   ```
   DEMO_MODE=true
   UI_PORT=8502
   ```

## Working with Data

Data persists between container restarts through volume mounts. These mounts create a direct link between directories on your host machine and inside the Docker container:

- **../data** → **/app/data**: Datasets, known faces, test images, results
- **../logs** → **/app/logs**: Application logs
- **../src** → **/app/src**: Source code for the application
- **../config** → **/app/config**: Configuration files
- **../docs** → **/app/docs**: Documentation
- **../.streamlit** → **/app/.streamlit**: Streamlit configuration

### Dataset Directory Structure

The following directory structure is used for datasets:

```
/app/data/
├── datasets/
│   └── utkface/
│       ├── utkface_aligned/    # Raw aligned face images
│       └── demographic_split/   # Images organized by demographic groups
├── known_faces/                 # Reference faces for matching
├── results/                     # Output files
├── test_datasets/
│   └── demographic_split_set/
│       ├── white/               # Images for bias testing
│       ├── black/
│       ├── asian/
│       ├── indian/
│       └── others/
└── test_images/
    ├── known/                   # Test images of known people
    └── unknown/                 # Test images of unknown people
```

### Sample Data

The container automatically initializes the directory structure on startup if `DEMO_MODE=true` (default). To use with the UTKFace dataset:

1. Download the UTKFace dataset manually following the instructions in the main README
2. Place the dataset files in the appropriate local directories on your host machine:
   - Raw aligned images: `./data/datasets/utkface/utkface_aligned/`
   - Demographic split images: `./data/datasets/utkface/demographic_split/`
3. The container will automatically access these files through the volume mounts

### Data Permissions

Ensure that your data directories have appropriate permissions:

```bash
# On Unix/Linux/macOS systems, ensure read/write permissions
chmod -R 755 ./data
chmod -R 755 ./logs
```

## Interactive Development

The Docker setup is optimized for demonstration with volume mounts that update in real-time:

1. **Code Changes**: Edit files in `./src` and see changes immediately 
2. **Configuration**: Modify files in `./config` to adjust application settings
3. **Documentation**: Update files in `./docs` to improve project documentation
4. **Testing**: The main scripts (`run_demo.py` and `run_tests.py`) are mounted for easy testing

When you edit mapped files, the changes will be reflected in the container without rebuilding. This is especially useful for portfolio demonstration and educational purposes.

## Using the Docker Container

### Running the Streamlit Interface

By default, the container starts the Streamlit interface. You can access it at http://localhost:8501 (or at the port you specified with UI_PORT).

### Shell Access to Container

For debugging or advanced usage, you can access a shell inside the container:

```bash
docker exec -it facerec-app /bin/bash
```

### Data Directory Structure

The Docker container expects the following data directory structure on your host machine:

```
./data/
├── datasets/
│   └── utkface/
│       ├── utkface_aligned/     # Raw aligned face images
│       └── demographic_split/   # Images organized by demographic groups
├── known_faces/                 # Reference faces for matching
├── results/                     # Output files
├── test_datasets/
│   └── demographic_split_set/
│       ├── white/               # Images for bias testing
│       ├── black/
│       ├── asian/
│       ├── indian/
│       └── others/
└── test_images/
    ├── known/                   # Test images of known people
    └── unknown/                 # Test images of unknown people
```

The Docker container automatically creates these directories on startup (when DEMO_MODE=true).

#### Adding the UTKFace Dataset

To use the UTKFace dataset with the Docker container:

1. Download the UTKFace dataset following the instructions in the main README.md or docs/quick_guides/dataset_setup.md

2. The directories for the dataset are already created in the container:
   ```
   ./data/datasets/utkface/utkface_aligned/
   ./data/datasets/utkface/demographic_split/
   ```

3. Extract the downloaded dataset to the appropriate host directories

4. The mounted volumes will make the data available inside the container at:
   ```
   /app/data/datasets/utkface/utkface_aligned/
   /app/data/datasets/utkface/demographic_split/
   ```

5. Use the Streamlit interface's Dataset Management feature to prepare the data for bias testing

## Troubleshooting

Common Docker-related issues and their solutions:

1. **Port Already In Use**:
   - Error: `Error starting userland proxy: port is already allocated`
   - Solution: Change the UI_PORT environment variable
     ```bash
     UI_PORT=8502 docker-compose up
     ```

2. **Permission Denied on Volumes**:
   - Error: `permission denied` when accessing volumes
   - Solution: Check file permissions on host directories
     ```bash
     # Set appropriate permissions for data directories
     chmod -R 755 ../data
     chmod -R 755 ../logs
     
     # If using Linux, you may need to fix ownership
     sudo chown -R $(id -u):$(id -g) ../data ../logs
     ```

3. **Container Fails Health Check**:
   - Error: `Container ... is unhealthy`
   - Solution: 
     1. Check logs with `docker-compose logs -f`
     2. Verify Streamlit is running correctly by accessing http://localhost:8501 in your browser
     3. If Streamlit fails to start, check the container's internal logs:
        ```bash
        docker exec -it facerec-app cat /app/logs/error.log
        ```

4. **Missing Dataset Directory**:
   - Error: Messages about missing datasets
   - Solution: 
     1. Ensure the initial directory structure is created:
        ```bash
        # Stop container if running
        docker-compose down
        
        # Set DEMO_MODE to true
        export DEMO_MODE=true
        
        # Restart the container
        docker-compose up -d
        ```
     2. Follow README instructions to download datasets to the correct paths
     3. Check that the volume mounts are correct in docker-compose.yml

5. **Docker Commands Not Working**:
   - Error: `Could not find a docker-compose.yml file in the current directory`
   - Solution: Make sure you're in the docker directory before running commands
     ```bash
     cd /path/to/Facial-Recognition-Software-Project/docker
     docker-compose up
     ```

6. **Dataset Not Visible in Container**:
   - Error: Application shows "No dataset found" even though files exist on host
   - Solution:
     1. Verify the volume mount is correct in docker-compose.yml
     2. Check if the file paths mentioned in the application match Docker container paths
     3. Restart the container to refresh volume mounts:
        ```bash
        docker-compose down && docker-compose up -d
        ```

7. **Container Crashes on Startup**:
   - Error: Container exits immediately after starting
   - Solution:
     1. Check startup logs: `docker-compose logs`
     2. Make sure all required directories exist on the host
     3. Ensure the entrypoint script has execute permissions:
        ```bash
        chmod +x ../docker/entrypoint.sh
        ```
     4. Try rebuilding the container: `docker-compose build --no-cache`

## Performance Considerations

- The Docker container runs in headless mode by default
- Webcam features are disabled by default since containers typically don't have camera access
- For resource-intensive operations like demographic bias testing, ensure your host has adequate CPU and memory
