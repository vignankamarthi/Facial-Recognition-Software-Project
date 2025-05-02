# ===== BUILD STAGE =====
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Metadata about the image
LABEL maintainer="Facial Recognition Project"
LABEL description="Facial Recognition Software Project - Portfolio Demonstration"
LABEL version="1.0"

# Install system dependencies required for OpenCV and face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Additional dependencies for face_recognition and dlib
    libopenblas-dev \
    liblapack-dev \
    # Cleanup to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===== FINAL STAGE =====
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenblas-dev \
    # Additional packages for webcam access
    libv4l-dev \
    v4l-utils \
    ffmpeg \
    curl \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    libavdevice-dev \
    libavutil-dev \
    # Cleanup to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security and add to video group for webcam access
RUN groupadd -r facerec && useradd -r -g facerec facerec && \
    getent group video || groupadd -r video && \
    usermod -a -G video facerec

# Create directory structure matching the project
RUN mkdir -p /app/config \
    /app/data/datasets/utkface/utkface_aligned \
    /app/data/datasets/utkface/demographic_split \
    /app/data/known_faces \
    /app/data/results \
    /app/data/test_datasets/demographic_split_set/asian \
    /app/data/test_datasets/demographic_split_set/black \
    /app/data/test_datasets/demographic_split_set/indian \
    /app/data/test_datasets/demographic_split_set/others \
    /app/data/test_datasets/demographic_split_set/white \
    /app/data/test_datasets/results \
    /app/data/test_images/known \
    /app/data/test_images/unknown \
    /app/docs/quick_guides \
    /app/logs \
    /app/src/backend \
    /app/src/ui/streamlit/pages \
    /app/src/utils \
    /app/.github/workflows \
    /app/src/ui/streamlit/.streamlit

# Create volume mount points for data persistence
VOLUME ["/app/data", "/app/logs"]

# Copy source code (following the directory structure)
COPY src /app/src
COPY config /app/config
COPY docs /app/docs
COPY .github /app/.github
COPY run_demo.py /app/
COPY run_tests.py /app/
COPY pytest.ini /app/
COPY requirements.txt /app/
COPY README.md /app/
COPY PROJECT_STRUCTURE.md /app/
# Streamlit config is part of src directory that's already copied

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEMO_MODE=true \
    WEBCAM_ENABLED=true \
    LOG_LEVEL=INFO \
    UI_PORT=8501 \
    FORCE_HEADLESS=true \
    STREAMLIT_HOME_DIR=/app/src/ui/streamlit/.streamlit \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=true \
    STREAMLIT_RUNTIME_METRICS_ENABLED=false \
    # OpenCV webcam configuration
    OPENCV_VIDEOIO_DEBUG=1 \
    OPENCV_VIDEOIO_PRIORITY_MSMF=0 \
    OPENCV_VIDEOIO_PRIORITY_QT=0 \
    OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION=1000 \
    OPENCV_FFMPEG_CAPTURE_OPTIONS=video_device_index=0 \
    OPENCV_AVFOUNDATION_SKIP_AUTH=1 \
    # macOS specific webcam settings
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Set permissions
RUN chown -R facerec:facerec /app

# Add Docker-specific files
COPY docker/init_demo_data.py /app/
COPY docker/entrypoint.sh /app/

# Create directory for camera access on macOS
RUN mkdir -p /tmp/.X11-unix && \
    chmod 1777 /tmp/.X11-unix

# Make entry point executable and ensure proper permissions
USER root
RUN chmod +x /app/entrypoint.sh

# Add permissions for webcam access
RUN chmod -R 777 /tmp/.X11-unix
RUN chmod 777 /app

# Stay as root for webcam access in Docker
# USER facerec

# Expose Streamlit port
EXPOSE 8501

# Entry point
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "run_demo.py", "--server.headless=true"]
