# Base Python devcontainer image aligned with project version
FROM mcr.microsoft.com/devcontainers/python:3.12

# Upgrade pip up front to ensure recent wheels
RUN python -m pip install --upgrade pip

# Copy requirements to leverage Docker layer caching when dependencies change
COPY ../requirements.txt /tmp/pip-tmp/requirements.txt

# Install project dependencies inside the container; torch wheel may require CUDA support on host
RUN python -m pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    || echo "WARNING: Review PyTorch installation instructions if the pinned wheel is unavailable for this environment."

# Clean up temporary files so the image stays as small as practical
RUN rm -rf /tmp/pip-tmp
