FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app

# Copy requirements if it exists, otherwise install common dependencies
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir \
            fastapi \
            uvicorn \
            temporalio \
            redis \
            opencv-python-headless \
            mediapipe \
            numpy \
            pyyaml \
            loguru \
            boto3 \
            pillow; \
    fi

# Copy application code
COPY . /app

# Set Python path
ENV PYTHONPATH=/app

# Expose FastAPI server port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["python", "server.py"]
