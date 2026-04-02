# Face Recognition API Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/splits \
    uploads/temp uploads/faces \
    logs \
    trained_models/checkpoints trained_models/finetuned trained_models/logs \
    models/buffalo_l \
    static

# Download InsightFace model on build (optional, can also download at runtime)
# RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')"

# Expose port
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_HOST=0.0.0.0
ENV API_PORT=5001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/api/health')" || exit 1

# Run the application
CMD ["python", "scripts/run_api.py"]
