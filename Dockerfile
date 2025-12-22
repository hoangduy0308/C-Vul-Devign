# Devign Vulnerability Scanner - Docker Image
#
# Build:
#   docker build -t devign-scanner:latest .
#
# Run:
#   docker run -v /path/to/code:/code devign-scanner:latest scan /code
#
# For GitHub Actions:
#   Uses this as container image for faster CI runs

FROM python:3.10-slim

LABEL maintainer="hoangduy0308"
LABEL description="AI-powered C/C++ vulnerability scanner using BiGRU"
LABEL version="1.0.0"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (for caching)
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy scanner code
COPY devign_pipeline/devign_infer/ ./devign_infer/
COPY devign_pipeline/devign_scan.py ./
COPY devign_pipeline/src/tokenization/ ./src/tokenization/

# Copy model files (optional - can also mount at runtime)
# COPY models/ ./models/

# Create models directory for mounting
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models/best_model.pt
ENV VOCAB_PATH=/app/models/vocab.json

# Create entrypoint script
RUN echo '#!/bin/bash\n\
exec python /app/devign_scan.py \
    --model "${MODEL_PATH}" \
    --vocab "${VOCAB_PATH}" \
    "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
