# Multi-stage build for Secret Hitler AI
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    sphinx \
    sphinx-rtd-theme \
    sphinx-autodoc-typehints

# Copy source code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/data

# Expose port
EXPOSE 8000

# Development command
CMD ["python", "backend/simple_server.py"]

# Production stage
FROM base as production

# Copy only necessary files
COPY backend/ /app/backend/
COPY frontend/public/ /app/frontend/public/
COPY docs/ /app/docs/
COPY README.md /app/
COPY LICENSE /app/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories with proper permissions
RUN mkdir -p /app/checkpoints /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/ai/training-status || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "backend/simple_server.py"]

# Training stage (for ML workloads)
FROM base as training

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    torch \
    transformers \
    peft \
    wandb \
    tensorboard \
    jupyter

# Copy source code
COPY . /app/

# Create directories for training artifacts
RUN mkdir -p /app/checkpoints /app/logs /app/data /app/models /app/wandb

# Expose ports for Jupyter and TensorBoard
EXPOSE 8000 8888 6006

# Training command
CMD ["python", "backend/training/self_trainer.py"]
