Docker Deployment
=================

This guide covers deploying Secret Hitler AI using Docker containers for different environments and use cases.

Quick Start
-----------

The fastest way to get started is using Docker Compose:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/secretHitlerAI.git
   cd secretHitlerAI

   # Start the production service
   docker-compose up secret-hitler-ai

   # Access the training interface
   open http://localhost:8000/training

Docker Images
-------------

The project provides three specialized Docker images:

**Production Image**
   Optimized for production deployment with minimal dependencies and security hardening.

   .. code-block:: bash

      docker build --target production -t secret-hitler-ai:prod .
      docker run -p 8000:8000 secret-hitler-ai:prod

**Development Image**
   Includes development tools, debugging capabilities, and hot-reload functionality.

   .. code-block:: bash

      docker-compose --profile dev up secret-hitler-ai-dev

**Training Image**
   Specialized for ML workloads with GPU support, Jupyter notebooks, and TensorBoard.

   .. code-block:: bash

      docker-compose --profile training up secret-hitler-ai-training

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Environment Variables
   :widths: 25 25 50
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``PYTHONPATH``
     - ``/app``
     - Python module search path
   * - ``PORT``
     - ``8000``
     - Server port
   * - ``NODE_ENV``
     - ``production``
     - Environment mode
   * - ``WANDB_MODE``
     - ``offline``
     - WandB logging mode
   * - ``CUDA_VISIBLE_DEVICES``
     - ``0``
     - GPU device selection

Volume Mounts
~~~~~~~~~~~~~

Persistent data storage:

.. code-block:: yaml

   volumes:
     - ./checkpoints:/app/checkpoints  # Model checkpoints
     - ./logs:/app/logs                # Application logs
     - ./data:/app/data                # Training data
     - ./models:/app/models            # Trained models
     - ./wandb:/app/wandb              # WandB artifacts

Docker Compose Profiles
-----------------------

The project uses Docker Compose profiles for different deployment scenarios:

**Default Profile (Production)**

.. code-block:: bash

   docker-compose up

Starts only the production service with minimal resources.

**Development Profile**

.. code-block:: bash

   docker-compose --profile dev up

Includes:
- Development server with hot-reload
- Volume mounts for live code editing
- Debug tools and utilities

**Training Profile**

.. code-block:: bash

   docker-compose --profile training up

Includes:
- ML training environment
- Jupyter notebook server (port 8888)
- TensorBoard server (port 6006)
- GPU support (if available)

**Full Stack Profile**

.. code-block:: bash

   docker-compose --profile cache --profile database up

Includes:
- Redis for caching
- PostgreSQL for data storage
- Nginx reverse proxy

Production Deployment
---------------------

Multi-Stage Build
~~~~~~~~~~~~~~~~~

The Dockerfile uses multi-stage builds for optimal production images:

.. code-block:: dockerfile

   # Base stage with common dependencies
   FROM python:3.11-slim as base
   
   # Development stage with dev tools
   FROM base as development
   
   # Production stage optimized for deployment
   FROM base as production

Security Features
~~~~~~~~~~~~~~~~~

Production images include security hardening:

- **Non-root user**: Runs as ``appuser`` with limited privileges
- **Minimal attack surface**: Only necessary packages installed
- **Health checks**: Built-in container health monitoring
- **Resource limits**: Memory and CPU constraints

Health Checks
~~~~~~~~~~~~~

Built-in health check endpoint:

.. code-block:: bash

   # Manual health check
   curl -f http://localhost:8000/api/ai/training-status

   # Docker health check (automatic)
   docker ps  # Shows health status

Scaling and Load Balancing
--------------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

Scale the application across multiple containers:

.. code-block:: bash

   # Scale to 3 instances
   docker-compose up --scale secret-hitler-ai=3

   # With load balancer
   docker-compose --profile proxy up --scale secret-hitler-ai=3

Load Balancing with Nginx
~~~~~~~~~~~~~~~~~~~~~~~~~

The included Nginx configuration provides:

- **Load balancing**: Distribute requests across instances
- **SSL termination**: HTTPS support
- **Static file serving**: Efficient asset delivery
- **Health checks**: Automatic failover

GPU Support
-----------

For training workloads with GPU acceleration:

**Docker Compose with GPU**

.. code-block:: yaml

   services:
     secret-hitler-ai-training:
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]

**Manual GPU Setup**

.. code-block:: bash

   # Build training image
   docker build --target training -t secret-hitler-ai:training .

   # Run with GPU support
   docker run --gpus all -p 8000:8000 secret-hitler-ai:training

Monitoring and Logging
----------------------

Container Logs
~~~~~~~~~~~~~~

Access application logs:

.. code-block:: bash

   # View logs
   docker-compose logs secret-hitler-ai

   # Follow logs in real-time
   docker-compose logs -f secret-hitler-ai

   # View specific service logs
   docker logs <container_id>

Metrics Collection
~~~~~~~~~~~~~~~~~~

The training image includes monitoring tools:

- **TensorBoard**: ML metrics visualization
- **WandB**: Experiment tracking
- **Prometheus**: System metrics (optional)

Persistent Storage
------------------

Data Persistence
~~~~~~~~~~~~~~~~

Important data directories that should be persisted:

.. code-block:: bash

   # Create host directories
   mkdir -p checkpoints logs data models wandb

   # Set proper permissions
   chmod 755 checkpoints logs data models wandb

Backup Strategy
~~~~~~~~~~~~~~~

Recommended backup approach:

.. code-block:: bash

   # Backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   
   # Backup checkpoints
   tar -czf "backup_checkpoints_${DATE}.tar.gz" checkpoints/
   
   # Backup training data
   tar -czf "backup_data_${DATE}.tar.gz" data/
   
   # Upload to cloud storage
   aws s3 cp "backup_checkpoints_${DATE}.tar.gz" s3://your-backup-bucket/

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Port Already in Use**

.. code-block:: bash

   # Find process using port 8000
   lsof -i :8000
   
   # Kill the process
   kill -9 <PID>
   
   # Or use different port
   docker-compose up -e PORT=8001

**Permission Denied**

.. code-block:: bash

   # Fix volume permissions
   sudo chown -R 1000:1000 checkpoints logs data

**Out of Memory**

.. code-block:: bash

   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   
   # Or reduce batch size in training config
   curl -X POST http://localhost:8000/api/ai/configure-training \
     -d '{"games_per_session": 10}'

**GPU Not Detected**

.. code-block:: bash

   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   
   # Install nvidia-container-toolkit if needed
   sudo apt-get install nvidia-container-toolkit

Debugging
~~~~~~~~~

Debug container issues:

.. code-block:: bash

   # Run interactive shell
   docker run -it --entrypoint /bin/bash secret-hitler-ai:prod
   
   # Check container logs
   docker logs --details <container_id>
   
   # Inspect container
   docker inspect <container_id>

Performance Optimization
------------------------

Resource Limits
~~~~~~~~~~~~~~~

Set appropriate resource limits:

.. code-block:: yaml

   services:
     secret-hitler-ai:
       deploy:
         resources:
           limits:
             memory: 2G
             cpus: '1.0'
           reservations:
             memory: 1G
             cpus: '0.5'

Caching
~~~~~~~

Optimize Docker build caching:

.. code-block:: bash

   # Use BuildKit for better caching
   export DOCKER_BUILDKIT=1
   
   # Build with cache mount
   docker build --cache-from secret-hitler-ai:latest .

Multi-Architecture Builds
~~~~~~~~~~~~~~~~~~~~~~~~~

Build for multiple architectures:

.. code-block:: bash

   # Create multi-arch builder
   docker buildx create --name multiarch --use
   
   # Build for multiple platforms
   docker buildx build --platform linux/amd64,linux/arm64 \
     --push -t secret-hitler-ai:latest .
