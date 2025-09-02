Welcome to Secret Hitler AI Documentation
==========================================

Secret Hitler AI is an advanced artificial intelligence system designed to play the strategic social deduction game Secret Hitler. This project combines cutting-edge machine learning techniques with game theory to create intelligent agents capable of strategic gameplay, deception, and social reasoning.

.. note::
   This project is for educational and research purposes, exploring AI capabilities in strategic social games.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <h3>🧠 Advanced AI Training</h3>
       <p>LoRA-based parameter-efficient fine-tuning with RLHF for strategic gameplay optimization.</p>
     </div>
     <div class="feature-card">
       <h3>🎯 Strategic Intelligence</h3>
       <p>Role-specific agent behaviors with comprehensive world view and decision-making frameworks.</p>
     </div>
     <div class="feature-card">
       <h3>� Strategy Analysis</h3>
       <p>Computational analysis of 10,000+ games revealing optimal strategies and behavioral patterns.</p>
     </div>
     <div class="feature-card">
       <h3>�📈 Real-time Learning</h3>
       <p>Continuous improvement through self-play with automated training orchestration.</p>
     </div>
     <div class="feature-card">
       <h3>🐳 Docker Deployment</h3>
       <p>Containerized deployment with multi-stage builds and orchestration support.</p>
     </div>
   </div>


Quick Start
-----------

Get started with Secret Hitler AI in just a few steps:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/secretHitlerAI.git
   cd secretHitlerAI

   # Start with Docker Compose
   docker-compose up

   # Access the training interface
   open http://localhost:8000/training

.. tip::
   For GPU-accelerated training, use the training profile:
   
   .. code-block:: bash
   
      docker-compose --profile training up

System Architecture
-------------------

The Secret Hitler AI system follows a modular, scalable architecture:

.. mermaid::

   graph TB
       subgraph "Frontend"
           UI[Training Interface]
       end
       
       subgraph "API Layer"
           API[FastAPI Server]
           ENDPOINTS[REST Endpoints]
       end
       
       subgraph "AI Core"
           LLM[LLM Trainer]
           AGENTS[Enhanced Agents]
           WORLD[World View System]
           CHECKPOINT[Checkpoint Manager]
       end
       
       subgraph "Training"
           LORA[LoRA Adaptation]
           RLHF[RLHF Training]
           SELF[Self Training]
       end
       
       subgraph "Storage"
           MODELS[Model Storage]
           DATA[Training Data]
           LOGS[Metrics & Logs]
       end
       
       UI --> API
       API --> ENDPOINTS
       ENDPOINTS --> LLM
       ENDPOINTS --> AGENTS
       LLM --> LORA
       LLM --> RLHF
       AGENTS --> WORLD
       AGENTS --> SELF
       CHECKPOINT --> MODELS
       SELF --> DATA
       LLM --> LOGS

Key Components
--------------

**Strategy Analysis System**
   Comprehensive gameplay analysis derived from 10,000+ AI vs AI games, revealing optimal strategies, behavioral patterns, and psychological insights with mathematical precision.

**LLM Training System**
   Advanced language model fine-tuning using LoRA (Low-Rank Adaptation) and RLHF (Reinforcement Learning from Human Feedback) for parameter-efficient strategic optimization.

**Agent World View**
   Comprehensive strategic context system that maintains player profiles, voting history, conversation analysis, and strategic recommendations for intelligent decision-making.

**Enhanced AI Agents**
   Role-specific intelligent agents (Liberal, Fascist, Hitler) with continuous learning capabilities, performance tracking, and strategic adaptation.

**Training Infrastructure**
   Automated self-training orchestration with curriculum learning, multi-agent coordination, and comprehensive performance monitoring.

**Checkpoint Management**
   Intelligent model persistence with automatic cleanup, metadata tracking, and recovery mechanisms for robust training workflows.

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Design:

   architecture/overview

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/endpoints

.. toctree::
   :maxdepth: 2
   :caption: Strategy & Analysis:

   strategy/index

.. toctree::
   :maxdepth: 2
   :caption: Training System:

   training/overview

.. toctree::
   :maxdepth: 2
   :caption: Deployment:

   deployment/docker

.. toctree::
   :maxdepth: 2
   :caption: Development:

   development/setup

.. toctree::
   :maxdepth: 2
   :caption: Code Reference:

   modules/backend

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

   examples/usage
   license
   contributing

Getting Help
------------

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Sample code and usage patterns
- **Community**: Join discussions and share experiences

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
