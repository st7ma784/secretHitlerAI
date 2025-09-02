Architecture Overview
====================

The Secret Hitler AI system is built with a modular, scalable architecture that combines advanced machine learning techniques with strategic game theory. This document provides a high-level overview of the system's components and their interactions.

System Architecture
-------------------

.. mermaid::

   graph TB
       subgraph "Frontend Layer"
           UI[Training Interface]
           API_CLIENT[API Client]
       end
       
       subgraph "API Layer"
           FASTAPI[FastAPI Server]
           ENDPOINTS[REST Endpoints]
       end
       
       subgraph "Core AI System"
           LLM[LLM Trainer]
           WORLD[World View System]
           AGENTS[Enhanced Agents]
           CHECKPOINT[Checkpoint Manager]
       end
       
       subgraph "Training System"
           LORA[LoRA Adaptation]
           RLHF[RLHF Training]
           SELF[Self Training]
           METRICS[Metrics Tracking]
       end
       
       subgraph "Game Engine"
           GAME[Game Logic]
           RULES[Rule Engine]
           STATE[State Management]
       end
       
       subgraph "Storage Layer"
           MODELS[Model Storage]
           CHECKPOINTS[Checkpoints]
           LOGS[Training Logs]
           DATA[Game Data]
       end
       
       UI --> API_CLIENT
       API_CLIENT --> FASTAPI
       FASTAPI --> ENDPOINTS
       ENDPOINTS --> LLM
       ENDPOINTS --> WORLD
       ENDPOINTS --> AGENTS
       ENDPOINTS --> CHECKPOINT
       
       LLM --> LORA
       LLM --> RLHF
       LLM --> SELF
       LLM --> METRICS
       
       AGENTS --> WORLD
       AGENTS --> GAME
       WORLD --> STATE
       GAME --> RULES
       
       CHECKPOINT --> MODELS
       METRICS --> LOGS
       SELF --> DATA
       LLM --> CHECKPOINTS

Core Components
---------------

1. **LLM Training System**
   
   * **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
   * **RLHF (Reinforcement Learning from Human Feedback)**: Strategic optimization
   * **Model Checkpointing**: Persistent learning and state management
   * **Performance Metrics**: Real-time training monitoring

2. **Agent World View System**
   
   * **Strategic Context**: Comprehensive game state awareness
   * **Player Profiling**: Trust, suspicion, and behavioral analysis
   * **Historical Analysis**: Voting patterns and conversation insights
   * **Decision Framework**: LLM + rule-based strategic reasoning

3. **Enhanced AI Agents**
   
   * **Role-Specific Intelligence**: Liberal, Fascist, and Hitler strategies
   * **Adaptive Learning**: Continuous improvement through experience
   * **Context Integration**: Real-time world view updates
   * **Performance Tracking**: Decision confidence and outcome analysis

4. **Training Infrastructure**
   
   * **Self-Training Orchestrator**: Automated training sessions
   * **Multi-Agent Coordination**: Parallel agent development
   * **Curriculum Learning**: Progressive difficulty scaling
   * **WandB Integration**: Experiment tracking and visualization

Data Flow
---------

The system follows a clear data flow pattern:

1. **Input Processing**
   
   * Game state updates from the game engine
   * Player actions and conversation data
   * Training configuration from the UI

2. **Context Building**
   
   * World view system processes all inputs
   * Strategic context is built for each agent
   * Historical patterns are analyzed and stored

3. **Decision Making**
   
   * Enhanced agents receive strategic context
   * LLM generates action suggestions
   * Rule-based logic provides strategic modifiers
   * Final decisions are made and executed

4. **Learning Loop**
   
   * Decision outcomes are recorded
   * Training data is collected and processed
   * Models are updated through RLHF
   * Checkpoints are saved for persistence

Technology Stack
----------------

**Backend**

* **Python 3.11+**: Core runtime
* **FastAPI**: REST API framework
* **PyTorch**: Deep learning framework
* **Transformers**: Pre-trained language models
* **PEFT**: Parameter-efficient fine-tuning
* **WandB**: Experiment tracking

**Frontend**

* **HTML/CSS/JavaScript**: Training interface
* **Tailwind CSS**: Styling framework
* **Chart.js**: Metrics visualization

**Infrastructure**

* **Docker**: Containerization
* **Docker Compose**: Multi-service orchestration
* **GitHub Actions**: CI/CD pipeline
* **Sphinx**: Documentation generation

Scalability Considerations
--------------------------

The architecture is designed for scalability:

**Horizontal Scaling**

* Multiple training processes can run in parallel
* Agent instances can be distributed across machines
* API endpoints support load balancing

**Vertical Scaling**

* GPU acceleration for training workloads
* Memory-efficient LoRA adapters
* Optimized checkpoint storage

**Performance Optimization**

* Asynchronous processing throughout
* Efficient model loading and caching
* Batch processing for training operations

Security Features
-----------------

* **Input Validation**: All API inputs are validated
* **Rate Limiting**: Protection against abuse
* **Secure Defaults**: Safe configuration options
* **Containerization**: Isolated execution environment

Monitoring and Observability
-----------------------------

* **Real-time Metrics**: Training progress and performance
* **Structured Logging**: Comprehensive system logs
* **Health Checks**: Service availability monitoring
* **Performance Profiling**: Resource usage tracking

Extension Points
----------------

The architecture provides several extension points:

* **Custom Agents**: Implement new agent strategies
* **Training Methods**: Add new training algorithms
* **Game Variants**: Support different game rules
* **Storage Backends**: Alternative data storage options
* **Deployment Targets**: Various hosting environments
