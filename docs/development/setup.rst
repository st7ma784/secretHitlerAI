Development Setup
=================

This guide covers setting up a development environment for Secret Hitler AI.

Prerequisites
-------------

**System Requirements:**

- Python 3.9+ (3.11 recommended)
- Git
- Docker and Docker Compose (optional but recommended)
- CUDA-compatible GPU (optional, for training acceleration)

**Hardware Recommendations:**

- 8GB+ RAM (16GB+ for training)
- 50GB+ free disk space
- NVIDIA GPU with 8GB+ VRAM (for GPU training)

Local Development Setup
-----------------------

**1. Clone the Repository**

.. code-block:: bash

   git clone https://github.com/yourusername/secretHitlerAI.git
   cd secretHitlerAI

**2. Create Virtual Environment**

.. code-block:: bash

   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Or using conda
   conda create -n secret-hitler-ai python=3.11
   conda activate secret-hitler-ai

**3. Install Dependencies**

.. code-block:: bash

   # Backend dependencies
   pip install -r backend/requirements.txt

   # Development dependencies
   pip install -r backend/requirements-dev.txt

   # Documentation dependencies
   pip install -r docs/requirements.txt

**4. Environment Configuration**

Create a `.env` file in the project root:

.. code-block:: bash

   # Development settings
   NODE_ENV=development
   DEBUG=true
   PORT=8000

   # Training settings
   WANDB_MODE=offline
   CUDA_VISIBLE_DEVICES=0

   # Paths
   PYTHONPATH=/path/to/secretHitlerAI

**5. Start Development Server**

.. code-block:: bash

   cd backend
   python simple_server.py

   # Access the interface
   open http://localhost:8000/training

Docker Development Setup
------------------------

**1. Development with Docker Compose**

.. code-block:: bash

   # Start development environment
   docker-compose --profile dev up

   # Or with specific services
   docker-compose up secret-hitler-ai-dev

**2. Development Container**

The development container includes:

- Hot-reload functionality
- Development tools and debuggers
- Volume mounts for live code editing
- Jupyter notebook server
- TensorBoard for metrics visualization

**3. Access Development Services**

- **Main Application**: http://localhost:8000
- **Jupyter Notebooks**: http://localhost:8888
- **TensorBoard**: http://localhost:6006

Code Organization
-----------------

**Project Structure:**

.. code-block:: text

   secretHitlerAI/
   ├── backend/                 # Python backend
   │   ├── api/                # API endpoints
   │   ├── game/               # Game logic and agents
   │   ├── training/           # ML training system
   │   ├── models/             # Data models
   │   └── simple_server.py    # Main server
   ├── frontend/               # Frontend assets
   ├── docs/                   # Documentation
   ├── tests/                  # Test suites
   ├── checkpoints/            # Model checkpoints
   ├── logs/                   # Application logs
   ├── data/                   # Training data
   └── docker-compose.yml      # Docker orchestration

**Backend Structure:**

.. code-block:: text

   backend/
   ├── api/
   │   ├── __init__.py
   │   └── game.py             # Game API endpoints
   ├── game/
   │   ├── __init__.py
   │   ├── agent_worldview.py  # Agent world view system
   │   ├── enhanced_ai_agent.py # Enhanced AI agents
   │   └── secret_hitler.py    # Game logic
   ├── training/
   │   ├── __init__.py
   │   ├── llm_trainer.py      # LLM training system
   │   ├── model_checkpointing.py # Checkpoint management
   │   └── self_trainer.py     # Self-training orchestrator
   └── models/
       ├── __init__.py
       └── game_models.py      # Pydantic models

Development Workflow
--------------------

**1. Feature Development**

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/new-training-method

   # Make changes and test
   python -m pytest tests/

   # Run linting
   flake8 backend/
   black backend/

   # Commit changes
   git add .
   git commit -m "Add new training method"

**2. Testing**

.. code-block:: bash

   # Run all tests
   pytest

   # Run specific test file
   pytest tests/test_training.py

   # Run with coverage
   pytest --cov=backend tests/

   # Run integration tests
   pytest tests/integration/

**3. Code Quality**

.. code-block:: bash

   # Format code
   black backend/

   # Check style
   flake8 backend/

   # Type checking
   mypy backend/

   # Security check
   bandit -r backend/

Development Tools
-----------------

**Recommended IDE Setup:**

- **VS Code** with Python extension
- **PyCharm Professional** for advanced debugging
- **Jupyter Lab** for interactive development

**VS Code Extensions:**

- Python
- Pylance
- Docker
- GitLens
- Thunder Client (for API testing)

**Debugging:**

.. code-block:: python

   # Add breakpoints in code
   import pdb; pdb.set_trace()

   # Or use VS Code debugger
   # Set breakpoints and run with F5

**Environment Variables:**

.. code-block:: bash

   # Development
   export DEBUG=true
   export LOG_LEVEL=debug

   # Training
   export WANDB_MODE=offline
   export CUDA_VISIBLE_DEVICES=0

Testing
-------

**Test Structure:**

.. code-block:: text

   tests/
   ├── unit/                   # Unit tests
   │   ├── test_agents.py
   │   ├── test_training.py
   │   └── test_api.py
   ├── integration/            # Integration tests
   │   ├── test_game_flow.py
   │   └── test_training_flow.py
   └── fixtures/               # Test fixtures
       ├── game_states.json
       └── training_data.json

**Running Tests:**

.. code-block:: bash

   # All tests
   pytest

   # Unit tests only
   pytest tests/unit/

   # Integration tests only
   pytest tests/integration/

   # With coverage
   pytest --cov=backend --cov-report=html

   # Parallel execution
   pytest -n auto

**Writing Tests:**

.. code-block:: python

   import pytest
   from backend.game.enhanced_ai_agent import EnhancedAIAgent

   class TestEnhancedAIAgent:
       def test_agent_initialization(self):
           agent = EnhancedAIAgent(role="liberal")
           assert agent.role == "liberal"
           assert agent.world_view is not None

       @pytest.mark.asyncio
       async def test_agent_decision_making(self):
           agent = EnhancedAIAgent(role="liberal")
           decision = await agent.make_decision(game_state)
           assert decision is not None

Documentation Development
-------------------------

**Building Documentation:**

.. code-block:: bash

   cd docs/

   # Install dependencies
   pip install -r requirements.txt

   # Generate API docs
   make apidoc

   # Build HTML documentation
   make html

   # Serve documentation locally
   make livehtml

**Documentation Structure:**

- **Architecture**: System design and components
- **API**: Endpoint documentation
- **Training**: ML system documentation
- **Deployment**: Docker and production guides
- **Development**: Setup and contribution guides

**Writing Documentation:**

.. code-block:: rst

   My New Feature
   ==============

   This section describes the new feature.

   Usage Example
   -------------

   .. code-block:: python

      from backend.new_feature import NewFeature
      
      feature = NewFeature()
      result = feature.process()

Troubleshooting
---------------

**Common Issues:**

**Port Already in Use:**

.. code-block:: bash

   # Find process using port
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>

**Import Errors:**

.. code-block:: bash

   # Set PYTHONPATH
   export PYTHONPATH=/path/to/secretHitlerAI:$PYTHONPATH

**CUDA Issues:**

.. code-block:: bash

   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"

   # Check GPU memory
   nvidia-smi

**Docker Issues:**

.. code-block:: bash

   # Rebuild containers
   docker-compose build --no-cache

   # Clean up
   docker system prune -a

**Performance Issues:**

.. code-block:: bash

   # Profile code
   python -m cProfile -o profile.stats script.py

   # Memory profiling
   pip install memory-profiler
   python -m memory_profiler script.py

Contributing Guidelines
-----------------------

**Code Style:**

- Follow PEP 8
- Use Black for formatting
- Add type hints
- Write docstrings for all functions
- Keep functions small and focused

**Commit Messages:**

.. code-block:: text

   feat: add new training method
   fix: resolve checkpoint loading issue
   docs: update API documentation
   test: add unit tests for agents
   refactor: improve code organization

**Pull Request Process:**

1. Create feature branch
2. Make changes with tests
3. Update documentation
4. Run all tests and linting
5. Submit pull request
6. Address review feedback
7. Merge after approval

**Code Review Checklist:**

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed
