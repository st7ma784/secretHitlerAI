Contributing to Secret Hitler AI
=================================

We welcome contributions from the community! This guide will help you get started with contributing to the Secret Hitler AI project.

Code of Conduct
---------------

By participating in this project, you agree to abide by our Code of Conduct:

- **Be respectful**: Treat all community members with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get started
- **Be constructive**: Provide helpful feedback and suggestions
- **Be collaborative**: Work together to improve the project
- **Be educational**: Remember this is a research and educational project

Types of Contributions
----------------------

We welcome various types of contributions:

**Code Contributions**
- Bug fixes and improvements
- New features and enhancements
- Performance optimizations
- Test coverage improvements

**Documentation**
- API documentation improvements
- Tutorial and example additions
- Architecture documentation
- Translation contributions

**Research and Analysis**
- Training methodology improvements
- Performance analysis and benchmarking
- Strategic gameplay research
- AI ethics and safety considerations

**Community Support**
- Helping other users with issues
- Answering questions in discussions
- Reviewing pull requests
- Sharing usage examples and case studies

Getting Started
---------------

**1. Fork and Clone the Repository**

.. code-block:: bash

   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/yourusername/secretHitlerAI.git
   cd secretHitlerAI

   # Add the upstream repository
   git remote add upstream https://github.com/originalowner/secretHitlerAI.git

**2. Set Up Development Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install dependencies
   pip install -r backend/requirements.txt
   pip install -r backend/requirements-dev.txt

   # Install pre-commit hooks
   pre-commit install

**3. Create a Feature Branch**

.. code-block:: bash

   # Create and switch to a new branch
   git checkout -b feature/your-feature-name

   # Or for bug fixes
   git checkout -b fix/issue-description

Development Workflow
--------------------

**1. Development Process**

.. code-block:: bash

   # Make your changes
   # Write tests for new functionality
   # Update documentation as needed

   # Run tests
   pytest

   # Run linting
   flake8 backend/
   black backend/

   # Type checking
   mypy backend/

**2. Commit Guidelines**

Use conventional commit messages:

.. code-block:: text

   feat: add new training algorithm
   fix: resolve checkpoint loading issue
   docs: update API documentation
   test: add unit tests for agents
   refactor: improve code organization
   style: fix formatting issues
   perf: optimize training performance
   chore: update dependencies

**3. Pull Request Process**

.. code-block:: bash

   # Push your changes
   git push origin feature/your-feature-name

   # Create pull request on GitHub
   # Fill out the pull request template
   # Wait for review and address feedback

Code Standards
--------------

**Python Code Style**

- Follow PEP 8 guidelines
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints for all functions
- Write comprehensive docstrings

**Example Code Style:**

.. code-block:: python

   from typing import Optional, Dict, Any
   import logging

   logger = logging.getLogger(__name__)

   class EnhancedAIAgent:
       """Enhanced AI agent with strategic reasoning capabilities.
       
       This class implements an AI agent that can play Secret Hitler
       with advanced strategic reasoning and learning capabilities.
       
       Args:
           role: The agent's role in the game
           model_path: Path to the trained model
           config: Configuration parameters
       """
       
       def __init__(
           self, 
           role: str, 
           model_path: Optional[str] = None,
           config: Optional[Dict[str, Any]] = None
       ) -> None:
           self.role = role
           self.model_path = model_path
           self.config = config or {}
           
           logger.info(f"Initialized {role} agent with config: {self.config}")
       
       async def make_decision(
           self, 
           game_state: Dict[str, Any]
       ) -> Dict[str, Any]:
           """Make a strategic decision based on current game state.
           
           Args:
               game_state: Current state of the game
               
           Returns:
               Dictionary containing the agent's decision
               
           Raises:
               ValueError: If game_state is invalid
           """
           if not game_state:
               raise ValueError("Game state cannot be empty")
           
           # Implementation here
           decision = {"action": "vote", "target": "player_1"}
           
           logger.debug(f"Agent {self.role} decided: {decision}")
           return decision

**Documentation Standards**

- Use reStructuredText (RST) for documentation
- Include docstrings for all public functions and classes
- Provide usage examples in documentation
- Keep documentation up-to-date with code changes

**Testing Standards**

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Use pytest for testing framework
- Include integration tests for complex features

.. code-block:: python

   import pytest
   from unittest.mock import Mock, patch
   from backend.game.enhanced_ai_agent import EnhancedAIAgent

   class TestEnhancedAIAgent:
       """Test suite for EnhancedAIAgent class."""
       
       def test_agent_initialization(self):
           """Test agent initializes correctly."""
           agent = EnhancedAIAgent(role="liberal")
           assert agent.role == "liberal"
           assert agent.config == {}
       
       @pytest.mark.asyncio
       async def test_make_decision_valid_state(self):
           """Test decision making with valid game state."""
           agent = EnhancedAIAgent(role="liberal")
           game_state = {"phase": "election", "players": 6}
           
           decision = await agent.make_decision(game_state)
           
           assert "action" in decision
           assert decision["action"] in ["vote", "nominate", "investigate"]
       
       @pytest.mark.asyncio
       async def test_make_decision_invalid_state(self):
           """Test decision making with invalid game state."""
           agent = EnhancedAIAgent(role="liberal")
           
           with pytest.raises(ValueError, match="Game state cannot be empty"):
               await agent.make_decision({})

Contribution Areas
------------------

**High Priority Areas**

1. **Training Algorithm Improvements**
   - Implement new LoRA configurations
   - Optimize RLHF training process
   - Add curriculum learning enhancements
   - Improve convergence speed

2. **Agent Strategic Intelligence**
   - Enhance world view system
   - Improve decision-making algorithms
   - Add role-specific strategies
   - Implement meta-learning capabilities

3. **Performance Optimization**
   - Optimize training performance
   - Reduce memory usage
   - Improve inference speed
   - Add GPU acceleration support

4. **Testing and Quality Assurance**
   - Increase test coverage
   - Add integration tests
   - Implement performance benchmarks
   - Add security testing

**Medium Priority Areas**

1. **Documentation and Examples**
   - Add more usage examples
   - Improve API documentation
   - Create tutorial content
   - Add architecture diagrams

2. **User Interface Improvements**
   - Enhance training dashboard
   - Add real-time monitoring
   - Improve visualization
   - Add configuration management

3. **Deployment and Infrastructure**
   - Improve Docker configurations
   - Add Kubernetes support
   - Enhance CI/CD pipeline
   - Add monitoring and logging

**Research Areas**

1. **AI Ethics and Safety**
   - Implement bias detection
   - Add fairness metrics
   - Research AI alignment
   - Study social implications

2. **Advanced AI Techniques**
   - Explore multi-agent reinforcement learning
   - Implement advanced neural architectures
   - Research emergent communication
   - Study strategic reasoning

3. **Game Theory Applications**
   - Analyze optimal strategies
   - Study equilibrium points
   - Research coalition formation
   - Analyze information asymmetry

Issue Guidelines
----------------

**Reporting Bugs**

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce the problem
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

**Bug Report Template:**

.. code-block:: text

   **Bug Description**
   A clear description of what the bug is.

   **To Reproduce**
   Steps to reproduce the behavior:
   1. Go to '...'
   2. Click on '....'
   3. Scroll down to '....'
   4. See error

   **Expected Behavior**
   A clear description of what you expected to happen.

   **Screenshots**
   If applicable, add screenshots to help explain your problem.

   **Environment:**
   - OS: [e.g. Ubuntu 20.04]
   - Python Version: [e.g. 3.11.0]
   - Docker Version: [e.g. 20.10.17]
   - GPU: [e.g. NVIDIA RTX 3080]

   **Additional Context**
   Add any other context about the problem here.

**Feature Requests**

When requesting features, please include:

- Clear description of the desired feature
- Use case and motivation
- Proposed implementation approach
- Potential impact on existing functionality

**Feature Request Template:**

.. code-block:: text

   **Feature Description**
   A clear description of what you want to happen.

   **Motivation**
   Why is this feature needed? What problem does it solve?

   **Proposed Solution**
   A clear description of how you envision this feature working.

   **Alternatives Considered**
   Any alternative solutions or features you've considered.

   **Additional Context**
   Add any other context, mockups, or examples about the feature request.

Pull Request Guidelines
-----------------------

**Before Submitting**

- [ ] Code follows the project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

**Pull Request Template**

.. code-block:: text

   ## Description
   Brief description of changes made.

   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed
   - [ ] All tests passing

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No new warnings introduced
   - [ ] Performance impact considered

**Review Process**

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality manually if needed
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer merges after all checks pass

Community Guidelines
--------------------

**Communication Channels**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Pull Requests**: Code review and collaboration
- **Documentation**: Comprehensive guides and references

**Getting Help**

If you need help:

1. Check existing documentation and examples
2. Search GitHub issues for similar problems
3. Ask questions in GitHub Discussions
4. Join community chat channels (if available)

**Mentorship**

New contributors are welcome! We provide:

- **Good First Issues**: Labeled issues suitable for beginners
- **Mentorship**: Experienced contributors help newcomers
- **Documentation**: Comprehensive setup and development guides
- **Code Review**: Constructive feedback on contributions

**Recognition**

We recognize contributors through:

- **Contributors List**: All contributors listed in README
- **Release Notes**: Major contributions highlighted
- **Community Spotlight**: Outstanding contributions featured
- **Maintainer Opportunities**: Active contributors invited to join team

Legal Considerations
--------------------

**Contributor License Agreement**

By contributing, you agree that:

- Your contributions will be licensed under the project's MIT License
- You have the right to make the contribution
- Your contribution is your original work or properly attributed
- You understand the educational and research nature of the project

**Intellectual Property**

- Ensure all contributions are original or properly licensed
- Respect third-party intellectual property rights
- Follow fair use guidelines for game-related content
- Maintain the educational and research focus of the project

**Privacy and Data**

- Do not include personal data in contributions
- Respect user privacy in any features added
- Follow data protection best practices
- Maintain the local-first approach of the project

Thank You!
----------

Thank you for your interest in contributing to Secret Hitler AI! Your contributions help advance AI research and education. Together, we can build amazing tools for understanding strategic reasoning and social interaction in AI systems.

For questions about contributing, please:

- Open an issue on GitHub
- Start a discussion in GitHub Discussions
- Contact the maintainers directly

We look forward to your contributions!
