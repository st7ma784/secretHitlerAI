Training System Overview
========================

The Secret Hitler AI training system implements state-of-the-art machine learning techniques to create intelligent agents capable of strategic gameplay. This document provides a comprehensive overview of the training methodology and implementation.

Training Philosophy
-------------------

Our training approach is built on three core principles:

1. **Parameter Efficiency**: Using LoRA (Low-Rank Adaptation) to minimize computational requirements while maintaining model performance
2. **Human Alignment**: Implementing RLHF (Reinforcement Learning from Human Feedback) to align AI behavior with strategic gameplay
3. **Continuous Learning**: Enabling agents to improve through self-play and experience accumulation

Training Architecture
---------------------

The training system architecture is visualized in our comprehensive training flow diagram (see :doc:`../strategy/visualizations` for detailed analysis).

.. mermaid::

   graph TB
       subgraph "Training Pipeline"
           DATA[Game Data] --> PROCESS[Data Processing]
           PROCESS --> LORA[LoRA Adaptation]
           LORA --> RLHF[RLHF Training]
           RLHF --> EVAL[Evaluation]
           EVAL --> CHECKPOINT[Checkpointing]
       end
       
       subgraph "Self-Training Loop"
           AGENTS[AI Agents] --> GAMES[Game Sessions]
           GAMES --> EXPERIENCE[Experience Collection]
           EXPERIENCE --> ANALYSIS[Performance Analysis]
           ANALYSIS --> UPDATE[Model Updates]
           UPDATE --> AGENTS
       end
       
       subgraph "Monitoring"
           METRICS[Training Metrics]
           WANDB[WandB Logging]
           DASHBOARD[Training Dashboard]
       end
       
       RLHF --> METRICS
       EVAL --> WANDB
       CHECKPOINT --> DASHBOARD

LoRA (Low-Rank Adaptation)
--------------------------

LoRA enables parameter-efficient fine-tuning by introducing low-rank matrices that adapt the pre-trained model weights.

**Key Benefits:**

* **99%+ Parameter Reduction**: Only fine-tune a small subset of parameters
* **Memory Efficiency**: Significantly reduced memory requirements
* **Fast Training**: Faster convergence compared to full fine-tuning
* **Modular Adapters**: Different adapters for different roles/strategies

**Implementation Details:**

.. code-block:: python

   # LoRA Configuration
   lora_config = LoRAConfig(
       rank=16,  # Liberal agents
       alpha=32,
       dropout=0.1,
       target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
   )

**Role-Specific Configurations:**

* **Liberal Agents (Rank 16)**: Focus on truth-telling and coalition building
* **Fascist Agents (Rank 32)**: Enhanced deception and misdirection capabilities  
* **Hitler Agent (Rank 64)**: Maximum complexity for staying hidden

RLHF (Reinforcement Learning from Human Feedback)
--------------------------------------------------

RLHF aligns the AI's behavior with strategic gameplay through reward modeling and policy optimization.

**Training Process:**

1. **Reward Modeling**: Learn to predict game outcomes and strategic value
2. **Policy Training**: Optimize agent actions using PPO (Proximal Policy Optimization)
3. **KL Divergence Penalty**: Prevent the model from deviating too far from the base model

**Reward Function Design:**

.. code-block:: python

   def calculate_reward(action, outcome, role):
       base_reward = 1.0 if outcome['success'] else -0.5
       
       # Role-specific bonuses
       if role == 'liberal':
           if action['type'] == 'investigate' and outcome['revealed_fascist']:
               base_reward += 2.0
       elif role == 'fascist':
           if action['type'] == 'mislead' and outcome['liberals_confused']:
               base_reward += 1.5
       elif role == 'hitler':
           if action['type'] == 'stay_hidden' and not outcome['suspected']:
               base_reward += 1.0
               
       return base_reward

Self-Training Orchestrator
--------------------------

The self-training system enables continuous improvement through automated game sessions.

**Training Loop:**

1. **Game Generation**: Create diverse game scenarios
2. **Agent Deployment**: Deploy agents with different strategies
3. **Experience Collection**: Gather gameplay data and outcomes
4. **Batch Training**: Update models using collected experience
5. **Performance Evaluation**: Assess agent improvements
6. **Checkpoint Management**: Save successful model states

**Configuration Options:**

.. code-block:: yaml

   training:
     games_per_session: 20
     training_interval_minutes: 30
     enable_live_learning: true
     curriculum_learning: true
     difficulty_progression: [easy, medium, hard, expert]

Training Metrics
----------------

Comprehensive metrics tracking enables monitoring and optimization:

**Core Metrics:**

* **Training Loss**: Model convergence indicator
* **Reward Score**: Strategic performance measure
* **Policy Gradient**: Learning progress indicator
* **KL Divergence**: Model stability measure
* **Value Function**: State evaluation accuracy

**Performance Metrics:**

* **Win Rate by Role**: Success rate for each agent type
* **Decision Confidence**: Agent certainty in choices
* **Strategic Accuracy**: Alignment with optimal play
* **Adaptation Speed**: Learning rate measurement

**Real-time Monitoring:**

The training dashboard provides live updates on all metrics, enabling:

* **Early Stopping**: Prevent overfitting
* **Hyperparameter Tuning**: Optimize training parameters
* **Performance Comparison**: Evaluate different configurations
* **Resource Monitoring**: Track computational usage

Curriculum Learning
-------------------

Progressive difficulty scaling improves training efficiency:

**Difficulty Levels:**

1. **Beginner**: Simple scenarios, clear roles
2. **Intermediate**: Mixed strategies, some deception
3. **Advanced**: Complex social dynamics, multiple threats
4. **Expert**: Professional-level gameplay, subtle strategies

**Progression Criteria:**

* Minimum win rate thresholds
* Consistent performance over multiple sessions
* Strategic diversity in decision making
* Adaptation to opponent strategies

Multi-Agent Training
--------------------

Simultaneous training of multiple agents with different specializations:

**Agent Specializations:**

* **Truth-Teller**: Focuses on honest communication
* **Investigator**: Specializes in role detection
* **Deceiver**: Masters misdirection techniques
* **Coordinator**: Excels at team coordination
* **Infiltrator**: Expert at staying hidden

**Training Coordination:**

* **Parallel Training**: Multiple agents train simultaneously
* **Experience Sharing**: Agents learn from each other's experiences
* **Competitive Evolution**: Agents improve by competing against each other
* **Diversity Maintenance**: Prevent convergence to single strategy

Checkpointing Strategy
----------------------

Intelligent checkpoint management ensures optimal model persistence:

**Checkpoint Triggers:**

* **Performance Milestones**: Save when agents reach new performance levels
* **Training Intervals**: Regular saves every N training steps
* **Stability Checks**: Save stable, well-performing models
* **Experiment Boundaries**: Save at the end of training experiments

**Checkpoint Content:**

* Model weights and optimizer state
* Training metrics and performance history
* Configuration parameters and hyperparameters
* Agent-specific world view and experience data

**Storage Management:**

* **Automatic Cleanup**: Remove old, low-performing checkpoints
* **Compression**: Efficient storage of checkpoint data
* **Metadata Tracking**: Detailed information about each checkpoint
* **Recovery Mechanisms**: Robust loading and fallback procedures

Training Best Practices
------------------------

**Hyperparameter Tuning:**

* Start with conservative learning rates (1e-4 to 1e-5)
* Use gradient clipping to prevent instability
* Monitor KL divergence to prevent policy collapse
* Adjust LoRA rank based on task complexity

**Data Management:**

* Maintain diverse training scenarios
* Balance positive and negative examples
* Regular data quality audits
* Efficient batch processing

**Performance Optimization:**

* Use mixed precision training when available
* Implement gradient accumulation for large batches
* Optimize data loading and preprocessing
* Monitor GPU/CPU utilization

**Monitoring and Debugging:**

* Set up comprehensive logging
* Use visualization tools for training curves
* Implement early stopping mechanisms
* Regular model validation on held-out data
