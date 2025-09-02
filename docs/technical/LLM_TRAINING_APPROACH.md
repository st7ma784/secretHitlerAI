# LLM Training Approach for Secret Hitler AI

## Overview

This document outlines the advanced training methodology for training Large Language Models (LLMs) to play Secret Hitler strategically using **LoRA (Low-Rank Adaptation)** and **RLHF (Reinforcement Learning from Human Feedback)** techniques.

## Why LoRA + RLHF for Secret Hitler?

### The Challenge
Secret Hitler is a complex social deduction game requiring:
- **Strategic reasoning** about hidden information
- **Deception and bluffing** capabilities
- **Social dynamics** understanding
- **Long-term planning** across multiple game phases
- **Role-specific behavior** (Liberal, Fascist, Hitler)

### Traditional Approaches vs. Our Solution

| Approach | Pros | Cons |
|----------|------|------|
| **Stochastic Gradient Descent** | Simple, well-understood | Poor sample efficiency, struggles with sparse rewards |
| **Standard Fine-tuning** | Direct optimization | Expensive, catastrophic forgetting, overfitting |
| **LoRA + RLHF** | Efficient, stable, human-aligned | More complex implementation |

## LoRA (Low-Rank Adaptation)

### What is LoRA?
LoRA is a parameter-efficient fine-tuning technique that:
- Freezes the original model weights
- Adds small trainable matrices (adapters) to attention layers
- Reduces trainable parameters by 99%+ while maintaining performance

### LoRA Configuration for Secret Hitler

```python
@dataclass
class LoRAConfig:
    rank: int = 16              # Adaptation complexity (8-64)
    alpha: int = 32             # Scaling factor (typically 2x rank)
    dropout: float = 0.1        # Regularization
    target_modules: List[str] = [
        "q_proj", "v_proj",     # Query/Value attention
        "k_proj", "o_proj",     # Key/Output attention  
        "gate_proj", "up_proj", # MLP layers
        "down_proj"
    ]
```

### Benefits for Secret Hitler:
1. **Memory Efficiency**: Train on consumer GPUs
2. **Fast Iteration**: Quick experiments with different strategies
3. **Modular Agents**: Different LoRA adapters for different roles
4. **Catastrophic Forgetting Prevention**: Base model knowledge preserved

## RLHF (Reinforcement Learning from Human Feedback)

### Why RLHF for Secret Hitler?

Traditional supervised learning fails because:
- **No ground truth**: Multiple valid strategies exist
- **Context-dependent**: Same action can be good/bad based on game state
- **Human preferences**: Strategic play requires human-like reasoning

### RLHF Pipeline

#### 1. Reward Model Training
```python
def calculate_reward(action: Dict, outcome: Dict) -> float:
    base_reward = 0.0
    player_role = action.get("player_role")
    winner = outcome.get("winner")
    
    # Role-based rewards
    if player_role == winner:
        base_reward = 1.0
    elif player_role == "hitler" and winner == "hitler":
        base_reward = 1.5  # Higher reward for Hitler wins
    else:
        base_reward = -0.5
    
    # Action-specific bonuses
    if action["type"] == "vote" and strategic_vote(action):
        base_reward += 0.1
    elif action["type"] == "policy_choice" and optimal_choice(action):
        base_reward += 0.2
    
    return np.clip(base_reward, -1.0, 2.0)
```

#### 2. Policy Optimization
- **PPO (Proximal Policy Optimization)** for stable training
- **KL divergence penalty** to prevent model collapse
- **Value function** for variance reduction

#### 3. Training Loop
```python
async def rlhf_training_step(self, training_examples):
    for batch in training_examples:
        # Generate actions with current policy
        actions = self.model.generate(batch.contexts)
        
        # Calculate rewards
        rewards = self.reward_model(actions, batch.outcomes)
        
        # Compute policy gradient
        policy_loss = self.compute_policy_loss(actions, rewards)
        
        # Update LoRA parameters only
        self.optimizer.step()
        
        # Track metrics
        self.log_metrics(policy_loss, rewards, kl_divergence)
```

## Training Architecture

### Model Selection
- **Base Model**: `microsoft/DialoGPT-medium`
- **Reasoning**: Conversational model pre-trained on dialogue
- **Size**: 345M parameters (manageable for fine-tuning)

### Special Tokens for Secret Hitler
```python
special_tokens = [
    "[LIBERAL]", "[FASCIST]", "[HITLER]",     # Roles
    "[VOTE_YES]", "[VOTE_NO]", "[NOMINATE]",  # Actions
    "[POLICY_LIBERAL]", "[POLICY_FASCIST]",   # Policies
    "[INVESTIGATE]", "[EXECUTE]", "[PEEK]"    # Special powers
]
```

### Multi-Agent Training Strategy

#### Agent Specialization
1. **Liberal Agents**
   - LoRA Rank: 16 (balanced complexity)
   - Focus: Truth-telling, coalition building
   - Reward: Liberal team wins

2. **Fascist Agents**
   - LoRA Rank: 32 (higher complexity for deception)
   - Focus: Deception, misdirection, Hitler protection
   - Reward: Fascist/Hitler wins

3. **Hitler Agent**
   - LoRA Rank: 64 (maximum complexity)
   - Focus: Staying hidden, opportunistic play
   - Reward: Hitler-specific win conditions

### Training Process

#### Phase 1: Supervised Fine-Tuning (SFT)
- Train on expert human games
- Learn basic game mechanics
- Establish baseline behavior

#### Phase 2: Reward Model Training
- Collect human preferences on game outcomes
- Train reward model to predict human judgments
- Validate on held-out preference data

#### Phase 3: RLHF Optimization
- Self-play training with reward model feedback
- Policy gradient optimization
- Continuous evaluation against human players

## Training Metrics & Monitoring

### Core Metrics
- **Training Loss**: Model prediction accuracy
- **Reward Score**: Average reward per episode
- **Policy Gradient Norm**: Training stability indicator
- **KL Divergence**: Distance from base model
- **Value Function Loss**: Critic accuracy

### Game-Specific Metrics
- **Win Rates by Role**: Liberal/Fascist/Hitler success rates
- **Game Length**: Average rounds per game
- **Action Confidence**: Model certainty in decisions
- **Strategic Coherence**: Consistency of role-playing

### WandB Integration
```python
# Log training metrics
wandb.log({
    "training/loss": training_loss,
    "training/reward_score": reward_score,
    "training/policy_gradient": policy_gradient_norm,
    "training/kl_divergence": kl_divergence,
    "game/liberal_win_rate": liberal_wins / total_games,
    "game/fascist_win_rate": fascist_wins / total_games,
    "game/hitler_win_rate": hitler_wins / total_games,
    "model/lora_rank": lora_config.rank,
    "model/learning_rate": current_lr
})
```

## Implementation Details

### Training Infrastructure
- **GPU Requirements**: RTX 3090 or better (24GB VRAM)
- **Training Time**: ~2-3 days for full training
- **Batch Size**: 8 (limited by memory)
- **Learning Rate**: 1e-4 with cosine decay

### Hyperparameter Optimization
- **LoRA Rank**: Grid search [8, 16, 32, 64]
- **Learning Rate**: [1e-5, 5e-5, 1e-4, 5e-4]
- **KL Penalty**: [0.01, 0.1, 0.5]
- **Temperature**: [0.7, 0.8, 0.9, 1.0]

### Evaluation Protocol
1. **Self-Play Tournaments**: Agents play against each other
2. **Human Evaluation**: Expert players rate agent performance
3. **Ablation Studies**: Compare different LoRA ranks and methods
4. **Generalization Tests**: Performance on unseen game variants

## Advanced Techniques

### Curriculum Learning
Start with simpler scenarios and gradually increase complexity:
1. **Phase 1**: 5-player games, basic roles only
2. **Phase 2**: 6-7 player games, add special powers
3. **Phase 3**: Full 8-10 player games, all mechanics

### Multi-Task Learning
Train on related tasks to improve generalization:
- **Mafia/Werewolf**: Similar social deduction mechanics
- **Resistance**: Team-based hidden role games
- **One Night Ultimate Werewolf**: Fast-paced deduction

### Constitutional AI
Add constitutional principles for ethical gameplay:
- No toxic behavior or harassment
- Respect for human players
- Fair play within game rules

## Results & Performance

### Expected Outcomes
- **Win Rate vs Random**: >80% across all roles
- **Win Rate vs Novice Humans**: >60%
- **Win Rate vs Expert Humans**: 40-50%
- **Training Efficiency**: 10x faster than full fine-tuning

### Deployment Strategy
1. **Gradual Rollout**: Start with easy difficulty bots
2. **A/B Testing**: Compare LoRA vs traditional approaches
3. **Continuous Learning**: Update models based on human games
4. **Multi-Model Ensemble**: Combine different LoRA adapters

## Future Improvements

### Technical Enhancements
- **Mixture of Experts**: Specialized sub-models for different game phases
- **Memory Networks**: Long-term memory of player behaviors
- **Graph Neural Networks**: Model player relationships explicitly
- **Transformer-XL**: Handle longer game contexts

### Training Innovations
- **Self-Supervised Learning**: Learn from game transcripts without labels
- **Adversarial Training**: Robust agents that handle edge cases
- **Meta-Learning**: Quickly adapt to new player styles
- **Federated Learning**: Learn from distributed human games

## Conclusion

The LoRA + RLHF approach provides an optimal balance of:
- **Efficiency**: Parameter-efficient training
- **Performance**: Human-level strategic play
- **Flexibility**: Adaptable to different roles and strategies
- **Scalability**: Can be extended to other social deduction games

This methodology represents the state-of-the-art in training LLMs for complex strategic games, combining the latest advances in parameter-efficient fine-tuning with human preference optimization.

---

*For implementation details, see `backend/training/llm_trainer.py`*
*For training interface, visit `/training` endpoint*
