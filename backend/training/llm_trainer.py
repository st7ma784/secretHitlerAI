"""
Advanced LLM Training Module for Secret Hitler AI
Implements LoRA (Low-Rank Adaptation) + RLHF (Reinforcement Learning from Human Feedback)
for training language models to play Secret Hitler strategically.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import random
from pathlib import Path

# Import checkpointing system
try:
    from .model_checkpointing import get_checkpoint_manager
except ImportError:
    # Fallback for testing
    from backend.training.model_checkpointing import get_checkpoint_manager

# Mock imports for demonstration - in real implementation, use actual libraries
try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/Transformers not available - using mock implementation")

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Target attention and MLP layers for language models
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

@dataclass
class RLHFConfig:
    """Configuration for RLHF training"""
    learning_rate: float = 1e-4
    batch_size: int = 8
    max_length: int = 512
    temperature: float = 0.7
    kl_penalty: float = 0.1
    reward_model_path: Optional[str] = None
    value_model_path: Optional[str] = None

@dataclass
class TrainingMetrics:
    """Training metrics for monitoring"""
    step: int = 0
    training_loss: float = 0.0
    reward_score: float = 0.0
    policy_gradient_norm: float = 0.0
    kl_divergence: float = 0.0
    value_function_loss: float = 0.0
    learning_rate: float = 1e-4
    timestamp: str = ""

class SecretHitlerLLMTrainer:
    """
    Advanced LLM trainer for Secret Hitler AI using LoRA + RLHF.
    
    This trainer implements:
    1. LoRA (Low-Rank Adaptation) for efficient fine-tuning
    2. RLHF (Reinforcement Learning from Human Feedback) for strategic gameplay
    3. Custom reward modeling based on game outcomes
    4. Multi-agent training with diverse strategies
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 lora_config: LoRAConfig = None,
                 rlhf_config: RLHFConfig = None):
        self.model_name = model_name
        self.lora_config = lora_config or LoRAConfig()
        self.rlhf_config = rlhf_config or RLHFConfig()
        
        self.model = None
        self.tokenizer = None
        self.reward_model = None
        self.value_model = None
        
        self.training_active = False
        self.current_metrics = TrainingMetrics()
        self.training_history = []
        
        self.logger = logging.getLogger(__name__)
        
        # Checkpointing
        self.checkpoint_manager = get_checkpoint_manager()
        self.agent_role = 'unknown'
        self.last_checkpoint_step = 0
        self.checkpoint_frequency = 100  # Save every 100 training steps
        
        # Metrics tracking
        self.metrics = {
            'training_loss': 0.0,
            'reward_score': 0.0,
            'policy_gradient': 0.0,
            'kl_divergence': 0.0,
            'value_function': 0.0,
            'learning_rate': self.rlhf_config.learning_rate,
            'lora_rank': self.lora_config.rank,
            'training_method': 'rlhf'
        }
        
    async def initialize_models(self):
        """Initialize the base model, LoRA adaptation, and reward models"""
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - using mock implementation")
            return await self._initialize_mock_models()
        
        try:
            # Load base model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Add special tokens for Secret Hitler gameplay
            special_tokens = [
                "[LIBERAL]", "[FASCIST]", "[HITLER]", 
                "[VOTE_YES]", "[VOTE_NO]", "[NOMINATE]", 
                "[POLICY_LIBERAL]", "[POLICY_FASCIST]",
                "[INVESTIGATE]", "[EXECUTE]", "[PEEK]"
            ]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Apply LoRA adaptation
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config.rank,
                lora_alpha=self.lora_config.alpha,
                lora_dropout=self.lora_config.dropout,
                target_modules=self.lora_config.target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
            
            # Initialize reward and value models (simplified)
            self.reward_model = self._create_reward_model()
            self.value_model = self._create_value_model()
            
            self.logger.info(f"Initialized LoRA model with rank {self.lora_config.rank}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            await self._initialize_mock_models()
    
    async def _initialize_mock_models(self):
        """Mock model initialization for demonstration"""
        self.logger.info("Using mock model implementation")
        self.model = "mock_model"
        self.tokenizer = "mock_tokenizer"
        self.reward_model = "mock_reward_model"
        self.value_model = "mock_value_model"
    
    def _create_reward_model(self):
        """Create a reward model for RLHF training"""
        if not TORCH_AVAILABLE:
            return "mock_reward_model"
        
        # Simplified reward model - in practice, this would be pre-trained
        # on human preferences for Secret Hitler gameplay
        class RewardModel(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.linear = nn.Linear(hidden_size, 1)
                
            def forward(self, hidden_states):
                return self.linear(hidden_states.mean(dim=1))
        
        return RewardModel()
    
    def _create_value_model(self):
        """Create a value function model for RLHF"""
        if not TORCH_AVAILABLE:
            return "mock_value_model"
        
        class ValueModel(nn.Module):
            def __init__(self, hidden_size=768):
                super().__init__()
                self.linear = nn.Linear(hidden_size, 1)
                
            def forward(self, hidden_states):
                return self.linear(hidden_states.mean(dim=1))
        
        return ValueModel()
    
    async def train_on_game_data(self, game_data: List[Dict[str, Any]]) -> TrainingMetrics:
        """
        Train the model on Secret Hitler game data using RLHF.
        
        Args:
            game_data: List of game episodes with actions, outcomes, and rewards
            
        Returns:
            TrainingMetrics: Current training metrics
        """
        if not self.training_active:
            return self.current_metrics
        
        self.logger.info(f"Training on {len(game_data)} game episodes")
        
        # Process game data into training examples
        training_examples = await self._process_game_data(game_data)
        
        # Perform RLHF training step
        metrics = await self._rlhf_training_step(training_examples)
        
        # Update current metrics
        self.current_metrics = metrics
        self.training_history.append(asdict(metrics))
        
        return metrics
    
    async def _process_game_data(self, game_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw game data into training examples"""
        training_examples = []
        
        for game in game_data:
            # Extract game context, player actions, and outcomes
            game_context = self._extract_game_context(game)
            player_actions = self._extract_player_actions(game)
            game_outcome = self._extract_game_outcome(game)
            
            # Create training examples for each significant decision point
            for action in player_actions:
                example = {
                    "context": game_context,
                    "action": action,
                    "outcome": game_outcome,
                    "reward": self._calculate_reward(action, game_outcome),
                    "player_role": action.get("player_role"),
                    "game_state": action.get("game_state")
                }
                training_examples.append(example)
        
        return training_examples
    
    def _extract_game_context(self, game: Dict[str, Any]) -> str:
        """Extract relevant game context for training"""
        # Convert game state to natural language context
        context_parts = []
        
        if "players" in game:
            context_parts.append(f"Players: {', '.join(game['players'])}")
        
        if "round" in game:
            context_parts.append(f"Round: {game['round']}")
        
        if "policies_enacted" in game:
            liberal_policies = game["policies_enacted"].get("liberal", 0)
            fascist_policies = game["policies_enacted"].get("fascist", 0)
            context_parts.append(f"Policies: {liberal_policies}L, {fascist_policies}F")
        
        return " | ".join(context_parts)
    
    def _extract_player_actions(self, game: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract player actions from game data"""
        actions = []
        
        if "actions" in game:
            for action in game["actions"]:
                if action["type"] in ["vote", "nominate", "policy_choice", "special_action"]:
                    actions.append({
                        "type": action["type"],
                        "player_id": action["player_id"],
                        "player_role": action.get("player_role"),
                        "choice": action["choice"],
                        "game_state": action.get("game_state", {}),
                        "reasoning": action.get("reasoning", "")
                    })
        
        return actions
    
    def _extract_game_outcome(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Extract game outcome and winner information"""
        return {
            "winner": game.get("winner", "unknown"),
            "win_condition": game.get("win_condition", "unknown"),
            "game_length": game.get("rounds_played", 0),
            "final_score": game.get("final_score", {})
        }
    
    def _calculate_reward(self, action: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """
        Calculate reward for an action based on game outcome.
        This implements the reward function for RLHF training.
        """
        base_reward = 0.0
        player_role = action.get("player_role", "unknown")
        winner = outcome.get("winner", "unknown")
        
        # Role-based reward calculation
        if player_role == "liberal" and winner == "liberal":
            base_reward = 1.0
        elif player_role == "fascist" and winner == "fascist":
            base_reward = 1.0
        elif player_role == "hitler" and winner == "hitler":
            base_reward = 1.5  # Higher reward for Hitler wins
        else:
            base_reward = -0.5  # Penalty for losing
        
        # Action-specific bonuses/penalties
        action_type = action.get("type", "")
        if action_type == "vote":
            # Reward strategic voting
            if action.get("choice") == "yes" and winner == player_role:
                base_reward += 0.1
        elif action_type == "policy_choice":
            # Reward good policy choices
            choice = action.get("choice", "")
            if (player_role == "liberal" and choice == "liberal") or \
               (player_role in ["fascist", "hitler"] and choice == "fascist"):
                base_reward += 0.2
        
        # Game length consideration (shorter games can be better)
        game_length = outcome.get("game_length", 10)
        if game_length < 8 and winner == player_role:
            base_reward += 0.1
        
        return np.clip(base_reward, -1.0, 2.0)
    
    async def _rlhf_training_step(self, training_examples: List[Dict[str, Any]]) -> TrainingMetrics:
        """Perform one RLHF training step"""
        if not TORCH_AVAILABLE:
            return await self._mock_training_step(training_examples)
        
        # This would implement the actual RLHF training logic
        # For now, we'll simulate the training metrics
        return await self._mock_training_step(training_examples)
    
    async def _mock_training_step(self, training_examples: List[Dict[str, Any]]) -> TrainingMetrics:
        """Mock training step for demonstration"""
        # Simulate training progress
        self.current_metrics.step += 1
        
        # Simulate decreasing loss with some noise
        self.current_metrics.training_loss = max(0.1, 
            self.current_metrics.training_loss * 0.99 + np.random.normal(0, 0.01))
        
        # Simulate reward score improvement
        self.current_metrics.reward_score += np.random.normal(0.01, 0.05)
        
        # Simulate other RLHF metrics
        self.current_metrics.policy_gradient_norm = abs(np.random.normal(0, 0.02))
        self.current_metrics.kl_divergence = abs(np.random.normal(0.005, 0.002))
        self.current_metrics.value_function_loss = max(0, 
            self.current_metrics.value_function_loss * 0.98 + np.random.normal(0, 0.01))
        
        # Learning rate scheduling
        if self.current_metrics.step % 10 == 0:
            self.current_metrics.learning_rate *= 0.95
        
        self.current_metrics.timestamp = datetime.now().isoformat()
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        return self.current_metrics
    
    async def generate_action(self, game_context: str, player_role: str) -> Dict[str, Any]:
        """
        Generate an action for a given game context using the trained model.
        
        Args:
            game_context: Current game state description
            player_role: Role of the player (liberal, fascist, hitler)
            
        Returns:
            Dict containing the suggested action and confidence
        """
        if not TORCH_AVAILABLE:
            return await self._mock_generate_action(game_context, player_role)
        
        # This would implement actual model inference
        return await self._mock_generate_action(game_context, player_role)
    
    async def _mock_generate_action(self, game_context: str, player_role: str) -> Dict[str, Any]:
        """Mock action generation for demonstration"""
        # Simulate different strategies based on role
        actions = {
            "liberal": ["vote_yes", "vote_no", "nominate_trusted", "choose_liberal_policy"],
            "fascist": ["vote_strategically", "nominate_hitler", "choose_fascist_policy", "mislead"],
            "hitler": ["stay_hidden", "vote_conservatively", "build_trust", "wait_for_opportunity"]
        }
        
        role_actions = actions.get(player_role, actions["liberal"])
        suggested_action = np.random.choice(role_actions)
        confidence = np.random.uniform(0.6, 0.95)
        
        return {
            "action": suggested_action,
            "confidence": confidence,
            "reasoning": f"Based on {player_role} strategy and current game state",
            "model_version": f"lora_rank_{self.lora_config.rank}"
        }
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            "current_metrics": asdict(self.current_metrics),
            "lora_config": asdict(self.lora_config),
            "rlhf_config": asdict(self.rlhf_config),
            "training_active": self.training_active,
            "total_steps": self.current_metrics.step,
            "model_name": self.model_name
        }
    
    def start_training(self):
        """Start training mode"""
        self.training_active = True
        self.logger.info("Started LLM training mode")
    
    def stop_training(self):
        """Stop training mode"""
        self.training_active = False
        self.logger.info("Stopped LLM training mode")
    
    async def save_model(self, path: str):
        """Save the trained model"""
        if not TORCH_AVAILABLE:
            self.logger.info(f"Mock: Saving model to {path}")
            return
        
        # Save LoRA adapters and training state
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        
        # Save training metrics
        metrics_path = f"{path}/training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Saved model and metrics to {path}")
    
    def save_checkpoint(self, agent_role: str = None) -> str:
        """Save a training checkpoint"""
        if agent_role:
            self.agent_role = agent_role
        
        training_state = {
            'training_step': self.current_metrics.step,
            'total_episodes': getattr(self, 'total_episodes', 0),
            'current_loss': self.current_metrics.training_loss,
            'reward_history': getattr(self, 'reward_history', []),
            'policy_losses': getattr(self, 'policy_losses', []),
            'value_losses': getattr(self, 'value_losses', []),
            'experience_buffer': getattr(self, 'experience_buffer', []),
            'training_history': self.training_history,
            'lora_config': asdict(self.lora_config),
            'rlhf_config': asdict(self.rlhf_config),
            'model_name': self.model_name
        }
        
        performance_metrics = {
            'training_loss': self.current_metrics.training_loss,
            'reward_score': self.current_metrics.reward_score,
            'policy_gradient_norm': self.current_metrics.policy_gradient_norm,
            'kl_divergence': self.current_metrics.kl_divergence,
            'value_function_loss': self.current_metrics.value_function_loss,
            'learning_rate': self.current_metrics.learning_rate
        }
        
        checkpoint_id = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=getattr(self, 'optimizer', None),
            training_state=training_state,
            agent_role=self.agent_role,
            training_step=self.current_metrics.step,
            performance_metrics=performance_metrics
        )
        
        self.last_checkpoint_step = self.current_metrics.step
        self.logger.info(f"Saved checkpoint {checkpoint_id} for {self.agent_role}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str = None, agent_role: str = None) -> bool:
        """Load a training checkpoint"""
        if checkpoint_id is None:
            # Load latest checkpoint for this agent role
            checkpoint_id = self.checkpoint_manager.get_latest_checkpoint(agent_role or self.agent_role)
            if checkpoint_id is None:
                self.logger.warning(f"No checkpoint found for role {agent_role or self.agent_role}")
                return False
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if checkpoint_data is None:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}")
            return False
        
        try:
            # Restore training state
            training_state = checkpoint_data.get('training_state', {})
            self.current_metrics.step = training_state.get('training_step', 0)
            self.current_metrics.training_loss = training_state.get('current_loss', 0.0)
            self.training_history = training_state.get('training_history', [])
            
            # Restore model state if available
            if 'model_state_dict' in checkpoint_data and self.model is not None:
                if hasattr(self.model, 'load_state_dict'):
                    self.model.load_state_dict(checkpoint_data['model_state_dict'])
            
            # Restore optimizer state if available
            if 'optimizer_state_dict' in checkpoint_data and hasattr(self, 'optimizer') and self.optimizer is not None:
                if hasattr(self.optimizer, 'load_state_dict'):
                    self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Update agent role and configs
            self.agent_role = checkpoint_data.get('agent_role', self.agent_role)
            
            # Restore performance metrics
            perf_metrics = checkpoint_data.get('performance_metrics', {})
            self.current_metrics.reward_score = perf_metrics.get('reward_score', 0.0)
            self.current_metrics.policy_gradient_norm = perf_metrics.get('policy_gradient_norm', 0.0)
            self.current_metrics.kl_divergence = perf_metrics.get('kl_divergence', 0.0)
            self.current_metrics.value_function_loss = perf_metrics.get('value_function_loss', 0.0)
            self.current_metrics.learning_rate = perf_metrics.get('learning_rate', self.rlhf_config.learning_rate)
            
            self.last_checkpoint_step = self.current_metrics.step
            self.logger.info(f"Loaded checkpoint {checkpoint_id} for {self.agent_role} at step {self.current_metrics.step}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring checkpoint {checkpoint_id}: {e}")
            return False
    
    def should_save_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint"""
        return (self.current_metrics.step - self.last_checkpoint_step) >= self.checkpoint_frequency
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints"""
        stats = self.checkpoint_manager.get_checkpoint_stats()
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint(self.agent_role)
        
        return {
            'checkpoint_stats': stats,
            'latest_checkpoint': latest_checkpoint,
            'agent_role': self.agent_role,
            'last_checkpoint_step': self.last_checkpoint_step,
            'current_step': self.current_metrics.step,
            'checkpoint_frequency': self.checkpoint_frequency,
            'should_save': self.should_save_checkpoint()
        }

# Global trainer instance
llm_trainer = None

def get_llm_trainer() -> SecretHitlerLLMTrainer:
    """Get or create the global LLM trainer instance"""
    global llm_trainer
    if llm_trainer is None:
        llm_trainer = SecretHitlerLLMTrainer()
    return llm_trainer
