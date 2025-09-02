import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, namedtuple
import random
import math
from dataclasses import dataclass, field
import json
from datetime import datetime

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'action_type', 'valid_actions', 'reasoning'
])

@dataclass
class ARBOConfig:
    """Configuration for ARBO (Advantage-based Rational Bayesian Optimization)"""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    
    # ARBO specific parameters
    bayesian_samples: int = 5  # Number of Bayesian samples
    uncertainty_weight: float = 0.1  # Weight for uncertainty in action selection
    rational_temperature: float = 1.0  # Temperature for rational decision making
    advantage_threshold: float = 0.1  # Threshold for advantage-based action selection
    
    # Replay buffer
    buffer_size: int = 10000
    batch_size: int = 32
    min_buffer_size: int = 1000
    
    # Self-training parameters
    self_play_games: int = 100
    training_frequency: int = 10  # Train every N games
    save_frequency: int = 50  # Save model every N games

class ARBOReplayBuffer:
    """Experience replay buffer for ARBO learning"""
    
    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def size(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

class BayesianLayer(nn.Module):
    """Bayesian neural network layer for uncertainty estimation"""
    
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(output_size, input_size))
        self.weight_logvar = nn.Parameter(torch.Tensor(output_size, input_size))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(output_size))
        self.bias_logvar = nn.Parameter(torch.Tensor(output_size))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize parameters
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -3)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -3)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # Sample weights and biases
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + torch.exp(0.5 * self.weight_logvar) * weight_eps
            bias = self.bias_mu + torch.exp(0.5 * self.bias_logvar) * bias_eps
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for regularization"""
        kl_weight = 0.5 * torch.sum(
            torch.exp(self.weight_logvar) + self.weight_mu.pow(2) - 1 - self.weight_logvar
        )
        kl_bias = 0.5 * torch.sum(
            torch.exp(self.bias_logvar) + self.bias_mu.pow(2) - 1 - self.bias_logvar
        )
        return kl_weight + kl_bias

class ARBONetwork(nn.Module):
    """ARBO neural network with Bayesian uncertainty and rational decision making"""
    
    def __init__(self, input_size: int, hidden_size: int, config: ARBOConfig):
        super().__init__()
        self.config = config
        
        # Bayesian layers for uncertainty estimation
        self.bayesian_layer1 = BayesianLayer(input_size, hidden_size)
        self.bayesian_layer2 = BayesianLayer(hidden_size, hidden_size)
        
        # Regular layers
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        
        # Output heads
        self.action_head = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        # Bayesian layers with uncertainty
        x = F.relu(self.bayesian_layer1(x, sample))
        x = self.dropout(x)
        x = F.relu(self.bayesian_layer2(x, sample))
        x = self.dropout(x)
        
        # Regular layers
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        
        # Output heads
        action_features = self.action_head(x)
        value = self.value_head(x)
        uncertainty = torch.sigmoid(self.uncertainty_head(x))
        
        return {
            'action_features': action_features,
            'value': value,
            'uncertainty': uncertainty
        }
    
    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for Bayesian regularization"""
        return self.bayesian_layer1.kl_divergence() + self.bayesian_layer2.kl_divergence()

class ARBOAgent:
    """ARBO (Advantage-based Rational Bayesian Optimization) Agent"""
    
    def __init__(self, model, config: ARBOConfig, agent_id: str):
        self.model = model
        self.config = config
        self.agent_id = agent_id
        
        # ARBO network for advanced decision making
        self.arbo_network = ARBONetwork(
            input_size=config.max_grad_norm,  # Will be set based on model hidden size
            hidden_size=512,
            config=config
        )
        
        # Optimizers
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.arbo_optimizer = torch.optim.Adam(self.arbo_network.parameters(), lr=config.learning_rate)
        
        # Experience replay
        self.replay_buffer = ARBOReplayBuffer(config.buffer_size)
        
        # Training statistics
        self.games_played = 0
        self.total_reward = 0
        self.win_rate = 0
        self.training_history = []
        
        # Goal-specific rewards (Liberal vs Fascist objectives)
        self.goal_weights = {
            'liberal': {'liberal_policy': 1.0, 'fascist_policy': -1.0, 'hitler_chancellor': -2.0, 'hitler_dead': 2.0},
            'fascist': {'liberal_policy': -1.0, 'fascist_policy': 1.0, 'hitler_chancellor': 2.0, 'hitler_dead': -2.0},
            'hitler': {'liberal_policy': -1.0, 'fascist_policy': 1.0, 'hitler_chancellor': 3.0, 'hitler_dead': -3.0}
        }
    
    def encode_state_features(self, game_state: Dict) -> torch.Tensor:
        """Convert game state to feature vector for ARBO network"""
        features = []
        
        # Basic game features
        features.extend([
            game_state.get('liberal_policies', 0) / 5.0,
            game_state.get('fascist_policies', 0) / 6.0,
            game_state.get('election_tracker', 0) / 3.0,
        ])
        
        # Phase encoding (one-hot)
        phases = ['lobby', 'election', 'legislative', 'executive', 'game_over']
        phase = game_state.get('phase', 'lobby')
        phase_encoding = [1.0 if phase == p else 0.0 for p in phases]
        features.extend(phase_encoding)
        
        # Player features
        players = game_state.get('players', [])
        max_players = 10
        
        # Pad or truncate to fixed size
        for i in range(max_players):
            if i < len(players):
                player = players[i]
                features.extend([
                    1.0 if player.get('is_alive') else 0.0,
                    1.0 if player.get('is_president') else 0.0,
                    1.0 if player.get('is_chancellor') else 0.0,
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
        
        # Role-specific features
        role = game_state.get('your_role', 'liberal')
        role_encoding = {
            'liberal': [1.0, 0.0, 0.0],
            'fascist': [0.0, 1.0, 0.0],
            'hitler': [0.0, 0.0, 1.0]
        }
        features.extend(role_encoding.get(role, [1.0, 0.0, 0.0]))
        
        # Team knowledge features
        fascist_team = game_state.get('fascist_team', [])
        features.append(len(fascist_team) / max_players)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def select_action_with_arbo(
        self, 
        game_state: Dict, 
        action_type: str,
        valid_targets: Optional[List[str]] = None
    ) -> Tuple[Any, float, str, Dict]:
        """
        Select action using ARBO (Advantage-based Rational Bayesian Optimization)
        
        Returns:
            action: Selected action
            confidence: Confidence score
            reasoning: Explanation
            metrics: Additional metrics for training
        """
        
        # Get base action from LAMA model
        base_action, base_confidence, reasoning = self.model.predict_action(
            game_state, self.agent_id, action_type, valid_targets
        )
        
        # Encode state for ARBO network
        state_features = self.encode_state_features(game_state)
        
        # Get multiple Bayesian samples for uncertainty estimation
        uncertainties = []
        action_values = []
        
        for _ in range(self.config.bayesian_samples):
            arbo_output = self.arbo_network(state_features.unsqueeze(0), sample=True)
            action_values.append(arbo_output['value'].item())
            uncertainties.append(arbo_output['uncertainty'].item())
        
        # Calculate uncertainty metrics
        value_mean = np.mean(action_values)
        value_std = np.std(action_values)
        uncertainty_mean = np.mean(uncertainties)
        
        # Rational decision making: consider uncertainty in action selection
        exploration_bonus = self.config.uncertainty_weight * uncertainty_mean
        adjusted_confidence = base_confidence + exploration_bonus
        
        # Advantage-based selection: only override if advantage is significant
        state_value = self.model.get_state_value(game_state, self.agent_id)
        advantage = value_mean - state_value
        
        final_action = base_action
        final_confidence = adjusted_confidence
        
        # Override action if ARBO suggests significant advantage
        if abs(advantage) > self.config.advantage_threshold:
            # Use rational temperature for exploration vs exploitation
            rationality = math.exp(advantage / self.config.rational_temperature)
            if random.random() < rationality:
                # Potentially select different action based on ARBO analysis
                # For now, we adjust confidence and keep the base action
                final_confidence = min(1.0, adjusted_confidence * (1 + advantage))
        
        # Enhanced reasoning with ARBO insights
        enhanced_reasoning = f"{reasoning} [ARBO: value={value_mean:.3f}±{value_std:.3f}, uncertainty={uncertainty_mean:.3f}, advantage={advantage:.3f}]"
        
        metrics = {
            'base_confidence': base_confidence,
            'arbo_value_mean': value_mean,
            'arbo_value_std': value_std,
            'uncertainty': uncertainty_mean,
            'advantage': advantage,
            'exploration_bonus': exploration_bonus,
            'final_confidence': final_confidence
        }
        
        return final_action, final_confidence, enhanced_reasoning, metrics
    
    def calculate_reward(self, old_state: Dict, new_state: Dict, action_type: str, action: Any) -> float:
        """Calculate reward based on game state changes and agent's goals"""
        
        role = old_state.get('your_role', 'liberal')
        weights = self.goal_weights.get(role, self.goal_weights['liberal'])
        
        reward = 0.0
        
        # Policy rewards
        old_liberal = old_state.get('liberal_policies', 0)
        new_liberal = new_state.get('liberal_policies', 0)
        old_fascist = old_state.get('fascist_policies', 0)
        new_fascist = new_state.get('fascist_policies', 0)
        
        if new_liberal > old_liberal:
            reward += weights['liberal_policy']
        if new_fascist > old_fascist:
            reward += weights['fascist_policy']
        
        # Hitler-specific rewards
        old_hitler_chancellor = any(
            p.get('is_chancellor') and p.get('role') == 'hitler' 
            for p in old_state.get('players', [])
        )
        new_hitler_chancellor = any(
            p.get('is_chancellor') and p.get('role') == 'hitler' 
            for p in new_state.get('players', [])
        )
        
        if new_hitler_chancellor and not old_hitler_chancellor:
            reward += weights['hitler_chancellor']
        
        # Game ending rewards
        winner = new_state.get('winner')
        if winner:
            if (role in ['liberal'] and winner == 'liberal') or \
               (role in ['fascist', 'hitler'] and winner == 'fascist'):
                reward += 5.0  # Big reward for winning
            else:
                reward -= 5.0  # Big penalty for losing
        
        # Action-specific rewards
        if action_type == 'vote':
            # Reward for consistent voting with team goals
            if role == 'liberal':
                # Liberals generally want to vote nein unless they trust the government
                if action == 'nein':
                    reward += 0.1
            else:  # Fascist or Hitler
                # Fascists might vote strategically
                if new_fascist > old_fascist:  # If a fascist policy was enacted
                    reward += 0.2
        
        return reward
    
    def store_experience(
        self, 
        state: Dict, 
        action: Any, 
        reward: float, 
        next_state: Dict,
        done: bool,
        action_type: str,
        valid_actions: List[str],
        reasoning: str
    ):
        """Store experience in replay buffer"""
        
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            action_type=action_type,
            valid_actions=valid_actions,
            reasoning=reasoning
        )
        
        self.replay_buffer.add(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Single training step using stored experiences"""
        
        if self.replay_buffer.size() < self.config.min_buffer_size:
            return {}
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # Prepare tensors
        states = [self.encode_state_features(exp.state) for exp in batch]
        next_states = [self.encode_state_features(exp.next_state) for exp in batch]
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32)
        dones = torch.tensor([exp.done for exp in batch], dtype=torch.float32)
        
        state_batch = torch.stack(states)
        next_state_batch = torch.stack(next_states)
        
        # Forward pass
        current_outputs = self.arbo_network(state_batch)
        next_outputs = self.arbo_network(next_state_batch, sample=False)
        
        current_values = current_outputs['value'].squeeze()
        next_values = next_outputs['value'].squeeze()
        
        # Compute targets using GAE
        targets = rewards + self.config.gamma * next_values * (1 - dones)
        advantages = targets - current_values
        
        # Value loss
        value_loss = F.mse_loss(current_values, targets.detach())
        
        # Bayesian regularization
        kl_loss = self.arbo_network.kl_divergence() / len(batch)
        
        # Total loss
        total_loss = (
            self.config.value_coefficient * value_loss +
            0.001 * kl_loss  # Small weight for KL regularization
        )
        
        # Backward pass
        self.arbo_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.arbo_network.parameters(), self.config.max_grad_norm)
        self.arbo_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantages.mean().item(),
            'mean_value': current_values.mean().item()
        }
    
    def self_train(self, num_games: int = None) -> Dict[str, Any]:
        """Self-training through gameplay simulation"""
        
        if num_games is None:
            num_games = self.config.self_play_games
        
        training_results = {
            'games_trained': 0,
            'total_reward': 0,
            'win_rate': 0,
            'training_losses': []
        }
        
        wins = 0
        total_reward = 0
        
        for game in range(num_games):
            # Simulate a game (this would integrate with the actual game loop)
            game_reward, won = self._simulate_game()
            
            total_reward += game_reward
            if won:
                wins += 1
            
            # Train periodically
            if (game + 1) % self.config.training_frequency == 0:
                loss_info = self.train_step()
                if loss_info:
                    training_results['training_losses'].append(loss_info)
            
            # Save periodically
            if (game + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f"checkpoint_game_{game + 1}")
        
        # Update statistics
        self.games_played += num_games
        self.total_reward += total_reward
        self.win_rate = wins / num_games if num_games > 0 else 0
        
        training_results.update({
            'games_trained': num_games,
            'total_reward': total_reward,
            'win_rate': self.win_rate,
            'average_reward': total_reward / num_games if num_games > 0 else 0
        })
        
        return training_results
    
    def _simulate_game(self) -> Tuple[float, bool]:
        """Simulate a single game for self-training"""
        # This is a placeholder - in practice, this would run actual game simulations
        # For now, return random results
        reward = random.uniform(-5, 5)
        won = random.choice([True, False])
        return reward, won
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'arbo_network_state_dict': self.arbo_network.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'arbo_optimizer_state_dict': self.arbo_optimizer.state_dict(),
            'config': self.config,
            'games_played': self.games_played,
            'total_reward': self.total_reward,
            'win_rate': self.win_rate,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, f"models/{filename}.pth")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(f"models/{filename}.pth")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.arbo_network.load_state_dict(checkpoint['arbo_network_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.arbo_optimizer.load_state_dict(checkpoint['arbo_optimizer_state_dict'])
        
        self.games_played = checkpoint.get('games_played', 0)
        self.total_reward = checkpoint.get('total_reward', 0)
        self.win_rate = checkpoint.get('win_rate', 0)
        self.training_history = checkpoint.get('training_history', [])