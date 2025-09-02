import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from datetime import datetime
import random
from threading import Thread, Event
import queue

from .lama import LamaModel, LamaModelManager, ModelConfig
from .arbo import ARBOAgent, ARBOConfig
from ..training.secret_hitler_env import SecretHitlerEnv
from ..game.rules import VoteType

logger = logging.getLogger(__name__)

class SecretHitlerAI:
    """
    AI Agent that can play Secret Hitler using LAMA + ARBO
    Supports self-training during gameplay
    """
    
    def __init__(
        self, 
        agent_id: str,
        model_config: ModelConfig = None,
        arbo_config: ARBOConfig = None,
        enable_self_training: bool = True
    ):
        self.agent_id = agent_id
        self.enable_self_training = enable_self_training
        
        # Initialize configurations
        self.model_config = model_config or ModelConfig()
        self.arbo_config = arbo_config or ARBOConfig()
        
        # Initialize models
        self.model_manager = LamaModelManager(self.model_config)
        self.lama_model = self.model_manager.create_agent_model(agent_id)
        
        # Initialize ARBO agent
        self.arbo_agent = ARBOAgent(self.lama_model, self.arbo_config, agent_id)
        
        # Training environment for self-play
        self.training_env = None
        
        # Game state tracking
        self.current_game_id = None
        self.player_id = None
        self.game_history = []
        self.last_action = None
        self.last_state = None
        
        # Training thread management
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.stop_training = Event()
        
        # Performance metrics
        self.performance_metrics = {
            'games_played': 0,
            'games_won': 0,
            'total_actions': 0,
            'successful_actions': 0,
            'average_confidence': 0.0,
            'win_rate_by_role': {'liberal': 0.0, 'fascist': 0.0, 'hitler': 0.0},
            'action_success_rate': 0.0,
            'training_sessions': 0,
            'last_training_time': None
        }
        
        if enable_self_training:
            self._start_training_thread()
    
    def _start_training_thread(self):
        """Start background training thread"""
        self.training_thread = Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        logger.info(f"Started training thread for agent {self.agent_id}")
    
    def _training_loop(self):
        """Background training loop"""
        while not self.stop_training.is_set():
            try:
                # Wait for training trigger or timeout
                training_signal = self.training_queue.get(timeout=60)
                
                if training_signal == "STOP":
                    break
                
                # Perform self-training
                self._perform_self_training()
                
            except queue.Empty:
                # Periodic training even without trigger
                if self.enable_self_training:
                    self._perform_self_training()
            except Exception as e:
                logger.error(f"Error in training loop for agent {self.agent_id}: {e}")
    
    def _perform_self_training(self):
        """Perform a self-training session"""
        try:
            logger.info(f"Starting self-training session for agent {self.agent_id}")
            
            # Create training environment if needed
            if self.training_env is None:
                self.training_env = SecretHitlerEnv(
                    num_players=7,
                    ai_players=[self.agent_id, "opponent1", "opponent2"]
                )
            
            # Run training games
            training_results = self.arbo_agent.self_train(num_games=10)
            
            # Update performance metrics
            self.performance_metrics['training_sessions'] += 1
            self.performance_metrics['last_training_time'] = datetime.now().isoformat()
            
            logger.info(f"Completed self-training for {self.agent_id}: {training_results}")
            
        except Exception as e:
            logger.error(f"Error during self-training for agent {self.agent_id}: {e}")
    
    async def join_game(self, game_id: str, player_id: str) -> bool:
        """Join a game as an AI player"""
        try:
            self.current_game_id = game_id
            self.player_id = player_id
            self.game_history = []
            self.last_action = None
            self.last_state = None
            
            logger.info(f"AI agent {self.agent_id} joined game {game_id} as player {player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error joining game for agent {self.agent_id}: {e}")
            return False
    
    async def make_decision(
        self, 
        game_state: Dict, 
        action_type: str,
        valid_targets: Optional[List[str]] = None
    ) -> Tuple[Any, float, str]:
        """
        Make a decision for the current game state
        
        Args:
            game_state: Current game state from player's perspective
            action_type: Type of action needed ('nominate', 'vote', 'policy')
            valid_targets: List of valid target IDs (for nominations, etc.)
        
        Returns:
            action: The chosen action
            confidence: Confidence score (0-1)
            reasoning: Text explanation of the decision
        """
        
        try:
            # Store current state for learning
            self.last_state = game_state.copy()
            
            # Use ARBO for enhanced decision making
            action, confidence, reasoning, metrics = self.arbo_agent.select_action_with_arbo(
                game_state, action_type, valid_targets
            )
            
            self.last_action = {
                'action': action,
                'action_type': action_type,
                'confidence': confidence,
                'reasoning': reasoning,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update performance metrics
            self.performance_metrics['total_actions'] += 1
            
            # Update average confidence
            total_actions = self.performance_metrics['total_actions']
            current_avg = self.performance_metrics['average_confidence']
            self.performance_metrics['average_confidence'] = (
                (current_avg * (total_actions - 1) + confidence) / total_actions
            )
            
            logger.info(f"AI {self.agent_id} decision: {action_type}={action} (confidence={confidence:.3f})")
            logger.debug(f"Reasoning: {reasoning}")
            
            return action, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error making decision for agent {self.agent_id}: {e}")
            return None, 0.0, f"Error: {str(e)}"
    
    async def receive_feedback(self, new_game_state: Dict, action_success: bool):
        """
        Receive feedback about the last action and learn from it
        
        Args:
            new_game_state: Game state after the action
            action_success: Whether the action was successful
        """
        
        try:
            if self.last_action and self.last_state:
                # Update success rate
                if action_success:
                    self.performance_metrics['successful_actions'] += 1
                
                self.performance_metrics['action_success_rate'] = (
                    self.performance_metrics['successful_actions'] / 
                    self.performance_metrics['total_actions']
                )
                
                # Calculate reward for learning
                reward = self.arbo_agent.calculate_reward(
                    self.last_state, 
                    new_game_state,
                    self.last_action['action_type'],
                    self.last_action['action']
                )
                
                # Store experience for training
                if self.enable_self_training:
                    self.arbo_agent.store_experience(
                        state=self.last_state,
                        action=self.last_action['action'],
                        reward=reward,
                        next_state=new_game_state,
                        done=new_game_state.get('winner') is not None,
                        action_type=self.last_action['action_type'],
                        valid_actions=[],  # Would need to compute valid actions
                        reasoning=self.last_action['reasoning']
                    )
                
                # Add to game history
                self.game_history.append({
                    'state': self.last_state,
                    'action': self.last_action,
                    'new_state': new_game_state,
                    'reward': reward,
                    'success': action_success
                })
                
                logger.debug(f"AI {self.agent_id} received feedback: success={action_success}, reward={reward:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing feedback for agent {self.agent_id}: {e}")
    
    async def game_ended(self, final_state: Dict, won: bool):
        """
        Handle game ending and update statistics
        
        Args:
            final_state: Final game state
            won: Whether this agent's team won
        """
        
        try:
            # Update game statistics
            self.performance_metrics['games_played'] += 1
            if won:
                self.performance_metrics['games_won'] += 1
            
            # Update role-specific win rate
            role = final_state.get('your_role', 'liberal')
            role_games = sum(1 for entry in self.game_history if entry['state'].get('your_role') == role)
            if role_games > 0:
                role_wins = sum(1 for entry in self.game_history 
                              if entry['state'].get('your_role') == role and 
                                 entry.get('won', False))
                self.performance_metrics['win_rate_by_role'][role] = role_wins / role_games
            
            # Trigger learning from the complete game
            if self.enable_self_training and len(self.game_history) > 5:
                # Add final reward signal to last few actions
                final_reward = 10.0 if won else -10.0
                for i in range(min(3, len(self.game_history))):
                    entry = self.game_history[-(i+1)]
                    entry['reward'] += final_reward * (0.5 ** i)  # Decay reward
                
                # Trigger training
                try:
                    self.training_queue.put("TRAIN", block=False)
                except queue.Full:
                    pass  # Training queue full, skip this trigger
            
            logger.info(f"Game ended for AI {self.agent_id}: won={won}, games_played={self.performance_metrics['games_played']}")
            
            # Reset for next game
            self.current_game_id = None
            self.player_id = None
            self.game_history = []
            self.last_action = None
            self.last_state = None
            
        except Exception as e:
            logger.error(f"Error handling game end for agent {self.agent_id}: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate additional metrics
        if metrics['games_played'] > 0:
            metrics['overall_win_rate'] = metrics['games_won'] / metrics['games_played']
        else:
            metrics['overall_win_rate'] = 0.0
        
        # Add model-specific metrics if available
        if hasattr(self.arbo_agent, 'games_played'):
            metrics['arbo_games_played'] = self.arbo_agent.games_played
            metrics['arbo_win_rate'] = self.arbo_agent.win_rate
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the AI model to disk"""
        try:
            self.arbo_agent.save_checkpoint(filepath)
            logger.info(f"Saved model for agent {self.agent_id} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model for agent {self.agent_id}: {e}")
    
    def load_model(self, filepath: str):
        """Load the AI model from disk"""
        try:
            self.arbo_agent.load_checkpoint(filepath)
            logger.info(f"Loaded model for agent {self.agent_id} from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model for agent {self.agent_id}: {e}")
    
    def shutdown(self):
        """Shutdown the AI agent and cleanup resources"""
        try:
            if self.training_thread and self.training_thread.is_alive():
                self.stop_training.set()
                self.training_queue.put("STOP")
                self.training_thread.join(timeout=10)
            
            logger.info(f"Shutdown AI agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error during shutdown for agent {self.agent_id}: {e}")

class AIPlayerManager:
    """Manages multiple AI players for a game"""
    
    def __init__(self):
        self.ai_agents: Dict[str, SecretHitlerAI] = {}
        self.active_games: Dict[str, List[str]] = {}  # game_id -> list of ai_agent_ids
    
    def create_ai_agent(
        self, 
        agent_id: str,
        difficulty: str = "medium",
        enable_self_training: bool = True
    ) -> SecretHitlerAI:
        """Create a new AI agent with specified difficulty"""
        
        # Configure model based on difficulty
        if difficulty == "easy":
            model_config = ModelConfig(temperature=1.2, top_p=0.95)
            arbo_config = ARBOConfig(
                uncertainty_weight=0.2,
                rational_temperature=1.5,
                advantage_threshold=0.2
            )
        elif difficulty == "hard":
            model_config = ModelConfig(temperature=0.6, top_p=0.85)
            arbo_config = ARBOConfig(
                uncertainty_weight=0.05,
                rational_temperature=0.8,
                advantage_threshold=0.05
            )
        else:  # medium
            model_config = ModelConfig(temperature=0.8, top_p=0.9)
            arbo_config = ARBOConfig()
        
        ai_agent = SecretHitlerAI(
            agent_id=agent_id,
            model_config=model_config,
            arbo_config=arbo_config,
            enable_self_training=enable_self_training
        )
        
        self.ai_agents[agent_id] = ai_agent
        logger.info(f"Created AI agent {agent_id} with difficulty {difficulty}")
        
        return ai_agent
    
    def get_ai_agent(self, agent_id: str) -> Optional[SecretHitlerAI]:
        """Get AI agent by ID"""
        return self.ai_agents.get(agent_id)
    
    async def add_ai_to_game(self, game_id: str, agent_id: str, player_id: str) -> bool:
        """Add AI agent to a game"""
        
        ai_agent = self.ai_agents.get(agent_id)
        if not ai_agent:
            return False
        
        success = await ai_agent.join_game(game_id, player_id)
        if success:
            if game_id not in self.active_games:
                self.active_games[game_id] = []
            self.active_games[game_id].append(agent_id)
        
        return success
    
    async def make_ai_decision(
        self,
        game_id: str,
        player_id: str,
        game_state: Dict,
        action_type: str,
        valid_targets: Optional[List[str]] = None
    ) -> Tuple[Any, float, str]:
        """Make decision for an AI player"""
        
        # Find which AI agent controls this player
        ai_agent = None
        for agent_id in self.active_games.get(game_id, []):
            agent = self.ai_agents[agent_id]
            if agent.player_id == player_id:
                ai_agent = agent
                break
        
        if not ai_agent:
            return None, 0.0, "AI agent not found"
        
        return await ai_agent.make_decision(game_state, action_type, valid_targets)
    
    async def notify_ai_feedback(
        self,
        game_id: str,
        player_id: str,
        new_game_state: Dict,
        action_success: bool
    ):
        """Send feedback to AI agent"""
        
        # Find AI agent
        for agent_id in self.active_games.get(game_id, []):
            agent = self.ai_agents[agent_id]
            if agent.player_id == player_id:
                await agent.receive_feedback(new_game_state, action_success)
                break
    
    async def notify_game_ended(self, game_id: str, final_state: Dict):
        """Notify all AI agents in a game that it has ended"""
        
        for agent_id in self.active_games.get(game_id, []):
            agent = self.ai_agents[agent_id]
            
            # Determine if this agent won
            agent_role = final_state.get('your_role', 'liberal')
            winner = final_state.get('winner')
            won = False
            
            if winner == 'liberal' and agent_role == 'liberal':
                won = True
            elif winner == 'fascist' and agent_role in ['fascist', 'hitler']:
                won = True
            
            await agent.game_ended(final_state, won)
        
        # Remove game from active games
        if game_id in self.active_games:
            del self.active_games[game_id]
    
    def get_all_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all AI agents"""
        return {
            agent_id: agent.get_performance_metrics()
            for agent_id, agent in self.ai_agents.items()
        }
    
    def shutdown_all(self):
        """Shutdown all AI agents"""
        for agent in self.ai_agents.values():
            agent.shutdown()
        
        self.ai_agents.clear()
        self.active_games.clear()
        logger.info("Shutdown all AI agents")

# Global AI manager instance
ai_manager = AIPlayerManager()