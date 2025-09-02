import asyncio
import logging
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
import time
import random
from collections import deque
import pickle
import os

from ..models.ai_agent import AIPlayerManager, SecretHitlerAI
from ..models.lama import ModelConfig
from ..models.arbo import ARBOConfig
from ..game.state import GameStateManager
from ..training.secret_hitler_env import SecretHitlerEnv
from ..training.wandb_logger import wandb_logger

logger = logging.getLogger(__name__)

class SelfTrainingOrchestrator:
    """
    Orchestrates self-training of AI agents during live gameplay
    Manages continuous learning and model improvement
    """
    
    def __init__(
        self,
        num_training_agents: int = 6,
        training_interval: timedelta = timedelta(minutes=30),
        games_per_session: int = 20,
        enable_live_learning: bool = True
    ):
        self.num_training_agents = num_training_agents
        self.training_interval = training_interval
        self.games_per_session = games_per_session
        self.enable_live_learning = enable_live_learning
        
        # Training infrastructure
        self.ai_manager = AIPlayerManager()
        self.game_manager = GameStateManager()
        self.training_env = None
        
        # Training scheduling
        self.training_thread = None
        self.training_active = False
        self.last_training_time = None
        self.stop_training = threading.Event()
        
        # Training data collection
        self.training_data = {
            'games': [],
            'agent_performances': {},
            'training_sessions': [],
            'model_versions': {}
        }
        
        # Live learning buffer
        self.live_game_buffer = deque(maxlen=1000)
        self.experience_buffer = deque(maxlen=10000)
        
        # Training statistics
        self.training_stats = {
            'total_training_games': 0,
            'total_training_sessions': 0,
            'agent_improvements': {},
            'convergence_metrics': {},
            'performance_trends': {}
        }
        
        # Initialize WandB logging
        self._initialize_wandb_logging()
        
        self._initialize_training_agents()
    
    def _initialize_wandb_logging(self):
        """Initialize WandB logging for training"""
        try:
            config = {
                "num_training_agents": self.num_training_agents,
                "training_interval_minutes": self.training_interval.total_seconds() / 60,
                "games_per_session": self.games_per_session,
                "enable_live_learning": self.enable_live_learning
            }
            
            wandb_logger.initialize(config=config, run_name=f"secret_hitler_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            logger.info("WandB logging initialized for training")
            
        except Exception as e:
            logger.warning(f"Failed to initialize WandB logging: {e}")
    
    def _initialize_training_agents(self):
        """Initialize AI agents for self-training"""
        
        try:
            # Create diverse AI agents with different configurations
            agent_configs = [
                ("aggressive_liberal", "hard", True),
                ("conservative_liberal", "medium", True),
                ("strategic_fascist", "hard", True),
                ("deceptive_fascist", "medium", True),
                ("hitler_specialist", "hard", True),
                ("balanced_player", "medium", True)
            ]
            
            for i, (agent_type, difficulty, enable_training) in enumerate(agent_configs[:self.num_training_agents]):
                agent_id = f"training_agent_{i}_{agent_type}"
                
                # Create agent with specialized configuration
                if agent_type.endswith("liberal"):
                    model_config = ModelConfig(temperature=0.7)
                    arbo_config = ARBOConfig(rational_temperature=0.9)
                elif agent_type.endswith("fascist"):
                    model_config = ModelConfig(temperature=0.8)
                    arbo_config = ARBOConfig(rational_temperature=1.1, uncertainty_weight=0.15)
                elif agent_type == "hitler_specialist":
                    model_config = ModelConfig(temperature=0.6)
                    arbo_config = ARBOConfig(rational_temperature=0.8, advantage_threshold=0.03)
                else:
                    model_config = ModelConfig()
                    arbo_config = ARBOConfig()
                
                agent = self.ai_manager.create_ai_agent(
                    agent_id=agent_id,
                    difficulty=difficulty,
                    enable_self_training=enable_training
                )
                
                # Customize agent for specific role preferences
                if hasattr(agent.arbo_agent, 'goal_weights'):
                    if "liberal" in agent_type:
                        agent.arbo_agent.goal_weights['liberal']['liberal_policy'] = 1.2
                        agent.arbo_agent.goal_weights['liberal']['fascist_policy'] = -1.2
                    elif "fascist" in agent_type:
                        agent.arbo_agent.goal_weights['fascist']['fascist_policy'] = 1.2
                        agent.arbo_agent.goal_weights['fascist']['liberal_policy'] = -1.2
                
                logger.info(f"Initialized training agent: {agent_id}")
        
        except Exception as e:
            logger.error(f"Error initializing training agents: {e}")
    
    def start_continuous_training(self):
        """Start continuous self-training in background"""
        
        if self.training_thread and self.training_thread.is_alive():
            logger.warning("Training already running")
            return
        
        self.training_active = True
        self.stop_training.clear()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("Started continuous self-training")
    
    def stop_continuous_training(self):
        """Stop continuous self-training"""
        
        self.training_active = False
        self.stop_training.set()
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        logger.info("Stopped continuous self-training")
    
    def _training_loop(self):
        """Main training loop running in background"""
        
        # Run the async training loop
        asyncio.run(self._async_training_loop())
    
    async def _async_training_loop(self):
        """Async training loop implementation"""
        
        while self.training_active and not self.stop_training.is_set():
            try:
                # Check if it's time for a training session
                should_train = (
                    self.last_training_time is None or
                    datetime.now() - self.last_training_time >= self.training_interval
                )
                
                if should_train:
                    await self._run_training_session()
                    self.last_training_time = datetime.now()
                
                # Process live learning data
                if self.enable_live_learning:
                    self._process_live_learning_batch()
                
                # Sleep before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _run_training_session(self):
        """Run a complete training session"""
        
        try:
            logger.info(f"Starting training session with {self.games_per_session} games")
            
            session_start_time = datetime.now()
            session_results = {
                'session_id': f"session_{int(time.time())}",
                'start_time': session_start_time.isoformat(),
                'games_played': 0,
                'agent_results': {},
                'performance_improvements': {},
                'convergence_metrics': {}
            }
            
            # Create training environment
            training_env = SecretHitlerEnv(
                num_players=7,
                ai_players=list(self.ai_manager.ai_agents.keys())[:6]
            )
            
            # Run training games
            for game_num in range(self.games_per_session):
                try:
                    game_result = await self._run_training_game(training_env, game_num)
                    session_results['games_played'] += 1
                    
                    # Collect results
                    for agent_id, result in game_result.items():
                        if agent_id not in session_results['agent_results']:
                            session_results['agent_results'][agent_id] = []
                        session_results['agent_results'][agent_id].append(result)
                    
                    # Periodic training during the session
                    if (game_num + 1) % 5 == 0:
                        self._train_agents_batch()
                
                except Exception as e:
                    logger.error(f"Error in training game {game_num}: {e}")
            
            # Final training after all games
            self._train_agents_batch()
            
            # Analyze session results
            session_results['end_time'] = datetime.now().isoformat()
            session_results['duration_minutes'] = (
                datetime.now() - session_start_time
            ).total_seconds() / 60
            
            # Calculate performance improvements
            session_results['performance_improvements'] = self._calculate_performance_improvements()
            
            # Store session results
            self.training_data['training_sessions'].append(session_results)
            self.training_stats['total_training_sessions'] += 1
            self.training_stats['total_training_games'] += session_results['games_played']
            
            # Log to WandB
            session_results['total_sessions'] = self.training_stats['total_training_sessions']
            wandb_logger.log_training_session(session_results)
            
            # Save training data
            self._save_training_data()
            
            logger.info(f"Completed training session: {session_results['games_played']} games in {session_results['duration_minutes']:.1f} minutes")
            
        except Exception as e:
            logger.error(f"Error in training session: {e}")
    
    async def _run_training_game(self, training_env, game_num: int) -> Dict[str, Any]:
        """Run a single training game and collect results"""
        
        try:
            # Reset environment
            obs = training_env.reset()
            done = False
            step_count = 0
            game_results = {}
            
            # Track agent performances in this game
            agent_performances = {agent_id: {
                'actions_taken': 0,
                'successful_actions': 0,
                'total_reward': 0.0,
                'final_role': None,
                'won': False
            } for agent_id in self.ai_manager.ai_agents.keys()}
            
            while not done and step_count < 200:  # Max steps to prevent infinite games
                # Get current player from environment
                current_player_id = training_env._get_current_player()
                if not current_player_id:
                    break
                
                # Find which AI agent controls this player
                ai_agent = None
                agent_id = None
                for aid, agent in self.ai_manager.ai_agents.items():
                    if agent.player_id == current_player_id:
                        ai_agent = agent
                        agent_id = aid
                        break
                
                if ai_agent:
                    # Get game state for decision making
                    game_state = training_env.game_manager.get_player_view(
                        training_env.current_game_id, current_player_id
                    )
                    
                    # Determine action type based on game phase
                    action_type = self._determine_action_type(game_state)
                    valid_targets = self._get_valid_targets(game_state, action_type)
                    
                    # Make AI decision
                    action, confidence, reasoning = await ai_agent.make_decision(
                        game_state, action_type, valid_targets
                    )
                    
                    # Convert AI action to environment action
                    env_action = self._convert_ai_action_to_env(action, game_state, action_type)
                    
                    # Execute action in environment
                    obs, reward, done, info = training_env.step(env_action, current_player_id)
                    
                    # Update agent performance tracking
                    agent_performances[agent_id]['actions_taken'] += 1
                    agent_performances[agent_id]['total_reward'] += reward
                    if info.get('action_info', {}).get('success'):
                        agent_performances[agent_id]['successful_actions'] += 1
                    
                    # Provide feedback to AI
                    new_game_state = training_env.game_manager.get_player_view(
                        training_env.current_game_id, current_player_id
                    )
                    await ai_agent.receive_feedback(new_game_state, info.get('action_info', {}).get('success', False))
                
                step_count += 1
            
            # Game finished - collect final results
            final_game_state = training_env.game_manager.get_game_state(training_env.current_game_id)
            
            for agent_id, performance in agent_performances.items():
                # Determine final role and win status
                for player in final_game_state.players:
                    agent = self.ai_manager.ai_agents.get(agent_id)
                    if agent and agent.player_id == player.id:
                        performance['final_role'] = player.role.value if player.role else None
                        
                        # Determine if won
                        if final_game_state.winner:
                            if (player.role.value == 'liberal' and final_game_state.winner.value == 'liberal') or \
                               (player.role.value in ['fascist', 'hitler'] and final_game_state.winner.value == 'fascist'):
                                performance['won'] = True
                
                # Notify agent of game end
                agent = self.ai_manager.ai_agents.get(agent_id)
                if agent:
                    await agent.game_ended(
                        training_env.game_manager.get_player_view(training_env.current_game_id, agent.player_id),
                        performance['won']
                    )
            
            game_results = {
                'game_number': game_num,
                'winner': final_game_state.winner.value if final_game_state.winner else None,
                'game_length': step_count,
                'agent_performances': agent_performances
            }
            
            # Log game to WandB
            game_id = f"training_game_{int(time.time())}_{game_num}"
            players = [{
                'id': agent_id,
                'is_ai': True,
                'role': perf.get('final_role'),
                'won': perf.get('won', False)
            } for agent_id, perf in agent_performances.items()]
            
            wandb_logger.log_game_start(game_id, players, {'training': True, 'game_type': 'self_play'})
            wandb_logger.log_game_end(game_id, 
                {'winning_team': final_game_state.winner.value if final_game_state.winner else 'unknown'},
                {
                    'duration_minutes': step_count / 60,  # Rough estimate
                    'num_rounds': step_count,
                    'player_stats': {agent_id: {
                        'is_ai': True,
                        'actions_taken': perf.get('actions_taken', 0),
                        'votes_cast': 0,  # Would need to track this
                        'policies_enacted': 0,  # Would need to track this
                        'avg_confidence': 0.8,  # Would need to track this
                        'won': perf.get('won', False)
                    } for agent_id, perf in agent_performances.items()}
                }
            )
            
            return game_results
            
        except Exception as e:
            logger.error(f"Error running training game {game_num}: {e}")
            return {}
    
    def _determine_action_type(self, game_state: Dict) -> str:
        """Determine what type of action is needed based on game state"""
        
        phase = game_state.get('phase', 'lobby')
        
        if phase == 'election':
            if not game_state.get('nominated_chancellor_id'):
                return 'nominate'
            else:
                return 'vote'
        elif phase == 'legislative':
            drawn_policies = game_state.get('drawn_policies', [])
            if len(drawn_policies) == 3:
                return 'discard'
            elif len(drawn_policies) == 2:
                return 'enact'
        elif phase == 'executive':
            # Would handle executive powers here
            return 'executive'
        
        return 'wait'
    
    def _get_valid_targets(self, game_state: Dict, action_type: str) -> Optional[List[str]]:
        """Get valid targets for the action"""
        
        if action_type == 'nominate':
            current_player_id = game_state.get('current_president_id')
            return [
                p['id'] for p in game_state.get('players', [])
                if p.get('is_alive') and p['id'] != current_player_id
            ]
        
        return None
    
    def _convert_ai_action_to_env(self, ai_action: Any, game_state: Dict, action_type: str) -> int:
        """Convert AI action to environment action ID"""
        
        if action_type == 'nominate':
            # Find player index for the nominated player ID
            players = game_state.get('players', [])
            for i, player in enumerate(players):
                if player['id'] == ai_action:
                    return i
            return 0
        
        elif action_type == 'vote':
            return 0 if ai_action == 'ja' else 1
        
        elif action_type in ['discard', 'enact']:
            # Find policy index
            drawn_policies = game_state.get('drawn_policies', [])
            for i, policy in enumerate(drawn_policies):
                if policy['id'] == ai_action:
                    return i
            return 0
        
        return 0
    
    def _train_agents_batch(self):
        """Train all agents using their collected experience"""
        
        try:
            training_results = {}
            
            for agent_id, agent in self.ai_manager.ai_agents.items():
                if hasattr(agent, 'arbo_agent'):
                    # Train the ARBO agent
                    loss_info = agent.arbo_agent.train_step()
                    if loss_info:
                        training_results[agent_id] = loss_info
            
            if training_results:
                logger.debug(f"Batch training completed: {len(training_results)} agents trained")
            
        except Exception as e:
            logger.error(f"Error in batch training: {e}")
    
    def _process_live_learning_batch(self):
        """Process live game data for continuous learning"""
        
        try:
            if len(self.live_game_buffer) < 10:
                return  # Not enough data
            
            # Process recent live games
            recent_games = list(self.live_game_buffer)[-10:]
            
            for game_data in recent_games:
                # Extract learning signals from live gameplay
                self._extract_learning_signals(game_data)
            
            # Apply incremental learning
            self._apply_incremental_learning()
            
        except Exception as e:
            logger.error(f"Error processing live learning batch: {e}")
    
    def _extract_learning_signals(self, game_data: Dict):
        """Extract learning signals from live game data"""
        
        try:
            # This would analyze live game patterns and extract insights
            # For now, we'll add the game data to the experience buffer
            
            experience = {
                'timestamp': datetime.now().isoformat(),
                'game_id': game_data.get('game_id'),
                'winner': game_data.get('winner'),
                'game_length': game_data.get('game_length', 0),
                'player_actions': game_data.get('player_actions', []),
                'key_moments': game_data.get('key_moments', [])
            }
            
            self.experience_buffer.append(experience)
            
        except Exception as e:
            logger.error(f"Error extracting learning signals: {e}")
    
    def _apply_incremental_learning(self):
        """Apply incremental learning from experience buffer"""
        
        try:
            if len(self.experience_buffer) < 50:
                return
            
            # Sample recent experiences
            recent_experiences = list(self.experience_buffer)[-20:]
            
            # Analyze patterns and update agent behaviors
            pattern_insights = self._analyze_gameplay_patterns(recent_experiences)
            
            # Apply insights to agent configurations
            self._update_agent_strategies(pattern_insights)
            
        except Exception as e:
            logger.error(f"Error applying incremental learning: {e}")
    
    def _analyze_gameplay_patterns(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in recent gameplay"""
        
        patterns = {
            'successful_strategies': {},
            'common_mistakes': {},
            'winning_behaviors': {},
            'role_performance': {}
        }
        
        try:
            # Analyze winning patterns
            winning_games = [exp for exp in experiences if exp.get('winner')]
            
            for game in winning_games:
                winner = game['winner']
                if winner not in patterns['winning_behaviors']:
                    patterns['winning_behaviors'][winner] = []
                
                # Extract key behaviors that led to win
                key_moments = game.get('key_moments', [])
                patterns['winning_behaviors'][winner].extend(key_moments)
            
            # Analyze role performance
            for exp in experiences:
                # This would be more sophisticated in practice
                game_length = exp.get('game_length', 0)
                winner = exp.get('winner')
                
                if winner not in patterns['role_performance']:
                    patterns['role_performance'][winner] = {
                        'average_game_length': 0,
                        'win_rate': 0,
                        'games_counted': 0
                    }
                
                role_perf = patterns['role_performance'][winner]
                role_perf['games_counted'] += 1
                role_perf['average_game_length'] = (
                    (role_perf['average_game_length'] * (role_perf['games_counted'] - 1) + game_length) /
                    role_perf['games_counted']
                )
        
        except Exception as e:
            logger.error(f"Error analyzing gameplay patterns: {e}")
        
        return patterns
    
    def _update_agent_strategies(self, pattern_insights: Dict):
        """Update agent strategies based on pattern insights"""
        
        try:
            for agent_id, agent in self.ai_manager.ai_agents.items():
                if not hasattr(agent, 'arbo_agent'):
                    continue
                
                # Adjust ARBO configuration based on insights
                arbo_config = agent.arbo_agent.config
                
                # Example: If fascists are winning too often, make liberal agents more aggressive
                fascist_performance = pattern_insights.get('role_performance', {}).get('fascist', {})
                if fascist_performance.get('win_rate', 0) > 0.6:
                    if 'liberal' in agent_id:
                        arbo_config.rational_temperature *= 0.95  # More decisive
                        arbo_config.uncertainty_weight *= 0.9  # Less exploration
                
                # Update goal weights based on successful strategies
                winning_behaviors = pattern_insights.get('winning_behaviors', {})
                for winner, behaviors in winning_behaviors.items():
                    if winner in agent_id:
                        # Reinforce successful behaviors
                        # This would be more sophisticated in practice
                        pass
            
            logger.debug("Updated agent strategies based on pattern insights")
            
        except Exception as e:
            logger.error(f"Error updating agent strategies: {e}")
    
    def _calculate_performance_improvements(self) -> Dict[str, float]:
        """Calculate performance improvements for each agent"""
        
        improvements = {}
        
        try:
            for agent_id, agent in self.ai_manager.ai_agents.items():
                current_metrics = agent.get_performance_metrics()
                
                # Compare with previous session if available
                if agent_id in self.training_stats.get('agent_improvements', {}):
                    previous_metrics = self.training_stats['agent_improvements'][agent_id]
                    
                    # Calculate win rate improvement
                    current_win_rate = current_metrics.get('overall_win_rate', 0)
                    previous_win_rate = previous_metrics.get('overall_win_rate', 0)
                    
                    win_rate_improvement = current_win_rate - previous_win_rate
                    
                    improvements[agent_id] = {
                        'win_rate_improvement': win_rate_improvement,
                        'current_win_rate': current_win_rate,
                        'games_played_since_last': current_metrics.get('games_played', 0) - previous_metrics.get('games_played', 0)
                    }
                
                # Store current metrics for next comparison
                self.training_stats['agent_improvements'][agent_id] = current_metrics
        
        except Exception as e:
            logger.error(f"Error calculating performance improvements: {e}")
        
        return improvements
    
    def _save_training_data(self):
        """Save training data to disk"""
        
        try:
            os.makedirs('training_data', exist_ok=True)
            
            # Save main training data
            with open('training_data/training_data.json', 'w') as f:
                json.dump(self.training_data, f, indent=2, default=str)
            
            # Save training statistics
            with open('training_data/training_stats.json', 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)
            
            # Save experience buffer
            with open('training_data/experience_buffer.pkl', 'wb') as f:
                pickle.dump(list(self.experience_buffer), f)
            
            logger.debug("Saved training data to disk")
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
    
    def add_live_game_data(self, game_data: Dict):
        """Add data from a live game for learning"""
        
        if self.enable_live_learning:
            self.live_game_buffer.append(game_data)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        
        return {
            'training_active': self.training_active,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'next_training_time': (
                self.last_training_time + self.training_interval
            ).isoformat() if self.last_training_time else None,
            'total_training_sessions': self.training_stats['total_training_sessions'],
            'total_training_games': self.training_stats['total_training_games'],
            'agent_count': len(self.ai_manager.ai_agents),
            'live_game_buffer_size': len(self.live_game_buffer),
            'experience_buffer_size': len(self.experience_buffer)
        }
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        
        return {
            'individual_performances': self.ai_manager.get_all_performance_metrics(),
            'training_stats': self.training_stats,
            'recent_improvements': self.training_stats.get('agent_improvements', {})
        }
    
    def shutdown(self):
        """Shutdown the training orchestrator"""
        
        try:
            self.stop_continuous_training()
            self._save_training_data()
            self.ai_manager.shutdown_all()
            
            logger.info("Self-training orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during training orchestrator shutdown: {e}")

# Global training orchestrator instance
training_orchestrator = SelfTrainingOrchestrator()