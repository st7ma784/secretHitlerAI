import wandb
import logging
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class WandBLogger:
    """
    WandB integration for logging Secret Hitler AI training metrics and games
    """
    
    def __init__(self, project_name: str = "secret-hitler-ai", entity: Optional[str] = None):
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.is_initialized = False
        
        # Game logging configuration
        self.log_games = True
        self.log_training_metrics = True
        self.log_agent_performance = True
        
        # Metrics buffers
        self.game_metrics_buffer = []
        self.training_metrics_buffer = []
        self.agent_performance_buffer = []
        
    def initialize(self, config: Optional[Dict] = None, run_name: Optional[str] = None):
        """Initialize WandB run"""
        try:
            # Default configuration
            default_config = {
                "model_type": "secret_hitler_ai",
                "training_mode": "self_play",
                "num_agents": 6,
                "games_per_session": 20,
                "training_interval_minutes": 30,
                "enable_live_learning": True
            }
            
            if config:
                default_config.update(config)
            
            # Generate run name if not provided
            if not run_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"secret_hitler_training_{timestamp}"
            
            # Initialize WandB
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=default_config,
                reinit=True
            )
            
            self.is_initialized = True
            logger.info(f"WandB initialized: {self.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")
            self.is_initialized = False
    
    def log_game_start(self, game_id: str, players: List[Dict], game_config: Dict):
        """Log the start of a new game"""
        if not self.is_initialized:
            return
        
        try:
            game_data = {
                "game_id": game_id,
                "timestamp": datetime.now().isoformat(),
                "num_players": len(players),
                "num_ai_players": sum(1 for p in players if p.get("is_ai", False)),
                "num_human_players": sum(1 for p in players if not p.get("is_ai", False)),
                "game_config": game_config
            }
            
            # Log to WandB
            wandb.log({
                "game/started": 1,
                "game/num_players": len(players),
                "game/num_ai_players": game_data["num_ai_players"],
                "game/num_human_players": game_data["num_human_players"]
            })
            
            # Store for detailed logging
            self.game_metrics_buffer.append({
                "type": "game_start",
                "data": game_data
            })
            
        except Exception as e:
            logger.error(f"Error logging game start: {e}")
    
    def log_game_end(self, game_id: str, result: Dict, game_stats: Dict):
        """Log the end of a game with results and statistics"""
        if not self.is_initialized:
            return
        
        try:
            # Extract key metrics
            winning_team = result.get("winning_team", "unknown")
            game_duration = game_stats.get("duration_minutes", 0)
            num_rounds = game_stats.get("num_rounds", 0)
            
            # Log basic metrics
            wandb.log({
                "game/completed": 1,
                "game/winning_team": winning_team,
                "game/duration_minutes": game_duration,
                "game/num_rounds": num_rounds,
                "game/liberal_wins": 1 if winning_team == "liberal" else 0,
                "game/fascist_wins": 1 if winning_team == "fascist" else 0,
                "game/hitler_wins": 1 if winning_team == "hitler" else 0
            })
            
            # Log detailed game statistics
            if "player_stats" in game_stats:
                for player_id, stats in game_stats["player_stats"].items():
                    if stats.get("is_ai", False):
                        wandb.log({
                            f"ai_player/{player_id}/actions_taken": stats.get("actions_taken", 0),
                            f"ai_player/{player_id}/votes_cast": stats.get("votes_cast", 0),
                            f"ai_player/{player_id}/policies_enacted": stats.get("policies_enacted", 0),
                            f"ai_player/{player_id}/confidence_avg": stats.get("avg_confidence", 0),
                            f"ai_player/{player_id}/won_game": 1 if stats.get("won", False) else 0
                        })
            
            # Store detailed game data
            game_end_data = {
                "type": "game_end",
                "game_id": game_id,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "stats": game_stats
            }
            
            self.game_metrics_buffer.append(game_end_data)
            
            # Create game summary table
            self._log_game_summary_table(game_id, result, game_stats)
            
        except Exception as e:
            logger.error(f"Error logging game end: {e}")
    
    def log_training_session(self, session_data: Dict):
        """Log training session metrics"""
        if not self.is_initialized:
            return
        
        try:
            session_id = session_data.get("session_id", "unknown")
            games_played = session_data.get("games_played", 0)
            session_duration = session_data.get("duration_minutes", 0)
            
            # Log session metrics
            wandb.log({
                "training/session_completed": 1,
                "training/games_per_session": games_played,
                "training/session_duration_minutes": session_duration,
                "training/total_sessions": session_data.get("total_sessions", 0)
            })
            
            # Log agent improvements
            if "agent_improvements" in session_data:
                for agent_id, improvement in session_data["agent_improvements"].items():
                    wandb.log({
                        f"training/{agent_id}/win_rate_improvement": improvement.get("win_rate_improvement", 0),
                        f"training/{agent_id}/current_win_rate": improvement.get("current_win_rate", 0),
                        f"training/{agent_id}/games_played": improvement.get("games_played_since_last", 0)
                    })
            
            self.training_metrics_buffer.append({
                "type": "training_session",
                "data": session_data,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error logging training session: {e}")
    
    def log_agent_performance(self, agent_id: str, performance_metrics: Dict):
        """Log individual agent performance metrics"""
        if not self.is_initialized:
            return
        
        try:
            # Log key performance indicators
            wandb.log({
                f"agent/{agent_id}/win_rate": performance_metrics.get("win_rate", 0),
                f"agent/{agent_id}/games_played": performance_metrics.get("games_played", 0),
                f"agent/{agent_id}/avg_confidence": performance_metrics.get("avg_confidence", 0),
                f"agent/{agent_id}/liberal_win_rate": performance_metrics.get("liberal_win_rate", 0),
                f"agent/{agent_id}/fascist_win_rate": performance_metrics.get("fascist_win_rate", 0),
                f"agent/{agent_id}/hitler_win_rate": performance_metrics.get("hitler_win_rate", 0),
                f"agent/{agent_id}/avg_game_duration": performance_metrics.get("avg_game_duration", 0)
            })
            
            # Store detailed performance data
            self.agent_performance_buffer.append({
                "type": "agent_performance",
                "agent_id": agent_id,
                "metrics": performance_metrics,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error logging agent performance: {e}")
    
    def log_model_update(self, agent_id: str, model_metrics: Dict):
        """Log model update/training metrics"""
        if not self.is_initialized:
            return
        
        try:
            wandb.log({
                f"model/{agent_id}/loss": model_metrics.get("loss", 0),
                f"model/{agent_id}/learning_rate": model_metrics.get("learning_rate", 0),
                f"model/{agent_id}/gradient_norm": model_metrics.get("gradient_norm", 0),
                f"model/{agent_id}/training_step": model_metrics.get("training_step", 0),
                f"model/{agent_id}/model_version": model_metrics.get("model_version", 0)
            })
            
        except Exception as e:
            logger.error(f"Error logging model update: {e}")
    
    def _log_game_summary_table(self, game_id: str, result: Dict, game_stats: Dict):
        """Create a detailed game summary table"""
        try:
            # Create table data
            table_data = []
            
            if "player_stats" in game_stats:
                for player_id, stats in game_stats["player_stats"].items():
                    table_data.append([
                        player_id,
                        stats.get("role", "unknown"),
                        stats.get("team", "unknown"),
                        "AI" if stats.get("is_ai", False) else "Human",
                        stats.get("actions_taken", 0),
                        stats.get("votes_cast", 0),
                        stats.get("policies_enacted", 0),
                        round(stats.get("avg_confidence", 0), 3),
                        "Won" if stats.get("won", False) else "Lost"
                    ])
            
            # Create WandB table
            table = wandb.Table(
                columns=[
                    "Player ID", "Role", "Team", "Type", 
                    "Actions", "Votes", "Policies", "Avg Confidence", "Result"
                ],
                data=table_data
            )
            
            wandb.log({f"game_summary/{game_id}": table})
            
        except Exception as e:
            logger.error(f"Error creating game summary table: {e}")
    
    def log_custom_metric(self, metric_name: str, value: Any, step: Optional[int] = None):
        """Log custom metric to WandB"""
        if not self.is_initialized:
            return
        
        try:
            log_data = {metric_name: value}
            if step is not None:
                log_data["step"] = step
            
            wandb.log(log_data)
            
        except Exception as e:
            logger.error(f"Error logging custom metric: {e}")
    
    def save_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "dataset"):
        """Save file as WandB artifact"""
        if not self.is_initialized:
            return
        
        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            
        except Exception as e:
            logger.error(f"Error saving artifact: {e}")
    
    def finish(self):
        """Finish the WandB run"""
        if self.is_initialized and self.run:
            try:
                # Log any remaining buffered data
                self._flush_buffers()
                
                wandb.finish()
                self.is_initialized = False
                logger.info("WandB run finished")
                
            except Exception as e:
                logger.error(f"Error finishing WandB run: {e}")
    
    def _flush_buffers(self):
        """Flush any remaining buffered metrics"""
        try:
            # Save buffered data as artifacts if significant amount
            if len(self.game_metrics_buffer) > 10:
                self._save_buffer_as_artifact("game_metrics", self.game_metrics_buffer)
            
            if len(self.training_metrics_buffer) > 5:
                self._save_buffer_as_artifact("training_metrics", self.training_metrics_buffer)
            
            if len(self.agent_performance_buffer) > 20:
                self._save_buffer_as_artifact("agent_performance", self.agent_performance_buffer)
                
        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
    
    def _save_buffer_as_artifact(self, buffer_name: str, buffer_data: List):
        """Save buffer data as WandB artifact"""
        try:
            import tempfile
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(buffer_data, f, indent=2, default=str)
                temp_path = f.name
            
            # Save as artifact
            self.save_artifact(temp_path, f"{buffer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "logs")
            
            # Clean up
            os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Error saving buffer as artifact: {e}")

# Global WandB logger instance
wandb_logger = WandBLogger()
