"""
Model Checkpointing System for Secret Hitler AI
Handles saving, loading, and managing model checkpoints with LoRA adapters.
"""

import os
import json
import torch
import pickle
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

class ModelCheckpointManager:
    """
    Manages model checkpoints for LoRA-trained Secret Hitler AI agents.
    Handles saving/loading of model weights, training state, and metadata.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Checkpoint metadata
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.load_metadata()
    
    def load_metadata(self):
        """Load checkpoint metadata from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "checkpoints": {},
                "latest_checkpoint": None,
                "total_checkpoints": 0
            }
    
    def save_metadata(self):
        """Save checkpoint metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_checkpoint_id(self, agent_role: str, training_step: int) -> str:
        """Create a unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{agent_role}_{training_step}_{timestamp}"
    
    def save_checkpoint(self, 
                       model, 
                       optimizer, 
                       training_state: Dict[str, Any],
                       agent_role: str,
                       training_step: int,
                       performance_metrics: Dict[str, float]) -> str:
        """
        Save a complete model checkpoint.
        
        Args:
            model: The LoRA-adapted model
            optimizer: Training optimizer
            training_state: Current training state
            agent_role: Role of the agent (liberal, fascist, hitler)
            training_step: Current training step
            performance_metrics: Performance metrics for this checkpoint
            
        Returns:
            checkpoint_id: Unique identifier for this checkpoint
        """
        checkpoint_id = self.create_checkpoint_id(agent_role, training_step)
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        # Prepare checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
            'training_state': training_state,
            'agent_role': agent_role,
            'training_step': training_step,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': {
                'base_model': getattr(model, 'base_model_name', 'unknown'),
                'lora_config': getattr(model, 'lora_config', {}),
                'special_tokens': getattr(model, 'special_tokens', [])
            }
        }
        
        try:
            # Save checkpoint
            if hasattr(torch, 'save'):
                torch.save(checkpoint_data, checkpoint_path)
            else:
                # Mock save for environments without PyTorch
                with open(checkpoint_path.with_suffix('.json'), 'w') as f:
                    # Convert non-serializable objects to strings
                    serializable_data = self._make_serializable(checkpoint_data)
                    json.dump(serializable_data, f, indent=2)
            
            # Update metadata
            self.metadata["checkpoints"][checkpoint_id] = {
                "path": str(checkpoint_path),
                "agent_role": agent_role,
                "training_step": training_step,
                "timestamp": checkpoint_data['timestamp'],
                "performance_metrics": performance_metrics,
                "file_size": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
            }
            
            self.metadata["latest_checkpoint"] = checkpoint_id
            self.metadata["total_checkpoints"] += 1
            self.save_metadata()
            
            self.logger.info(f"Saved checkpoint {checkpoint_id} for {agent_role} at step {training_step}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to load
            
        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_id not in self.metadata["checkpoints"]:
            self.logger.warning(f"Checkpoint {checkpoint_id} not found in metadata")
            return None
        
        checkpoint_info = self.metadata["checkpoints"][checkpoint_id]
        checkpoint_path = Path(checkpoint_info["path"])
        
        if not checkpoint_path.exists():
            # Try alternative extensions
            json_path = checkpoint_path.with_suffix('.json')
            if json_path.exists():
                checkpoint_path = json_path
            else:
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
        
        try:
            if checkpoint_path.suffix == '.pt' and hasattr(torch, 'load'):
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            else:
                # Load JSON checkpoint
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
            
            self.logger.info(f"Loaded checkpoint {checkpoint_id}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    def get_latest_checkpoint(self, agent_role: Optional[str] = None) -> Optional[str]:
        """
        Get the latest checkpoint ID, optionally filtered by agent role.
        
        Args:
            agent_role: Optional role filter
            
        Returns:
            Latest checkpoint ID or None
        """
        if not self.metadata["checkpoints"]:
            return None
        
        checkpoints = self.metadata["checkpoints"]
        
        if agent_role:
            # Filter by agent role
            role_checkpoints = {
                cid: info for cid, info in checkpoints.items() 
                if info.get("agent_role") == agent_role
            }
            if not role_checkpoints:
                return None
            checkpoints = role_checkpoints
        
        # Sort by timestamp and return latest
        latest = max(checkpoints.items(), key=lambda x: x[1]["timestamp"])
        return latest[0]
    
    def list_checkpoints(self, agent_role: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all checkpoints, optionally filtered by agent role.
        
        Args:
            agent_role: Optional role filter
            
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        for checkpoint_id, info in self.metadata["checkpoints"].items():
            if agent_role and info.get("agent_role") != agent_role:
                continue
            
            checkpoint_info = info.copy()
            checkpoint_info["checkpoint_id"] = checkpoint_id
            checkpoints.append(checkpoint_info)
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_count: int = 10, agent_role: Optional[str] = None):
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of checkpoints to keep per role
            agent_role: Optional role filter, if None cleans all roles
        """
        if agent_role:
            roles = [agent_role]
        else:
            roles = list(set(info.get("agent_role") for info in self.metadata["checkpoints"].values()))
        
        for role in roles:
            if not role:
                continue
                
            role_checkpoints = self.list_checkpoints(role)
            if len(role_checkpoints) <= keep_count:
                continue
            
            # Remove oldest checkpoints
            to_remove = role_checkpoints[keep_count:]
            for checkpoint_info in to_remove:
                checkpoint_id = checkpoint_info["checkpoint_id"]
                checkpoint_path = Path(checkpoint_info["path"])
                
                try:
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                    
                    # Also try JSON version
                    json_path = checkpoint_path.with_suffix('.json')
                    if json_path.exists():
                        json_path.unlink()
                    
                    del self.metadata["checkpoints"][checkpoint_id]
                    self.logger.info(f"Removed old checkpoint {checkpoint_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to remove checkpoint {checkpoint_id}: {e}")
        
        self.save_metadata()
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints"""
        stats = {
            "total_checkpoints": len(self.metadata["checkpoints"]),
            "checkpoints_by_role": {},
            "total_size_mb": 0,
            "oldest_checkpoint": None,
            "newest_checkpoint": None
        }
        
        if not self.metadata["checkpoints"]:
            return stats
        
        # Calculate stats
        timestamps = []
        for checkpoint_id, info in self.metadata["checkpoints"].items():
            role = info.get("agent_role", "unknown")
            if role not in stats["checkpoints_by_role"]:
                stats["checkpoints_by_role"][role] = 0
            stats["checkpoints_by_role"][role] += 1
            
            stats["total_size_mb"] += info.get("file_size", 0) / (1024 * 1024)
            timestamps.append((checkpoint_id, info["timestamp"]))
        
        # Find oldest and newest
        timestamps.sort(key=lambda x: x[1])
        stats["oldest_checkpoint"] = timestamps[0][0]
        stats["newest_checkpoint"] = timestamps[-1][0]
        
        return stats
    
    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif callable(obj):
            return f"<function {obj.__name__}>"
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

# Global checkpoint manager instance
checkpoint_manager = ModelCheckpointManager()

def get_checkpoint_manager() -> ModelCheckpointManager:
    """Get the global checkpoint manager instance"""
    return checkpoint_manager
