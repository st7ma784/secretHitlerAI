#!/usr/bin/env python3
"""
Simple FastAPI server for Secret Hitler AI training interface.
This is a minimal version that provides the training API endpoints
without requiring the full game logic dependencies.
"""

import os
import sys
import asyncio
import random
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import os

# Add the training module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.training.llm_trainer import get_llm_trainer, LoRAConfig, RLHFConfig
    from backend.training.model_checkpointing import get_checkpoint_manager
    from backend.game.enhanced_ai_agent import EnhancedSecretHitlerAgent
    from backend.game.agent_worldview import PlayerRole
    LLM_TRAINER_AVAILABLE = True
except ImportError:
    LLM_TRAINER_AVAILABLE = False
    print("LLM trainer not available - using mock implementation")

app = FastAPI(title="Secret Hitler AI Training API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
try:
    app.mount("/static", StaticFiles(directory="../frontend/public"), name="static")
except:
    pass  # Directory might not exist

# Training state
training_state = {
    "training_active": False,
    "total_training_sessions": 0,
    "total_training_games": 0,
    "agent_count": 6,
    "last_training_time": None,
    "next_training_time": None,
    "current_config": {
        "num_agents": 6,
        "games_per_session": 20,
        "training_interval_minutes": 30,
        "enable_live_learning": True,
        "lora_rank": 16,
        "training_method": "rlhf"
    }
}

# Initialize LLM trainer
llm_trainer = None
if LLM_TRAINER_AVAILABLE:
    llm_trainer = get_llm_trainer()

class TrainingConfig(BaseModel):
    num_agents: int = 6
    games_per_session: int = 20
    training_interval_minutes: int = 30
    enable_live_learning: bool = True
    lora_rank: Optional[int] = 16
    training_method: Optional[str] = "rlhf"  # rlhf, sft, dpo
    
    class Config:
        # Allow extra fields to be ignored
        extra = "ignore"

@app.get("/")
async def root():
    return {"message": "Secret Hitler AI Training API", "status": "running"}

@app.get("/training")
async def training_interface():
    """Serve the training interface HTML."""
    from fastapi.responses import FileResponse
    try:
        return FileResponse("../frontend/public/training-interface.html")
    except:
        return {"error": "Training interface not found"}

@app.get("/api/ai/training-status")
async def get_training_status():
    """Get current training status and statistics."""
    return training_state

@app.get("/api/ai/training-metrics")
async def get_training_metrics():
    """Get detailed training metrics"""
    if not LLM_TRAINER_AVAILABLE:
        return {
            "error": "LLM trainer not available",
            "mock_metrics": {
                "training_loss": 2.5 - (training_state["total_training_games"] * 0.01),
                "reward_score": min(1.0, training_state["total_training_games"] * 0.001),
                "policy_gradient": abs(random.uniform(-0.05, 0.05)),
                "kl_divergence": abs(random.uniform(0.001, 0.01)),
                "value_function": random.uniform(-0.1, 0.1),
                "learning_rate": 1e-4 * (0.95 ** (training_state["total_training_games"] // 10)),
                "lora_rank": training_state["current_config"].get("lora_rank", 16),
                "training_method": training_state["current_config"].get("training_method", "rlhf")
            }
        }
    
    try:
        trainer = get_llm_trainer()
        metrics = trainer.get_training_metrics()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        return {"error": f"Failed to get training metrics: {str(e)}"}

@app.post("/api/ai/configure-training-raw")
async def configure_training_raw(request: dict):
    """Debug endpoint to see raw request data."""
    print(f"Raw request data: {request}")
    return {"received": request}

@app.post("/api/ai/configure-training")
async def configure_training(request: dict):
    """Configure training parameters - accepts raw JSON."""
    try:
        print(f"Received raw config: {request}")
        
        # Extract and validate fields manually
        lora_rank_val = request.get("lora_rank", 16)
        if lora_rank_val is None:
            lora_rank_val = 16
        
        config_data = {
            "num_agents": int(request.get("num_agents", 6)),
            "games_per_session": int(request.get("games_per_session", 20)),
            "training_interval_minutes": int(request.get("training_interval_minutes", 30)),
            "enable_live_learning": bool(request.get("enable_live_learning", True)),
            "lora_rank": int(lora_rank_val),
            "training_method": str(request.get("training_method", "rlhf") or "rlhf")
        }
        
        print(f"Processed config: {config_data}")
        
        training_state["current_config"] = config_data
        training_state["agent_count"] = config_data["num_agents"]
        
        # If training is active, update next training time
        if training_state["training_active"]:
            next_time = datetime.now() + timedelta(minutes=config_data["training_interval_minutes"])
            training_state["next_training_time"] = next_time.isoformat()
        
        return {"message": "Training configuration updated", "config": config_data}
    except Exception as e:
        print(f"Configuration error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Configuration error: {str(e)}")

@app.post("/api/ai/start-training")
async def start_training():
    """Start AI training."""
    if training_state["training_active"]:
        raise HTTPException(status_code=400, detail="Training is already active")
    
    training_state["training_active"] = True
    training_state["last_training_time"] = datetime.now().isoformat()
    
    # Calculate next training time
    interval = training_state["current_config"]["training_interval_minutes"]
    next_time = datetime.now() + timedelta(minutes=interval)
    training_state["next_training_time"] = next_time.isoformat()
    
    # Start background training simulation
    asyncio.create_task(simulate_training())
    
    return {"message": "Training started successfully"}

@app.post("/api/ai/stop-training")
async def stop_training():
    """Stop AI training."""
    if not training_state["training_active"]:
        raise HTTPException(status_code=400, detail="Training is not active")
    
    training_state["training_active"] = False
    training_state["next_training_time"] = None
    
    return {"message": "Training stopped successfully"}

@app.get("/api/ai/checkpoints")
async def list_checkpoints(agent_role: str = None):
    """List available model checkpoints"""
    if not LLM_TRAINER_AVAILABLE:
        return {"error": "LLM trainer not available"}
    
    try:
        checkpoint_manager = get_checkpoint_manager()
        checkpoints = checkpoint_manager.list_checkpoints(agent_role)
        stats = checkpoint_manager.get_checkpoint_stats()
        
        return {
            "status": "success",
            "checkpoints": checkpoints,
            "stats": stats
        }
    except Exception as e:
        return {"error": f"Failed to list checkpoints: {str(e)}"}

@app.post("/api/ai/save-checkpoint")
async def save_checkpoint(request: dict):
    """Save a model checkpoint"""
    if not LLM_TRAINER_AVAILABLE:
        return {"error": "LLM trainer not available"}
    
    try:
        agent_role = request.get("agent_role", "unknown")
        trainer = get_llm_trainer()
        checkpoint_id = trainer.save_checkpoint(agent_role=agent_role)
        
        return {
            "status": "success",
            "checkpoint_id": checkpoint_id,
            "message": f"Saved checkpoint for {agent_role}"
        }
    except Exception as e:
        return {"error": f"Failed to save checkpoint: {str(e)}"}

@app.post("/api/ai/load-checkpoint")
async def load_checkpoint(request: dict):
    """Load a model checkpoint"""
    if not LLM_TRAINER_AVAILABLE:
        return {"error": "LLM trainer not available"}
    
    try:
        checkpoint_id = request.get("checkpoint_id")
        agent_role = request.get("agent_role")
        
        trainer = get_llm_trainer()
        success = trainer.load_checkpoint(checkpoint_id, agent_role)
        
        if success:
            return {
                "status": "success",
                "message": f"Loaded checkpoint {checkpoint_id or 'latest'} for {agent_role or 'current role'}"
            }
        else:
            return {"error": "Failed to load checkpoint"}
    except Exception as e:
        return {"error": f"Failed to load checkpoint: {str(e)}"}

@app.get("/api/ai/checkpoint-info")
async def get_checkpoint_info():
    """Get checkpoint information and statistics"""
    if not LLM_TRAINER_AVAILABLE:
        return {"error": "LLM trainer not available"}
    
    try:
        trainer = get_llm_trainer()
        info = trainer.get_checkpoint_info()
        return {"status": "success", "info": info}
    except Exception as e:
        return {"error": f"Failed to get checkpoint info: {str(e)}"}

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get AI performance metrics."""
    return {
        "total_games": training_state["total_training_games"],
        "total_sessions": training_state["total_training_sessions"],
        "win_rates": {
            "liberal": 0.52,
            "fascist": 0.35,
            "hitler": 0.13
        },
        "average_game_duration": 18.5,
        "agent_performance": [
            {"agent_id": f"agent_{i}", "win_rate": 0.45 + (i * 0.02)} 
            for i in range(training_state["agent_count"])
        ]
    }

async def simulate_training():
    """Simulate training sessions for demo purposes."""
    while training_state["training_active"]:
        interval = training_state["current_config"]["training_interval_minutes"]
        games_per_session = training_state["current_config"]["games_per_session"]
        
        # Wait for training interval
        await asyncio.sleep(interval * 60)  # Convert to seconds
        
        if not training_state["training_active"]:
            break
        
        # Simulate training session
        training_state["total_training_sessions"] += 1
        training_state["total_training_games"] += games_per_session
        training_state["last_training_time"] = datetime.now().isoformat()
        
        # Calculate next training time
        next_time = datetime.now() + timedelta(minutes=interval)
        training_state["next_training_time"] = next_time.isoformat()
        
        print(f"Completed training session {training_state['total_training_sessions']} "
              f"with {games_per_session} games")

# Game endpoints (simplified)
@app.post("/api/games")
async def create_game():
    """Create a new game."""
    return {"game_id": "demo_game_123", "message": "Game created"}

@app.post("/api/games/{game_id}/join")
async def join_game(game_id: str):
    """Join a game."""
    return {"message": f"Joined game {game_id}"}

@app.get("/api/games/{game_id}/players")
async def get_players(game_id: str):
    """Get players in a game."""
    return {"players": ["Player1", "AI_Bot1", "AI_Bot2"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
