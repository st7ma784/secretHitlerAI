from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import timedelta
import json

from ..game.state import GameStateManager
from ..game.rules import VoteType
from ..models.ai_agent import ai_manager
from ..training.self_trainer import training_orchestrator

app = FastAPI(title="Secret Hitler AI Game API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Game state manager
game_manager = GameStateManager()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            self.active_connections[game_id].remove(websocket)
    
    async def broadcast_to_game(self, game_id: str, message: dict):
        if game_id in self.active_connections:
            connections = self.active_connections[game_id].copy()
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    # Remove disconnected websockets
                    self.active_connections[game_id].remove(connection)

manager = ConnectionManager()

# Pydantic models for API requests
class CreateGameRequest(BaseModel):
    game_id: Optional[str] = None

class JoinGameRequest(BaseModel):
    player_name: str

class NominateChancellorRequest(BaseModel):
    chancellor_id: str

class VoteRequest(BaseModel):
    vote: str  # "ja" or "nein"

class DiscardPolicyRequest(BaseModel):
    policy_id: str

class EnactPolicyRequest(BaseModel):
    policy_id: str

class AddAIPlayerRequest(BaseModel):
    difficulty: str = "medium"  # easy, medium, hard
    enable_training: bool = True

# API Endpoints
@app.post("/api/games")
async def create_game(request: CreateGameRequest):
    try:
        game_id = game_manager.create_game(request.game_id)
        return {"game_id": game_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/join")
async def join_game(game_id: str, request: JoinGameRequest):
    try:
        player_id = game_manager.add_player(game_id, request.player_name)
        
        # Broadcast player joined event
        await manager.broadcast_to_game(game_id, {
            "type": "player_joined",
            "player_id": player_id,
            "player_name": request.player_name
        })
        
        return {"player_id": player_id, "status": "joined"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/start")
async def start_game(game_id: str):
    try:
        game_manager.start_game(game_id)
        
        # Broadcast game started event
        await manager.broadcast_to_game(game_id, {
            "type": "game_started"
        })
        
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/nominate")
async def nominate_chancellor(game_id: str, request: NominateChancellorRequest, player_id: str):
    try:
        game_manager.nominate_chancellor(game_id, player_id, request.chancellor_id)
        
        # Broadcast nomination event
        await manager.broadcast_to_game(game_id, {
            "type": "chancellor_nominated",
            "president_id": player_id,
            "chancellor_id": request.chancellor_id
        })
        
        return {"status": "nominated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/vote")
async def cast_vote(game_id: str, request: VoteRequest, player_id: str):
    try:
        vote_type = VoteType.JA if request.vote.lower() == "ja" else VoteType.NEIN
        game_manager.cast_vote(game_id, player_id, vote_type)
        
        # Broadcast vote cast event (without revealing the vote)
        await manager.broadcast_to_game(game_id, {
            "type": "vote_cast",
            "player_id": player_id
        })
        
        # Check if voting is complete
        game_state = game_manager.get_game_state(game_id)
        if game_state:
            alive_players = [p for p in game_state.players if p.is_alive]
            if len(game_state.votes) == len(alive_players):
                await manager.broadcast_to_game(game_id, {
                    "type": "election_resolved"
                })
        
        return {"status": "voted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/discard")
async def discard_policy(game_id: str, request: DiscardPolicyRequest, player_id: str):
    try:
        game_manager.president_discard_policy(game_id, player_id, request.policy_id)
        
        # Broadcast policy discarded event
        await manager.broadcast_to_game(game_id, {
            "type": "policy_discarded",
            "president_id": player_id
        })
        
        return {"status": "discarded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/games/{game_id}/enact")
async def enact_policy(game_id: str, request: EnactPolicyRequest, player_id: str):
    try:
        game_manager.chancellor_enact_policy(game_id, player_id, request.policy_id)
        
        # Broadcast policy enacted event
        game_state = game_manager.get_game_state(game_id)
        await manager.broadcast_to_game(game_id, {
            "type": "policy_enacted",
            "chancellor_id": player_id,
            "liberal_policies": game_state.liberal_policies,
            "fascist_policies": game_state.fascist_policies
        })
        
        # Check for game over
        if game_state.winner:
            await manager.broadcast_to_game(game_id, {
                "type": "game_over",
                "winner": game_state.winner.value
            })
        
        return {"status": "enacted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/games/{game_id}/state")
async def get_game_state(game_id: str, player_id: str):
    try:
        player_view = game_manager.get_player_view(game_id, player_id)
        if not player_view:
            raise HTTPException(status_code=404, detail="Game or player not found")
        return player_view
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/games/{game_id}/players")
async def get_players(game_id: str):
    try:
        game_state = game_manager.get_game_state(game_id)
        if not game_state:
            raise HTTPException(status_code=404, detail="Game not found")
        
        return {
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "is_alive": p.is_alive,
                    "is_president": p.is_president,
                    "is_chancellor": p.is_chancellor
                }
                for p in game_state.players
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint
@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await manager.connect(websocket, game_id)
    try:
        while True:
            # Keep connection alive and handle any client messages
            data = await websocket.receive_text()
            # For now, just echo back or handle specific commands
            message = json.loads(data)
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket, game_id)

# AI Player Management Endpoints
@app.post("/api/games/{game_id}/add-ai")
async def add_ai_player(game_id: str, request: AddAIPlayerRequest):
    try:
        # Create AI agent
        agent_id = f"ai_{game_id}_{len(ai_manager.ai_agents)}"
        ai_agent = ai_manager.create_ai_agent(
            agent_id=agent_id,
            difficulty=request.difficulty,
            enable_self_training=request.enable_training
        )
        
        # Add AI to game
        player_id = game_manager.add_player(game_id, f"AI-{request.difficulty}")
        
        # Connect AI to game
        success = await ai_manager.add_ai_to_game(game_id, agent_id, player_id)
        
        if success:
            # Broadcast AI player joined
            await manager.broadcast_to_game(game_id, {
                "type": "ai_player_joined",
                "player_id": player_id,
                "player_name": f"AI-{request.difficulty}",
                "difficulty": request.difficulty
            })
            
            return {
                "player_id": player_id,
                "agent_id": agent_id,
                "difficulty": request.difficulty,
                "status": "added"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to add AI to game")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/ai/performance")
async def get_ai_performance():
    """Get performance metrics for all AI agents"""
    try:
        metrics = ai_manager.get_all_performance_metrics()
        training_status = training_orchestrator.get_training_status()
        
        return {
            "agent_metrics": metrics,
            "training_status": training_status,
            "summary": training_orchestrator.get_agent_performance_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/start-training")
async def start_ai_training():
    """Start continuous AI training"""
    try:
        training_orchestrator.start_continuous_training()
        return {"status": "training_started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/stop-training")
async def stop_ai_training():
    """Stop continuous AI training"""
    try:
        training_orchestrator.stop_continuous_training()
        return {"status": "training_stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/training-status")
async def get_training_status():
    """Get current training status"""
    try:
        status = training_orchestrator.get_training_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/configure-training")
async def configure_training(config: dict):
    """Configure training parameters"""
    try:
        # Update training configuration
        if 'num_agents' in config:
            training_orchestrator.num_training_agents = config['num_agents']
        if 'games_per_session' in config:
            training_orchestrator.games_per_session = config['games_per_session']
        if 'training_interval_minutes' in config:
            training_orchestrator.training_interval = timedelta(minutes=config['training_interval_minutes'])
        if 'enable_live_learning' in config:
            training_orchestrator.enable_live_learning = config['enable_live_learning']
        
        return {"status": "configuration_updated", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced game action handlers with AI integration
async def handle_ai_action(game_id: str, player_id: str, action_type: str, valid_targets=None):
    """Handle AI player actions"""
    try:
        # Get game state for AI decision
        game_state = game_manager.get_player_view(game_id, player_id)
        if not game_state:
            return None
        
        # Make AI decision
        action, confidence, reasoning = await ai_manager.make_ai_decision(
            game_id, player_id, game_state, action_type, valid_targets
        )
        
        if action is not None:
            # Execute the AI action
            success = False
            error = None
            
            try:
                if action_type == "nominate":
                    game_manager.nominate_chancellor(game_id, player_id, action)
                    success = True
                elif action_type == "vote":
                    vote_type = VoteType.JA if action == "ja" else VoteType.NEIN
                    game_manager.cast_vote(game_id, player_id, vote_type)
                    success = True
                elif action_type == "discard":
                    game_manager.president_discard_policy(game_id, player_id, action)
                    success = True
                elif action_type == "enact":
                    game_manager.chancellor_enact_policy(game_id, player_id, action)
                    success = True
            except Exception as e:
                error = str(e)
            
            # Provide feedback to AI
            new_game_state = game_manager.get_player_view(game_id, player_id)
            await ai_manager.notify_ai_feedback(game_id, player_id, new_game_state, success)
            
            # Add to training data
            if success:
                game_data = {
                    'game_id': game_id,
                    'action_type': action_type,
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'player_id': player_id,
                    'game_state': game_state
                }
                training_orchestrator.add_live_game_data(game_data)
            
            return {
                'success': success,
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'error': error
            }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}
    
    return None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Initialize training on startup
@app.on_event("startup")
async def startup_event():
    """Initialize AI training on server startup"""
    try:
        # Start the training orchestrator
        training_orchestrator.start_continuous_training()
        print("AI training system initialized")
    except Exception as e:
        print(f"Error initializing AI training: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    try:
        training_orchestrator.shutdown()
        ai_manager.shutdown_all()
        print("AI systems shutdown complete")
    except Exception as e:
        print(f"Error during shutdown: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)