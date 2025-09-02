import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    hidden_size: int = 768
    max_sequence_length: int = 512
    temperature: float = 0.8
    top_p: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class LamaModel(nn.Module):
    """
    LAMA (Language-Augmented Model Architecture) for Secret Hitler AI
    Combines language understanding with game-specific reasoning
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Base language model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.language_model = AutoModel.from_pretrained(config.model_name)
        
        # Add special tokens for game concepts
        special_tokens = [
            "<LIBERAL>", "<FASCIST>", "<HITLER>", 
            "<PRESIDENT>", "<CHANCELLOR>", "<POLICY>",
            "<VOTE_JA>", "<VOTE_NEIN>", "<NOMINATE>",
            "<INVESTIGATE>", "<EXECUTE>", "<PEEK>",
            "<GAME_STATE>", "<ACTION>", "<REASONING>"
        ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        
        # Game-specific layers
        self.game_state_encoder = nn.Linear(config.hidden_size, config.hidden_size)
        self.action_decoder = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Action heads for different game actions
        self.nomination_head = nn.Linear(config.hidden_size, 10)  # Max 10 players
        self.vote_head = nn.Linear(config.hidden_size, 2)  # JA/NEIN
        self.policy_head = nn.Linear(config.hidden_size, 3)  # Choose from 2-3 policies
        self.investigation_head = nn.Linear(config.hidden_size, 10)  # Investigate player
        self.execution_head = nn.Linear(config.hidden_size, 10)  # Execute player
        
        # Value estimation for RL
        self.value_head = nn.Linear(config.hidden_size, 1)
        
        # Role-specific reasoning
        self.role_embeddings = nn.Embedding(3, config.hidden_size)  # Liberal, Fascist, Hitler
        
        self.dropout = nn.Dropout(0.1)
        
    def encode_game_state(self, game_state: Dict, player_perspective: str) -> str:
        """Convert game state to text representation for the model"""
        
        # Create textual description of game state
        text_parts = ["<GAME_STATE>"]
        
        # Basic game info
        text_parts.append(f"Liberal policies: {game_state.get('liberal_policies', 0)}/5")
        text_parts.append(f"Fascist policies: {game_state.get('fascist_policies', 0)}/6")
        text_parts.append(f"Election tracker: {game_state.get('election_tracker', 0)}/3")
        text_parts.append(f"Phase: {game_state.get('phase', 'unknown')}")
        
        # Player information
        players = game_state.get('players', [])
        text_parts.append(f"Players ({len(players)}):")
        
        for player in players:
            status = []
            if player.get('is_president'):
                status.append("PRESIDENT")
            if player.get('is_chancellor'):
                status.append("CHANCELLOR")
            if not player.get('is_alive'):
                status.append("DEAD")
            
            status_str = f" ({', '.join(status)})" if status else ""
            text_parts.append(f"- {player['name']}{status_str}")
        
        # Current government
        if game_state.get('current_president_id'):
            president = next((p for p in players if p['id'] == game_state['current_president_id']), None)
            if president:
                text_parts.append(f"Current president: {president['name']}")
        
        if game_state.get('nominated_chancellor_id'):
            chancellor = next((p for p in players if p['id'] == game_state['nominated_chancellor_id']), None)
            if chancellor:
                text_parts.append(f"Nominated chancellor: {chancellor['name']}")
        
        # Player's private information
        if player_perspective:
            role = game_state.get('your_role')
            party = game_state.get('your_party')
            if role:
                text_parts.append(f"Your role: <{role.upper()}> ({party})")
            
            # Fascist team knowledge
            fascist_team = game_state.get('fascist_team', [])
            if fascist_team:
                team_names = [f"{p['name']} ({p['role']})" for p in fascist_team]
                text_parts.append(f"Fascist team: {', '.join(team_names)}")
        
        # Available policies (if any)
        drawn_policies = game_state.get('drawn_policies', [])
        if drawn_policies:
            policy_types = [p['type'] for p in drawn_policies]
            text_parts.append(f"Available policies: {', '.join(policy_types)}")
        
        return " ".join(text_parts)
    
    def forward(
        self, 
        game_state_text: str, 
        role: Optional[int] = None,
        action_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            game_state_text: Textual representation of game state
            role: Player role (0=Liberal, 1=Fascist, 2=Hitler)
            action_type: Type of action to predict
        """
        
        # Tokenize input
        inputs = self.tokenizer(
            game_state_text,
            return_tensors="pt",
            max_length=self.config.max_sequence_length,
            truncation=True,
            padding=True
        ).to(self.config.device)
        
        # Get language model embeddings
        outputs = self.language_model(**inputs)
        sequence_output = outputs.last_hidden_state
        
        # Pool to get single representation
        pooled_output = sequence_output.mean(dim=1)  # Average pooling
        
        # Add role information if available
        if role is not None:
            role_tensor = torch.tensor([role], device=self.config.device)
            role_embedding = self.role_embeddings(role_tensor)
            pooled_output = pooled_output + role_embedding
        
        # Apply game-specific encoding
        game_encoding = self.game_state_encoder(pooled_output)
        game_encoding = F.relu(game_encoding)
        game_encoding = self.dropout(game_encoding)
        
        # Decode to action space
        action_encoding = self.action_decoder(game_encoding)
        
        results = {}
        
        # Generate action logits based on action type
        if action_type == "nominate" or action_type is None:
            results["nomination_logits"] = self.nomination_head(action_encoding)
        
        if action_type == "vote" or action_type is None:
            results["vote_logits"] = self.vote_head(action_encoding)
        
        if action_type == "policy" or action_type is None:
            results["policy_logits"] = self.policy_head(action_encoding)
        
        if action_type == "investigate" or action_type is None:
            results["investigation_logits"] = self.investigation_head(action_encoding)
        
        if action_type == "execute" or action_type is None:
            results["execution_logits"] = self.execution_head(action_encoding)
        
        # Always compute value for RL
        results["value"] = self.value_head(action_encoding)
        
        # Add hidden state for further processing
        results["hidden_state"] = action_encoding
        
        return results
    
    def generate_reasoning(self, game_state_text: str, max_length: int = 100) -> str:
        """Generate reasoning text for the current game state"""
        
        reasoning_prompt = f"{game_state_text} <REASONING>"
        
        inputs = self.tokenizer.encode(reasoning_prompt, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Extract reasoning part
        if "<REASONING>" in generated_text:
            reasoning = generated_text.split("<REASONING>")[1].strip()
            return reasoning
        
        return "No reasoning generated"
    
    def predict_action(
        self, 
        game_state: Dict, 
        player_id: str,
        action_type: str,
        valid_targets: Optional[List[str]] = None
    ) -> Tuple[Any, float, str]:
        """
        Predict the best action for a given game state
        
        Returns:
            action: The predicted action
            confidence: Confidence score (0-1)
            reasoning: Text explanation of the decision
        """
        
        # Get player role
        role_map = {"liberal": 0, "fascist": 1, "hitler": 2}
        role = role_map.get(game_state.get("your_role"), 0)
        
        # Encode game state
        game_state_text = self.encode_game_state(game_state, player_id)
        
        # Generate reasoning
        reasoning = self.generate_reasoning(game_state_text)
        
        # Get action predictions
        with torch.no_grad():
            outputs = self.forward(game_state_text, role, action_type)
        
        if action_type == "nominate":
            logits = outputs["nomination_logits"].squeeze()
            
            # Mask invalid targets
            if valid_targets:
                mask = torch.full_like(logits, float('-inf'))
                players = game_state.get('players', [])
                for i, player in enumerate(players[:len(logits)]):
                    if player['id'] in valid_targets:
                        mask[i] = 0
                logits = logits + mask
            
            probabilities = F.softmax(logits, dim=0)
            action_idx = torch.argmax(probabilities).item()
            confidence = probabilities[action_idx].item()
            
            # Map back to player ID
            players = game_state.get('players', [])
            if action_idx < len(players):
                action = players[action_idx]['id']
            else:
                action = None
                
        elif action_type == "vote":
            logits = outputs["vote_logits"].squeeze()
            probabilities = F.softmax(logits, dim=0)
            action_idx = torch.argmax(probabilities).item()
            confidence = probabilities[action_idx].item()
            action = "ja" if action_idx == 0 else "nein"
            
        elif action_type == "policy":
            logits = outputs["policy_logits"].squeeze()
            
            # Only consider available policies
            drawn_policies = game_state.get('drawn_policies', [])
            if drawn_policies:
                mask = torch.full_like(logits, float('-inf'))
                for i in range(min(len(drawn_policies), len(logits))):
                    mask[i] = 0
                logits = logits + mask
            
            probabilities = F.softmax(logits, dim=0)
            action_idx = torch.argmax(probabilities).item()
            confidence = probabilities[action_idx].item()
            
            # Map to policy ID
            if drawn_policies and action_idx < len(drawn_policies):
                action = drawn_policies[action_idx]['id']
            else:
                action = None
                
        else:
            action = None
            confidence = 0.0
        
        return action, confidence, reasoning
    
    def get_state_value(self, game_state: Dict, player_id: str) -> float:
        """Estimate the value of the current game state for the player"""
        
        role_map = {"liberal": 0, "fascist": 1, "hitler": 2}
        role = role_map.get(game_state.get("your_role"), 0)
        
        game_state_text = self.encode_game_state(game_state, player_id)
        
        with torch.no_grad():
            outputs = self.forward(game_state_text, role)
            value = outputs["value"].squeeze().item()
        
        return value

class LamaModelManager:
    """Manages multiple LAMA models for different AI agents"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, LamaModel] = {}
        
    def create_agent_model(self, agent_id: str) -> LamaModel:
        """Create a new model instance for an AI agent"""
        model = LamaModel(self.config).to(self.config.device)
        self.models[agent_id] = model
        return model
    
    def get_model(self, agent_id: str) -> Optional[LamaModel]:
        """Get model for a specific agent"""
        return self.models.get(agent_id)
    
    def save_model(self, agent_id: str, path: str):
        """Save agent model to disk"""
        if agent_id in self.models:
            torch.save(self.models[agent_id].state_dict(), path)
    
    def load_model(self, agent_id: str, path: str):
        """Load agent model from disk"""
        if agent_id not in self.models:
            self.models[agent_id] = LamaModel(self.config).to(self.config.device)
        
        self.models[agent_id].load_state_dict(torch.load(path, map_location=self.config.device))