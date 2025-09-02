from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
import random
import uuid
from datetime import datetime

from .rules import (
    SecretHitlerRules, Player, Policy, Vote, Role, Party, 
    GamePhase, VoteType
)

@dataclass
class GameAction:
    id: str
    player_id: str
    action_type: str
    data: Dict
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GameState:
    id: str
    phase: GamePhase
    players: List[Player]
    policy_deck: List[Policy]
    discard_pile: List[Policy]
    liberal_policies: int = 0
    fascist_policies: int = 0
    election_tracker: int = 0
    current_president_id: Optional[str] = None
    nominated_chancellor_id: Optional[str] = None
    votes: List[Vote] = field(default_factory=list)
    drawn_policies: List[Policy] = field(default_factory=list)
    winner: Optional[Party] = None
    history: List[GameAction] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class GameStateManager:
    def __init__(self):
        self.rules = SecretHitlerRules()
        self.games: Dict[str, GameState] = {}
    
    def create_game(self, game_id: Optional[str] = None) -> str:
        if not game_id:
            game_id = str(uuid.uuid4())
        
        game_state = GameState(
            id=game_id,
            phase=GamePhase.LOBBY,
            players=[],
            policy_deck=self.rules.create_policy_deck(),
            discard_pile=[]
        )
        
        # Shuffle the policy deck
        random.shuffle(game_state.policy_deck)
        
        self.games[game_id] = game_state
        return game_id
    
    def add_player(self, game_id: str, player_name: str) -> str:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if game.phase != GamePhase.LOBBY:
            raise ValueError("Cannot add players after game has started")
        
        if len(game.players) >= self.rules.max_players:
            raise ValueError("Game is full")
        
        # Check if name is already taken
        if any(p.name == player_name for p in game.players):
            raise ValueError("Player name already taken")
        
        player_id = str(uuid.uuid4())
        player = Player(id=player_id, name=player_name)
        game.players.append(player)
        
        self._log_action(game_id, "system", "player_joined", {"player_id": player_id, "name": player_name})
        
        return player_id
    
    def start_game(self, game_id: str) -> bool:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if len(game.players) < self.rules.min_players:
            raise ValueError(f"Need at least {self.rules.min_players} players to start")
        
        if game.phase != GamePhase.LOBBY:
            raise ValueError("Game already started")
        
        # Assign roles
        self._assign_roles(game)
        
        # Set first president
        game.current_president_id = random.choice(game.players).id
        
        game.phase = GamePhase.ELECTION
        self._log_action(game_id, "system", "game_started", {"player_count": len(game.players)})
        
        return True
    
    def nominate_chancellor(self, game_id: str, president_id: str, chancellor_id: str) -> bool:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if game.phase != GamePhase.ELECTION:
            raise ValueError("Not in election phase")
        
        if game.current_president_id != president_id:
            raise ValueError("You are not the current president")
        
        if not self.rules.can_nominate_chancellor(president_id, chancellor_id, game.players):
            raise ValueError("Cannot nominate this player as chancellor")
        
        game.nominated_chancellor_id = chancellor_id
        game.votes = []
        
        self._log_action(game_id, president_id, "nominate_chancellor", {"chancellor_id": chancellor_id})
        
        return True
    
    def cast_vote(self, game_id: str, player_id: str, vote: VoteType) -> bool:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if game.phase != GamePhase.ELECTION:
            raise ValueError("Not in voting phase")
        
        player = next((p for p in game.players if p.id == player_id), None)
        if not player or not player.is_alive:
            raise ValueError("Player not found or not alive")
        
        # Check if player already voted
        if any(v.player_id == player_id for v in game.votes):
            raise ValueError("Player already voted")
        
        game.votes.append(Vote(player_id=player_id, vote=vote))
        
        self._log_action(game_id, player_id, "vote_cast", {"vote": vote.value})
        
        # Check if all alive players have voted
        alive_players = [p for p in game.players if p.is_alive]
        if len(game.votes) == len(alive_players):
            self._resolve_election(game)
        
        return True
    
    def president_discard_policy(self, game_id: str, player_id: str, policy_id: str) -> bool:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if game.phase != GamePhase.LEGISLATIVE:
            raise ValueError("Not in legislative phase")
        
        if game.current_president_id != player_id:
            raise ValueError("You are not the current president")
        
        if len(game.drawn_policies) != 3:
            raise ValueError("President should have 3 policies")
        
        # Find and remove the discarded policy
        policy_to_discard = next((p for p in game.drawn_policies if p.id == policy_id), None)
        if not policy_to_discard:
            raise ValueError("Policy not found")
        
        game.drawn_policies.remove(policy_to_discard)
        game.discard_pile.append(policy_to_discard)
        
        self._log_action(game_id, player_id, "president_discard", {"policy_type": policy_to_discard.type.value})
        
        return True
    
    def chancellor_enact_policy(self, game_id: str, player_id: str, policy_id: str) -> bool:
        game = self.games.get(game_id)
        if not game:
            raise ValueError(f"Game {game_id} not found")
        
        if game.phase != GamePhase.LEGISLATIVE:
            raise ValueError("Not in legislative phase")
        
        if game.nominated_chancellor_id != player_id:
            raise ValueError("You are not the current chancellor")
        
        if len(game.drawn_policies) != 2:
            raise ValueError("Chancellor should have 2 policies")
        
        # Find the enacted policy
        enacted_policy = next((p for p in game.drawn_policies if p.id == policy_id), None)
        if not enacted_policy:
            raise ValueError("Policy not found")
        
        # Remove enacted policy and discard the other
        game.drawn_policies.remove(enacted_policy)
        game.discard_pile.extend(game.drawn_policies)
        game.drawn_policies = []
        
        # Update policy counts
        if enacted_policy.type == Party.LIBERAL:
            game.liberal_policies += 1
        else:
            game.fascist_policies += 1
        
        self._log_action(game_id, player_id, "enact_policy", {"policy_type": enacted_policy.type.value})
        
        # Check for victory conditions
        winner = self.rules.check_victory_condition(
            game.liberal_policies, 
            game.fascist_policies,
            self._is_hitler_chancellor(game),
            self._is_hitler_dead(game)
        )
        
        if winner:
            game.winner = winner
            game.phase = GamePhase.GAME_OVER
            self._log_action(game_id, "system", "game_over", {"winner": winner.value})
        else:
            # Check for executive power
            power = self.rules.get_fascist_power(len(game.players), game.fascist_policies)
            if power and enacted_policy.type == Party.FASCIST:
                game.phase = GamePhase.EXECUTIVE
            else:
                self._advance_to_next_election(game)
        
        return True
    
    def get_game_state(self, game_id: str) -> Optional[GameState]:
        return self.games.get(game_id)
    
    def get_player_view(self, game_id: str, player_id: str) -> Dict:
        game = self.games.get(game_id)
        if not game:
            return {}
        
        player = next((p for p in game.players if p.id == player_id), None)
        if not player:
            return {}
        
        # Base game state visible to all players
        view = {
            "game_id": game.id,
            "phase": game.phase.value,
            "liberal_policies": game.liberal_policies,
            "fascist_policies": game.fascist_policies,
            "election_tracker": game.election_tracker,
            "current_president_id": game.current_president_id,
            "nominated_chancellor_id": game.nominated_chancellor_id,
            "winner": game.winner.value if game.winner else None,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "is_alive": p.is_alive,
                    "is_president": p.is_president,
                    "is_chancellor": p.is_chancellor
                }
                for p in game.players
            ]
        }
        
        # Add private information for the player
        view["your_role"] = player.role.value if player.role else None
        view["your_party"] = player.party.value if player.party else None
        
        # Add fascist team information for fascists and hitler
        if player.party == Party.FASCIST:
            fascists = [p for p in game.players if p.party == Party.FASCIST]
            view["fascist_team"] = [{"id": p.id, "name": p.name, "role": p.role.value} for p in fascists]
        
        # Add drawn policies if player is president or chancellor
        if (game.current_president_id == player_id and len(game.drawn_policies) > 0) or \
           (game.nominated_chancellor_id == player_id and len(game.drawn_policies) == 2):
            view["drawn_policies"] = [{"id": p.id, "type": p.type.value} for p in game.drawn_policies]
        
        return view
    
    def _assign_roles(self, game: GameState):
        role_distribution = self.rules.get_role_distribution(len(game.players))
        
        roles = []
        for role, count in role_distribution.items():
            roles.extend([role] * count)
        
        random.shuffle(roles)
        
        for i, player in enumerate(game.players):
            player.role = roles[i]
            player.party = Party.LIBERAL if roles[i] == Role.LIBERAL else Party.FASCIST
    
    def _resolve_election(self, game: GameState):
        alive_players = [p for p in game.players if p.is_alive]
        election_passed = self.rules.get_election_result(game.votes, alive_players)
        
        if election_passed:
            # Government elected
            game.election_tracker = 0
            
            # Update president and chancellor status
            for player in game.players:
                player.is_previous_president = player.is_president
                player.is_previous_chancellor = player.is_chancellor
                player.is_president = False
                player.is_chancellor = False
            
            president = next(p for p in game.players if p.id == game.current_president_id)
            chancellor = next(p for p in game.players if p.id == game.nominated_chancellor_id)
            
            president.is_president = True
            chancellor.is_chancellor = True
            
            # Draw 3 policies for president
            if len(game.policy_deck) < 3:
                self._reshuffle_deck(game)
            
            game.drawn_policies = game.policy_deck[:3]
            game.policy_deck = game.policy_deck[3:]
            
            game.phase = GamePhase.LEGISLATIVE
            
            self._log_action(game.id, "system", "election_passed", {
                "president_id": game.current_president_id,
                "chancellor_id": game.nominated_chancellor_id
            })
        else:
            # Election failed
            game.election_tracker += 1
            game.nominated_chancellor_id = None
            game.votes = []
            
            if game.election_tracker >= 3:
                # Chaos - enact top policy
                if len(game.policy_deck) < 1:
                    self._reshuffle_deck(game)
                
                chaos_policy = game.policy_deck.pop(0)
                if chaos_policy.type == Party.LIBERAL:
                    game.liberal_policies += 1
                else:
                    game.fascist_policies += 1
                
                game.election_tracker = 0
                
                self._log_action(game.id, "system", "chaos_policy", {"policy_type": chaos_policy.type.value})
                
                # Check victory conditions
                winner = self.rules.check_victory_condition(
                    game.liberal_policies, 
                    game.fascist_policies,
                    self._is_hitler_chancellor(game),
                    self._is_hitler_dead(game)
                )
                
                if winner:
                    game.winner = winner
                    game.phase = GamePhase.GAME_OVER
                    return
            
            self._advance_to_next_election(game)
            
            self._log_action(game.id, "system", "election_failed", {"election_tracker": game.election_tracker})
    
    def _advance_to_next_election(self, game: GameState):
        # Find next president
        alive_players = [p for p in game.players if p.is_alive]
        current_president_index = next(i for i, p in enumerate(alive_players) if p.id == game.current_president_id)
        next_president_index = (current_president_index + 1) % len(alive_players)
        game.current_president_id = alive_players[next_president_index].id
        
        game.nominated_chancellor_id = None
        game.votes = []
        game.phase = GamePhase.ELECTION
    
    def _reshuffle_deck(self, game: GameState):
        game.policy_deck.extend(game.discard_pile)
        game.discard_pile = []
        random.shuffle(game.policy_deck)
    
    def _is_hitler_chancellor(self, game: GameState) -> bool:
        if not game.nominated_chancellor_id:
            return False
        chancellor = next((p for p in game.players if p.id == game.nominated_chancellor_id), None)
        return chancellor and chancellor.role == Role.HITLER and chancellor.is_chancellor
    
    def _is_hitler_dead(self, game: GameState) -> bool:
        hitler = next((p for p in game.players if p.role == Role.HITLER), None)
        return hitler and not hitler.is_alive
    
    def _log_action(self, game_id: str, player_id: str, action_type: str, data: Dict):
        game = self.games[game_id]
        action = GameAction(
            id=str(uuid.uuid4()),
            player_id=player_id,
            action_type=action_type,
            data=data
        )
        game.history.append(action)
        game.updated_at = datetime.now()