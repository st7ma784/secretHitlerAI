from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import uuid

class Role(Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"
    HITLER = "hitler"

class Party(Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"

class GamePhase(Enum):
    LOBBY = "lobby"
    ROLE_ASSIGNMENT = "role_assignment"
    ELECTION = "election"
    LEGISLATIVE = "legislative"
    EXECUTIVE = "executive"
    GAME_OVER = "game_over"

class VoteType(Enum):
    JA = "ja"
    NEIN = "nein"

@dataclass
class Player:
    id: str
    name: str
    role: Optional[Role] = None
    party: Optional[Party] = None
    is_alive: bool = True
    is_president: bool = False
    is_chancellor: bool = False
    is_previous_president: bool = False
    is_previous_chancellor: bool = False

@dataclass
class Policy:
    type: Party
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Vote:
    player_id: str
    vote: VoteType

class SecretHitlerRules:
    def __init__(self):
        self.min_players = 5
        self.max_players = 10
        
        # Victory conditions
        self.liberal_policies_to_win = 5
        self.fascist_policies_to_win = 6
        
        # Board setup based on player count
        self.fascist_board_powers = {
            5: {3: "policy_peek", 4: "execution", 5: "execution"},
            6: {3: "policy_peek", 4: "execution", 5: "execution"},
            7: {2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
            8: {2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
            9: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execution", 5: "execution"},
            10: {1: "investigate", 2: "investigate", 3: "special_election", 4: "execution", 5: "execution"}
        }
    
    def get_role_distribution(self, player_count: int) -> Dict[Role, int]:
        if player_count < self.min_players or player_count > self.max_players:
            raise ValueError(f"Invalid player count: {player_count}")
        
        distributions = {
            5: {Role.LIBERAL: 3, Role.FASCIST: 1, Role.HITLER: 1},
            6: {Role.LIBERAL: 4, Role.FASCIST: 1, Role.HITLER: 1},
            7: {Role.LIBERAL: 4, Role.FASCIST: 2, Role.HITLER: 1},
            8: {Role.LIBERAL: 5, Role.FASCIST: 2, Role.HITLER: 1},
            9: {Role.LIBERAL: 5, Role.FASCIST: 3, Role.HITLER: 1},
            10: {Role.LIBERAL: 6, Role.FASCIST: 3, Role.HITLER: 1}
        }
        return distributions[player_count]
    
    def can_nominate_chancellor(self, president_id: str, chancellor_id: str, 
                               players: List[Player]) -> bool:
        president = next((p for p in players if p.id == president_id), None)
        chancellor = next((p for p in players if p.id == chancellor_id), None)
        
        if not president or not chancellor:
            return False
        
        # Cannot nominate yourself
        if president_id == chancellor_id:
            return False
        
        # Cannot nominate dead players
        if not chancellor.is_alive:
            return False
        
        # Cannot nominate previous president or chancellor if 5+ players alive
        alive_players = [p for p in players if p.is_alive]
        if len(alive_players) >= 5:
            if chancellor.is_previous_president or chancellor.is_previous_chancellor:
                return False
        
        return True
    
    def get_election_result(self, votes: List[Vote], alive_players: List[Player]) -> bool:
        ja_votes = sum(1 for vote in votes if vote.vote == VoteType.JA)
        total_votes = len(alive_players)
        return ja_votes > total_votes // 2
    
    def get_fascist_power(self, player_count: int, fascist_policies: int) -> Optional[str]:
        if player_count not in self.fascist_board_powers:
            return None
        return self.fascist_board_powers[player_count].get(fascist_policies)
    
    def check_victory_condition(self, liberal_policies: int, fascist_policies: int,
                               hitler_is_chancellor: bool, hitler_is_dead: bool) -> Optional[Party]:
        # Liberal victory conditions
        if liberal_policies >= self.liberal_policies_to_win:
            return Party.LIBERAL
        if hitler_is_dead:
            return Party.LIBERAL
        
        # Fascist victory conditions
        if fascist_policies >= self.fascist_policies_to_win:
            return Party.FASCIST
        if hitler_is_chancellor and fascist_policies >= 3:
            return Party.FASCIST
        
        return None
    
    def create_policy_deck(self) -> List[Policy]:
        policies = []
        # 6 Liberal policies
        for _ in range(6):
            policies.append(Policy(type=Party.LIBERAL))
        # 11 Fascist policies
        for _ in range(11):
            policies.append(Policy(type=Party.FASCIST))
        return policies