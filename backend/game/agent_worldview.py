"""
Agent World View System for Secret Hitler AI
Maintains comprehensive context and strategic awareness for each AI agent.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

class PlayerRole(Enum):
    LIBERAL = "liberal"
    FASCIST = "fascist"
    HITLER = "hitler"
    UNKNOWN = "unknown"

class VoteType(Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"

class ActionType(Enum):
    VOTE = "vote"
    NOMINATE = "nominate"
    POLICY_CHOICE = "policy_choice"
    SPECIAL_POWER = "special_power"
    CHAT = "chat"

@dataclass
class PlayerAction:
    """Represents a single player action"""
    player_id: str
    action_type: ActionType
    details: Dict[str, Any]
    round_number: int
    timestamp: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None

@dataclass
class VotingRecord:
    """Tracks voting patterns and outcomes"""
    round_number: int
    nomination: Dict[str, str]  # chancellor/president nominations
    votes: Dict[str, VoteType]  # player_id -> vote
    outcome: str  # "passed" or "failed"
    policies_before: Dict[str, int]  # liberal/fascist count before
    policies_after: Dict[str, int]  # liberal/fascist count after
    timestamp: str

@dataclass
class ConversationSummary:
    """Summarized conversation context"""
    round_number: int
    key_topics: List[str]
    player_statements: Dict[str, List[str]]  # player_id -> statements
    suspicions_raised: Dict[str, List[str]]  # who suspected whom
    alliances_formed: List[Tuple[str, str]]  # potential alliances
    deception_attempts: List[Dict[str, Any]]  # detected deception
    timestamp: str

@dataclass
class PlayerProfile:
    """Profile of another player based on observations"""
    player_id: str
    suspected_role: PlayerRole
    confidence: float  # 0.0 to 1.0
    voting_pattern: List[VoteType]
    trustworthiness: float  # -1.0 to 1.0
    alliance_potential: float  # -1.0 to 1.0
    recent_actions: List[PlayerAction]
    behavioral_notes: List[str]
    last_updated: str

class AgentWorldView:
    """
    Comprehensive world view for a Secret Hitler AI agent.
    Maintains strategic context, player profiles, and game history.
    """
    
    def __init__(self, agent_id: str, agent_role: PlayerRole):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.game_id: Optional[str] = None
        
        # Core game state
        self.current_round = 0
        self.total_players = 0
        self.policies_enacted = {"liberal": 0, "fascist": 0}
        self.game_phase = "nomination"  # nomination, voting, policy, special_power
        
        # Strategic goals based on role
        self.primary_goals = self._initialize_goals()
        self.current_strategy = "observe"  # observe, build_trust, deceive, execute_plan
        
        # Player tracking
        self.player_profiles: Dict[str, PlayerProfile] = {}
        self.known_roles: Dict[str, PlayerRole] = {agent_id: agent_role}
        self.suspected_hitler: Optional[str] = None
        self.trusted_players: Set[str] = set()
        self.suspicious_players: Set[str] = set()
        
        # Historical data
        self.voting_history: List[VotingRecord] = []
        self.conversation_history: List[ConversationSummary] = []
        self.action_history: List[PlayerAction] = []
        
        # Strategic analysis
        self.threat_assessment: Dict[str, float] = {}  # player_id -> threat level
        self.opportunity_analysis: Dict[str, Any] = {}
        self.win_probability: float = 0.5
        
        self.logger = logging.getLogger(f"agent_{agent_id}")
    
    def _initialize_goals(self) -> List[str]:
        """Initialize role-specific goals"""
        if self.agent_role == PlayerRole.LIBERAL:
            return [
                "Identify and eliminate fascists",
                "Prevent Hitler from becoming Chancellor",
                "Enact 5 liberal policies",
                "Build trust with other liberals",
                "Expose fascist deception"
            ]
        elif self.agent_role == PlayerRole.FASCIST:
            return [
                "Protect Hitler's identity",
                "Get Hitler elected as Chancellor",
                "Enact fascist policies",
                "Sow confusion among liberals",
                "Eliminate liberal threats"
            ]
        elif self.agent_role == PlayerRole.HITLER:
            return [
                "Stay hidden until the right moment",
                "Build trust with liberals",
                "Get elected as Chancellor after 3 fascist policies",
                "Avoid suspicion and investigation",
                "Manipulate voting patterns"
            ]
        else:
            return ["Survive", "Learn player roles", "Make strategic decisions"]
    
    def update_game_state(self, game_state: Dict[str, Any]):
        """Update core game state information"""
        self.current_round = game_state.get("round", self.current_round)
        self.total_players = game_state.get("player_count", self.total_players)
        self.policies_enacted = game_state.get("policies", self.policies_enacted)
        self.game_phase = game_state.get("phase", self.game_phase)
        
        # Update win probability based on game state
        self._calculate_win_probability()
        
        self.logger.debug(f"Updated game state: Round {self.current_round}, Phase {self.game_phase}")
    
    def add_player_action(self, action: PlayerAction):
        """Record a player action and update profiles"""
        self.action_history.append(action)
        
        # Update player profile
        if action.player_id not in self.player_profiles:
            self.player_profiles[action.player_id] = PlayerProfile(
                player_id=action.player_id,
                suspected_role=PlayerRole.UNKNOWN,
                confidence=0.0,
                voting_pattern=[],
                trustworthiness=0.0,
                alliance_potential=0.0,
                recent_actions=[],
                behavioral_notes=[],
                last_updated=datetime.now().isoformat()
            )
        
        profile = self.player_profiles[action.player_id]
        profile.recent_actions.append(action)
        profile.last_updated = datetime.now().isoformat()
        
        # Keep only recent actions (last 10)
        if len(profile.recent_actions) > 10:
            profile.recent_actions = profile.recent_actions[-10:]
        
        # Analyze action for strategic insights
        self._analyze_player_action(action)
    
    def add_voting_record(self, voting_record: VotingRecord):
        """Record voting results and analyze patterns"""
        self.voting_history.append(voting_record)
        
        # Update player voting patterns
        for player_id, vote in voting_record.votes.items():
            if player_id in self.player_profiles:
                self.player_profiles[player_id].voting_pattern.append(vote)
                # Keep only recent votes (last 15)
                if len(self.player_profiles[player_id].voting_pattern) > 15:
                    self.player_profiles[player_id].voting_pattern = \
                        self.player_profiles[player_id].voting_pattern[-15:]
        
        # Analyze voting patterns for strategic insights
        self._analyze_voting_patterns(voting_record)
        
        self.logger.info(f"Recorded voting for round {voting_record.round_number}: {voting_record.outcome}")
    
    def add_conversation_summary(self, conversation: ConversationSummary):
        """Add summarized conversation context"""
        self.conversation_history.append(conversation)
        
        # Update player profiles based on conversation
        for player_id, statements in conversation.player_statements.items():
            if player_id in self.player_profiles:
                profile = self.player_profiles[player_id]
                
                # Analyze statements for behavioral insights
                for statement in statements:
                    insight = self._analyze_statement(statement, player_id)
                    if insight:
                        profile.behavioral_notes.append(insight)
                
                # Keep only recent notes (last 20)
                if len(profile.behavioral_notes) > 20:
                    profile.behavioral_notes = profile.behavioral_notes[-20:]
        
        # Update suspicions and alliances
        self._update_social_dynamics(conversation)
        
        self.logger.debug(f"Added conversation summary for round {conversation.round_number}")
    
    def get_strategic_context(self) -> Dict[str, Any]:
        """Get comprehensive strategic context for decision making"""
        return {
            "agent_info": {
                "id": self.agent_id,
                "role": self.agent_role.value,
                "goals": self.primary_goals,
                "current_strategy": self.current_strategy
            },
            "game_state": {
                "round": self.current_round,
                "phase": self.game_phase,
                "policies": self.policies_enacted,
                "total_players": self.total_players
            },
            "player_analysis": {
                "known_roles": {pid: role.value for pid, role in self.known_roles.items()},
                "suspected_hitler": self.suspected_hitler,
                "trusted_players": list(self.trusted_players),
                "suspicious_players": list(self.suspicious_players),
                "threat_levels": self.threat_assessment
            },
            "strategic_summary": {
                "win_probability": self.win_probability,
                "key_opportunities": self._identify_opportunities(),
                "immediate_threats": self._identify_threats(),
                "recommended_actions": self._get_action_recommendations()
            },
            "historical_context": {
                "recent_votes": self._summarize_recent_votes(),
                "conversation_insights": self._summarize_conversations(),
                "behavioral_patterns": self._analyze_behavioral_patterns()
            }
        }
    
    def _analyze_player_action(self, action: PlayerAction):
        """Analyze a player action for strategic insights"""
        player_id = action.player_id
        profile = self.player_profiles[player_id]
        
        if action.action_type == ActionType.VOTE:
            vote = action.details.get("vote")
            # Analyze voting behavior
            if vote == VoteType.YES:
                profile.trustworthiness += 0.1 if self.agent_role == PlayerRole.LIBERAL else -0.1
            elif vote == VoteType.NO:
                profile.trustworthiness -= 0.1 if self.agent_role == PlayerRole.LIBERAL else 0.1
        
        elif action.action_type == ActionType.POLICY_CHOICE:
            policy = action.details.get("policy")
            if policy == "liberal":
                profile.suspected_role = PlayerRole.LIBERAL
                profile.confidence = min(1.0, profile.confidence + 0.3)
            elif policy == "fascist":
                if self.agent_role == PlayerRole.LIBERAL:
                    profile.suspected_role = PlayerRole.FASCIST
                    profile.confidence = min(1.0, profile.confidence + 0.4)
        
        # Clamp values
        profile.trustworthiness = max(-1.0, min(1.0, profile.trustworthiness))
        profile.confidence = max(0.0, min(1.0, profile.confidence))
    
    def _analyze_voting_patterns(self, voting_record: VotingRecord):
        """Analyze voting patterns for strategic insights"""
        # Look for coordinated voting (potential fascist coordination)
        yes_voters = [pid for pid, vote in voting_record.votes.items() if vote == VoteType.YES]
        no_voters = [pid for pid, vote in voting_record.votes.items() if vote == VoteType.NO]
        
        # Update threat assessment based on voting alignment
        for player_id in yes_voters:
            if player_id != self.agent_id:
                if voting_record.outcome == "passed" and self.agent_role == PlayerRole.LIBERAL:
                    self.threat_assessment[player_id] = \
                        self.threat_assessment.get(player_id, 0.0) + 0.1
        
        # Identify potential alliances
        if len(yes_voters) >= 2:
            for i, p1 in enumerate(yes_voters):
                for p2 in yes_voters[i+1:]:
                    if p1 != self.agent_id and p2 != self.agent_id:
                        # These players voted together
                        self._note_potential_alliance(p1, p2)
    
    def _analyze_statement(self, statement: str, player_id: str) -> Optional[str]:
        """Analyze a player statement for behavioral insights"""
        statement_lower = statement.lower()
        
        # Look for key phrases that indicate role or strategy
        if "trust" in statement_lower:
            return f"Emphasized trust (Round {self.current_round})"
        elif "suspicious" in statement_lower or "suspect" in statement_lower:
            return f"Raised suspicions (Round {self.current_round})"
        elif "liberal" in statement_lower:
            return f"Mentioned liberal policies/players (Round {self.current_round})"
        elif "fascist" in statement_lower:
            return f"Mentioned fascist policies/players (Round {self.current_round})"
        elif "hitler" in statement_lower:
            return f"Mentioned Hitler (Round {self.current_round})"
        
        return None
    
    def _update_social_dynamics(self, conversation: ConversationSummary):
        """Update social dynamics based on conversation"""
        # Update suspicions
        for accuser, suspects in conversation.suspicions_raised.items():
            for suspect in suspects:
                if suspect != self.agent_id:
                    self.suspicious_players.add(suspect)
                    self.threat_assessment[suspect] = \
                        self.threat_assessment.get(suspect, 0.0) + 0.2
        
        # Update alliances
        for p1, p2 in conversation.alliances_formed:
            if p1 != self.agent_id and p2 != self.agent_id:
                self._note_potential_alliance(p1, p2)
    
    def _note_potential_alliance(self, player1: str, player2: str):
        """Note a potential alliance between two players"""
        # Increase threat if both are suspected fascists
        if (player1 in self.suspicious_players and player2 in self.suspicious_players):
            self.threat_assessment[player1] = \
                self.threat_assessment.get(player1, 0.0) + 0.15
            self.threat_assessment[player2] = \
                self.threat_assessment.get(player2, 0.0) + 0.15
    
    def _calculate_win_probability(self):
        """Calculate current win probability based on game state"""
        liberal_policies = self.policies_enacted.get("liberal", 0)
        fascist_policies = self.policies_enacted.get("fascist", 0)
        
        if self.agent_role == PlayerRole.LIBERAL:
            # Liberal win conditions: 5 liberal policies or eliminate Hitler
            base_prob = liberal_policies / 5.0
            fascist_threat = fascist_policies / 6.0
            self.win_probability = max(0.1, min(0.9, base_prob - fascist_threat + 0.5))
        
        elif self.agent_role in [PlayerRole.FASCIST, PlayerRole.HITLER]:
            # Fascist win conditions: 6 fascist policies or Hitler as Chancellor (after 3 fascist)
            base_prob = fascist_policies / 6.0
            liberal_threat = liberal_policies / 5.0
            hitler_opportunity = 0.3 if fascist_policies >= 3 else 0.0
            self.win_probability = max(0.1, min(0.9, base_prob - liberal_threat + hitler_opportunity + 0.5))
    
    def _identify_opportunities(self) -> List[str]:
        """Identify current strategic opportunities"""
        opportunities = []
        
        if self.agent_role == PlayerRole.LIBERAL:
            if self.suspected_hitler:
                opportunities.append(f"Investigate/eliminate suspected Hitler: {self.suspected_hitler}")
            if len(self.suspicious_players) >= 2:
                opportunities.append("Coordinate with trusted players against fascist bloc")
            if self.policies_enacted.get("liberal", 0) >= 3:
                opportunities.append("Push for liberal policy majority")
        
        elif self.agent_role == PlayerRole.FASCIST:
            if self.policies_enacted.get("fascist", 0) >= 3:
                opportunities.append("Position Hitler for Chancellor nomination")
            if len(self.trusted_players) >= 1:
                opportunities.append("Manipulate trusted liberals")
        
        elif self.agent_role == PlayerRole.HITLER:
            if self.policies_enacted.get("fascist", 0) >= 3:
                opportunities.append("Seek Chancellor nomination for instant win")
            if self.trustworthiness > 0.5:
                opportunities.append("Maintain liberal facade while positioning for power")
        
        return opportunities
    
    def _identify_threats(self) -> List[str]:
        """Identify immediate threats"""
        threats = []
        
        high_threat_players = [
            pid for pid, threat in self.threat_assessment.items() 
            if threat > 0.5
        ]
        
        if high_threat_players:
            threats.append(f"High threat players: {', '.join(high_threat_players)}")
        
        if self.agent_role == PlayerRole.HITLER:
            if self.agent_id in self.suspicious_players:
                threats.append("Under suspicion - risk of investigation")
        
        if self.policies_enacted.get("liberal", 0) >= 4:
            threats.append("Liberals close to victory")
        elif self.policies_enacted.get("fascist", 0) >= 5:
            threats.append("Fascists close to victory")
        
        return threats
    
    def _get_action_recommendations(self) -> List[str]:
        """Get recommended actions based on current context"""
        recommendations = []
        
        if self.game_phase == "nomination":
            if self.agent_role == PlayerRole.LIBERAL:
                trusted = list(self.trusted_players)
                if trusted:
                    recommendations.append(f"Nominate trusted player: {trusted[0]}")
                else:
                    recommendations.append("Nominate least suspicious player")
            
            elif self.agent_role == PlayerRole.FASCIST:
                if self.policies_enacted.get("fascist", 0) >= 3:
                    recommendations.append("Consider nominating Hitler if safe")
                else:
                    recommendations.append("Nominate to advance fascist agenda")
        
        elif self.game_phase == "voting":
            if self.agent_role == PlayerRole.LIBERAL:
                recommendations.append("Vote based on trust levels and game state")
            else:
                recommendations.append("Vote to advance fascist interests")
        
        return recommendations
    
    def _summarize_recent_votes(self) -> List[Dict[str, Any]]:
        """Summarize recent voting patterns"""
        recent_votes = self.voting_history[-3:] if len(self.voting_history) >= 3 else self.voting_history
        
        summaries = []
        for vote_record in recent_votes:
            summaries.append({
                "round": vote_record.round_number,
                "outcome": vote_record.outcome,
                "yes_voters": [pid for pid, vote in vote_record.votes.items() if vote == VoteType.YES],
                "no_voters": [pid for pid, vote in vote_record.votes.items() if vote == VoteType.NO]
            })
        
        return summaries
    
    def _summarize_conversations(self) -> List[Dict[str, Any]]:
        """Summarize recent conversation insights"""
        recent_convos = self.conversation_history[-2:] if len(self.conversation_history) >= 2 else self.conversation_history
        
        summaries = []
        for convo in recent_convos:
            summaries.append({
                "round": convo.round_number,
                "key_topics": convo.key_topics,
                "suspicions": convo.suspicions_raised,
                "alliances": convo.alliances_formed
            })
        
        return summaries
    
    def _analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Analyze behavioral patterns across all players"""
        patterns = {}
        
        for player_id, profile in self.player_profiles.items():
            if len(profile.voting_pattern) >= 3:
                yes_rate = profile.voting_pattern.count(VoteType.YES) / len(profile.voting_pattern)
                patterns[player_id] = {
                    "yes_vote_rate": yes_rate,
                    "trustworthiness": profile.trustworthiness,
                    "suspected_role": profile.suspected_role.value,
                    "recent_behavior": profile.behavioral_notes[-3:] if profile.behavioral_notes else []
                }
        
        return patterns
    
    def export_worldview(self) -> Dict[str, Any]:
        """Export complete world view for checkpointing"""
        return {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role.value,
            "game_id": self.game_id,
            "current_round": self.current_round,
            "total_players": self.total_players,
            "policies_enacted": self.policies_enacted,
            "game_phase": self.game_phase,
            "primary_goals": self.primary_goals,
            "current_strategy": self.current_strategy,
            "player_profiles": {
                pid: asdict(profile) for pid, profile in self.player_profiles.items()
            },
            "known_roles": {pid: role.value for pid, role in self.known_roles.items()},
            "suspected_hitler": self.suspected_hitler,
            "trusted_players": list(self.trusted_players),
            "suspicious_players": list(self.suspicious_players),
            "voting_history": [asdict(vote) for vote in self.voting_history],
            "conversation_history": [asdict(convo) for convo in self.conversation_history],
            "action_history": [asdict(action) for action in self.action_history],
            "threat_assessment": self.threat_assessment,
            "opportunity_analysis": self.opportunity_analysis,
            "win_probability": self.win_probability,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def import_worldview(self, data: Dict[str, Any]):
        """Import world view from checkpointed data"""
        self.agent_id = data.get("agent_id", self.agent_id)
        self.agent_role = PlayerRole(data.get("agent_role", self.agent_role.value))
        self.game_id = data.get("game_id")
        self.current_round = data.get("current_round", 0)
        self.total_players = data.get("total_players", 0)
        self.policies_enacted = data.get("policies_enacted", {"liberal": 0, "fascist": 0})
        self.game_phase = data.get("game_phase", "nomination")
        self.primary_goals = data.get("primary_goals", self._initialize_goals())
        self.current_strategy = data.get("current_strategy", "observe")
        
        # Restore player profiles
        self.player_profiles = {}
        for pid, profile_data in data.get("player_profiles", {}).items():
            profile_data["suspected_role"] = PlayerRole(profile_data["suspected_role"])
            self.player_profiles[pid] = PlayerProfile(**profile_data)
        
        # Restore other data
        self.known_roles = {
            pid: PlayerRole(role) for pid, role in data.get("known_roles", {}).items()
        }
        self.suspected_hitler = data.get("suspected_hitler")
        self.trusted_players = set(data.get("trusted_players", []))
        self.suspicious_players = set(data.get("suspicious_players", []))
        self.threat_assessment = data.get("threat_assessment", {})
        self.opportunity_analysis = data.get("opportunity_analysis", {})
        self.win_probability = data.get("win_probability", 0.5)
        
        # Restore history (simplified for now)
        self.voting_history = []
        self.conversation_history = []
        self.action_history = []
        
        self.logger.info(f"Imported world view for agent {self.agent_id} from checkpoint")
