"""
Enhanced AI Agent for Secret Hitler with World View Integration
Combines LLM training with comprehensive strategic context awareness.
"""

import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .agent_worldview import (
    AgentWorldView, PlayerRole, VoteType, ActionType, 
    PlayerAction, VotingRecord, ConversationSummary
)

try:
    from ..training.llm_trainer import get_llm_trainer
except ImportError:
    from backend.training.llm_trainer import get_llm_trainer

class EnhancedSecretHitlerAgent:
    """
    Enhanced AI agent that combines LLM training with comprehensive world view.
    Maintains strategic context, learns from experience, and uses checkpointed models.
    """
    
    def __init__(self, agent_id: str, agent_role: PlayerRole, game_id: str = None):
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.game_id = game_id
        
        # Initialize world view system
        self.world_view = AgentWorldView(agent_id, agent_role)
        self.world_view.game_id = game_id
        
        # Initialize LLM trainer
        self.llm_trainer = get_llm_trainer()
        
        # Load existing checkpoint if available
        self._load_agent_checkpoint()
        
        self.logger = logging.getLogger(f"enhanced_agent_{agent_id}")
        
        # Action history for this game
        self.game_actions = []
        self.conversation_buffer = []
        
        # Performance tracking
        self.decision_confidence = []
        self.action_outcomes = []
    
    def _load_agent_checkpoint(self):
        """Load the latest checkpoint for this agent role"""
        try:
            success = self.llm_trainer.load_checkpoint(agent_role=self.agent_role.value)
            if success:
                self.logger.info(f"Loaded checkpoint for {self.agent_role.value} agent")
            else:
                self.logger.info(f"No checkpoint found for {self.agent_role.value} agent - starting fresh")
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
    
    def update_game_state(self, game_state: Dict[str, Any]):
        """Update the agent's understanding of the current game state"""
        self.world_view.update_game_state(game_state)
        
        # Extract and update player information
        players = game_state.get('players', [])
        for player_info in players:
            player_id = player_info.get('id')
            if player_id and player_id != self.agent_id:
                # Initialize player profile if new
                if player_id not in self.world_view.player_profiles:
                    self.world_view.player_profiles[player_id] = {
                        'player_id': player_id,
                        'suspected_role': PlayerRole.UNKNOWN,
                        'confidence': 0.0,
                        'voting_pattern': [],
                        'trustworthiness': 0.0,
                        'alliance_potential': 0.0,
                        'recent_actions': [],
                        'behavioral_notes': [],
                        'last_updated': datetime.now().isoformat()
                    }
        
        self.logger.debug(f"Updated game state: Round {self.world_view.current_round}")
    
    def record_player_action(self, player_id: str, action_type: str, action_details: Dict[str, Any]):
        """Record an action taken by any player"""
        action = PlayerAction(
            player_id=player_id,
            action_type=ActionType(action_type.lower()),
            details=action_details,
            round_number=self.world_view.current_round,
            timestamp=datetime.now().isoformat()
        )
        
        self.world_view.add_player_action(action)
        
        # Track our own actions for performance analysis
        if player_id == self.agent_id:
            self.game_actions.append(action)
        
        self.logger.debug(f"Recorded action: {player_id} -> {action_type}")
    
    def record_voting_round(self, 
                           nominations: Dict[str, str], 
                           votes: Dict[str, str], 
                           outcome: str,
                           policies_before: Dict[str, int],
                           policies_after: Dict[str, int]):
        """Record the results of a voting round"""
        vote_mapping = {
            'yes': VoteType.YES,
            'no': VoteType.NO,
            'abstain': VoteType.ABSTAIN
        }
        
        voting_record = VotingRecord(
            round_number=self.world_view.current_round,
            nomination=nominations,
            votes={pid: vote_mapping.get(vote.lower(), VoteType.ABSTAIN) for pid, vote in votes.items()},
            outcome=outcome,
            policies_before=policies_before,
            policies_after=policies_after,
            timestamp=datetime.now().isoformat()
        )
        
        self.world_view.add_voting_record(voting_record)
        self.logger.info(f"Recorded voting round {self.world_view.current_round}: {outcome}")
    
    def add_conversation_context(self, 
                               player_statements: Dict[str, List[str]],
                               key_topics: List[str] = None,
                               suspicions: Dict[str, List[str]] = None,
                               alliances: List[Tuple[str, str]] = None):
        """Add conversation context from the current round"""
        conversation = ConversationSummary(
            round_number=self.world_view.current_round,
            key_topics=key_topics or [],
            player_statements=player_statements,
            suspicions_raised=suspicions or {},
            alliances_formed=alliances or [],
            deception_attempts=[],  # Could be enhanced to detect deception
            timestamp=datetime.now().isoformat()
        )
        
        self.world_view.add_conversation_summary(conversation)
        self.logger.debug(f"Added conversation context for round {self.world_view.current_round}")
    
    async def make_decision(self, decision_type: str, options: List[Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a strategic decision based on current world view and LLM training"""
        # Get comprehensive strategic context
        strategic_context = self.world_view.get_strategic_context()
        
        # Prepare context for LLM
        game_context = self._format_context_for_llm(strategic_context, decision_type, options, context)
        
        # Generate action using trained model
        llm_response = await self.llm_trainer.generate_action(
            game_context=game_context,
            player_role=self.agent_role.value
        )
        
        # Combine LLM suggestion with strategic analysis
        decision = self._make_strategic_decision(
            decision_type=decision_type,
            options=options,
            llm_suggestion=llm_response,
            strategic_context=strategic_context,
            additional_context=context
        )
        
        # Record decision confidence
        self.decision_confidence.append({
            'decision_type': decision_type,
            'confidence': decision.get('confidence', 0.5),
            'round': self.world_view.current_round,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info(f"Made decision: {decision_type} -> {decision.get('choice')} (confidence: {decision.get('confidence', 0.5):.2f})")
        return decision
    
    def _format_context_for_llm(self, strategic_context: Dict[str, Any], decision_type: str, options: List[Any], additional_context: Dict[str, Any] = None) -> str:
        """Format the strategic context into a prompt for the LLM"""
        agent_info = strategic_context['agent_info']
        game_state = strategic_context['game_state']
        player_analysis = strategic_context['player_analysis']
        strategic_summary = strategic_context['strategic_summary']
        historical_context = strategic_context['historical_context']
        
        context_prompt = f"""SECRET HITLER AI AGENT CONTEXT
        
=== AGENT IDENTITY ===
Role: {agent_info['role']}
ID: {agent_info['id']}
Current Strategy: {agent_info['current_strategy']}

=== PRIMARY GOALS ===
{chr(10).join(f"- {goal}" for goal in agent_info['goals'])}

=== GAME STATE ===
Round: {game_state['round']}
Phase: {game_state['phase']}
Policies Enacted: Liberal {game_state['policies']['liberal']}, Fascist {game_state['policies']['fascist']}
Total Players: {game_state['total_players']}

=== PLAYER ANALYSIS ===
Known Roles: {player_analysis['known_roles']}
Suspected Hitler: {player_analysis['suspected_hitler'] or 'Unknown'}
Trusted Players: {', '.join(player_analysis['trusted_players']) or 'None'}
Suspicious Players: {', '.join(player_analysis['suspicious_players']) or 'None'}

=== STRATEGIC SITUATION ===
Win Probability: {strategic_summary['win_probability']:.1%}
Key Opportunities: {', '.join(strategic_summary['key_opportunities']) or 'None identified'}
Immediate Threats: {', '.join(strategic_summary['immediate_threats']) or 'None identified'}
Recommended Actions: {', '.join(strategic_summary['recommended_actions']) or 'No specific recommendations'}

=== RECENT HISTORY ===
Recent Votes: {len(historical_context['recent_votes'])} voting rounds recorded
Conversation Insights: {len(historical_context['conversation_insights'])} conversation summaries
Behavioral Patterns: {len(historical_context['behavioral_patterns'])} players analyzed

=== CURRENT DECISION ===
Decision Type: {decision_type}
Available Options: {options}
Additional Context: {additional_context or 'None'}

Based on your role as a {agent_info['role']} and the current strategic situation, what decision should you make?
Consider your goals, the threat assessment, opportunities, and historical patterns.
"""
        
        return context_prompt
    
    def _make_strategic_decision(self, 
                               decision_type: str, 
                               options: List[Any], 
                               llm_suggestion: Dict[str, Any],
                               strategic_context: Dict[str, Any],
                               additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Combine LLM suggestion with strategic analysis to make final decision"""
        
        # Base decision on LLM suggestion
        base_confidence = llm_suggestion.get('confidence', 0.5)
        suggested_action = llm_suggestion.get('action', 'unknown')
        
        # Apply strategic modifiers based on context
        strategic_summary = strategic_context['strategic_summary']
        player_analysis = strategic_context['player_analysis']
        
        # Adjust confidence based on strategic factors
        confidence_modifiers = []
        
        # Win probability modifier
        win_prob = strategic_summary['win_probability']
        if win_prob > 0.7:
            confidence_modifiers.append(0.1)  # More confident when winning
        elif win_prob < 0.3:
            confidence_modifiers.append(-0.1)  # Less confident when losing
        
        # Threat assessment modifier
        if strategic_summary['immediate_threats']:
            confidence_modifiers.append(-0.05)  # Less confident under threat
        
        # Opportunity modifier
        if strategic_summary['key_opportunities']:
            confidence_modifiers.append(0.05)  # More confident with opportunities
        
        # Apply modifiers
        final_confidence = base_confidence + sum(confidence_modifiers)
        final_confidence = max(0.1, min(0.95, final_confidence))  # Clamp to reasonable range
        
        # Make role-specific adjustments
        decision = self._apply_role_specific_logic(
            decision_type=decision_type,
            options=options,
            suggested_action=suggested_action,
            strategic_context=strategic_context,
            additional_context=additional_context
        )
        
        return {
            'choice': decision,
            'confidence': final_confidence,
            'reasoning': llm_suggestion.get('reasoning', 'Strategic analysis'),
            'llm_suggestion': suggested_action,
            'strategic_factors': {
                'win_probability': strategic_summary['win_probability'],
                'threats': strategic_summary['immediate_threats'],
                'opportunities': strategic_summary['key_opportunities']
            }
        }
    
    def _apply_role_specific_logic(self, 
                                 decision_type: str, 
                                 options: List[Any], 
                                 suggested_action: str,
                                 strategic_context: Dict[str, Any],
                                 additional_context: Dict[str, Any] = None) -> Any:
        """Apply role-specific decision logic"""
        
        player_analysis = strategic_context['player_analysis']
        game_state = strategic_context['game_state']
        
        if decision_type == 'vote':
            return self._make_voting_decision(options, suggested_action, strategic_context)
        elif decision_type == 'nominate':
            return self._make_nomination_decision(options, suggested_action, strategic_context)
        elif decision_type == 'policy_choice':
            return self._make_policy_decision(options, suggested_action, strategic_context)
        elif decision_type == 'special_power':
            return self._make_special_power_decision(options, suggested_action, strategic_context)
        else:
            # Default to first option or LLM suggestion
            if suggested_action in options:
                return suggested_action
            return options[0] if options else None
    
    def _make_voting_decision(self, options: List[str], suggested_action: str, strategic_context: Dict[str, Any]) -> str:
        """Make voting decision based on role and strategic context"""
        player_analysis = strategic_context['player_analysis']
        
        if self.agent_role == PlayerRole.LIBERAL:
            # Liberals should vote against suspected fascists
            if 'yes' in options and 'no' in options:
                # Check if we trust the nominated players
                trusted_players = set(player_analysis['trusted_players'])
                suspicious_players = set(player_analysis['suspicious_players'])
                
                # If more trusted than suspicious players in government, vote yes
                # This would need actual nomination context
                return suggested_action if suggested_action in options else 'no'
        
        elif self.agent_role == PlayerRole.FASCIST:
            # Fascists should coordinate to advance their agenda
            fascist_policies = strategic_context['game_state']['policies']['fascist']
            if fascist_policies >= 3:
                # Try to get Hitler into power
                return 'yes' if 'yes' in options else suggested_action
            else:
                # Build trust while advancing fascist policies
                return suggested_action if suggested_action in options else 'yes'
        
        elif self.agent_role == PlayerRole.HITLER:
            # Hitler should stay hidden and vote conservatively
            return 'no' if 'no' in options else suggested_action
        
        return suggested_action if suggested_action in options else options[0]
    
    def _make_nomination_decision(self, options: List[str], suggested_action: str, strategic_context: Dict[str, Any]) -> str:
        """Make nomination decision based on role and strategic context"""
        player_analysis = strategic_context['player_analysis']
        
        if self.agent_role == PlayerRole.LIBERAL:
            # Nominate most trusted player
            trusted_players = player_analysis['trusted_players']
            for player in trusted_players:
                if player in options:
                    return player
        
        elif self.agent_role == PlayerRole.FASCIST:
            # Nominate Hitler if safe, otherwise another fascist
            fascist_policies = strategic_context['game_state']['policies']['fascist']
            suspected_hitler = player_analysis['suspected_hitler']
            
            if fascist_policies >= 3 and suspected_hitler and suspected_hitler in options:
                return suspected_hitler
        
        elif self.agent_role == PlayerRole.HITLER:
            # Nominate a liberal to build trust
            trusted_players = player_analysis['trusted_players']
            for player in trusted_players:
                if player in options:
                    return player
        
        return suggested_action if suggested_action in options else options[0]
    
    def _make_policy_decision(self, options: List[str], suggested_action: str, strategic_context: Dict[str, Any]) -> str:
        """Make policy choice decision"""
        if self.agent_role == PlayerRole.LIBERAL:
            # Always choose liberal policy if available
            if 'liberal' in options:
                return 'liberal'
        
        elif self.agent_role in [PlayerRole.FASCIST, PlayerRole.HITLER]:
            # Choose fascist policy if it advances the agenda
            fascist_policies = strategic_context['game_state']['policies']['fascist']
            if 'fascist' in options and fascist_policies < 6:
                return 'fascist'
        
        return suggested_action if suggested_action in options else options[0]
    
    def _make_special_power_decision(self, options: List[str], suggested_action: str, strategic_context: Dict[str, Any]) -> str:
        """Make special power decision (investigate, eliminate, etc.)"""
        player_analysis = strategic_context['player_analysis']
        
        if self.agent_role == PlayerRole.LIBERAL:
            # Investigate most suspicious player
            suspicious_players = player_analysis['suspicious_players']
            for player in suspicious_players:
                if player in options:
                    return player
        
        elif self.agent_role in [PlayerRole.FASCIST, PlayerRole.HITLER]:
            # Target most threatening liberal
            threat_levels = player_analysis.get('threat_levels', {})
            highest_threat = max(threat_levels.items(), key=lambda x: x[1], default=(None, 0))
            if highest_threat[0] and highest_threat[0] in options:
                return highest_threat[0]
        
        return suggested_action if suggested_action in options else options[0]
    
    def record_decision_outcome(self, decision: Dict[str, Any], outcome: Dict[str, Any]):
        """Record the outcome of a decision for learning"""
        self.action_outcomes.append({
            'decision': decision,
            'outcome': outcome,
            'round': self.world_view.current_round,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update world view based on outcome
        if outcome.get('success', False):
            self.world_view.win_probability += 0.05
        else:
            self.world_view.win_probability -= 0.05
        
        # Clamp win probability
        self.world_view.win_probability = max(0.1, min(0.9, self.world_view.win_probability))
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this game"""
        avg_confidence = sum(d['confidence'] for d in self.decision_confidence) / len(self.decision_confidence) if self.decision_confidence else 0.0
        
        successful_outcomes = sum(1 for outcome in self.action_outcomes if outcome['outcome'].get('success', False))
        success_rate = successful_outcomes / len(self.action_outcomes) if self.action_outcomes else 0.0
        
        return {
            'agent_id': self.agent_id,
            'agent_role': self.agent_role.value,
            'game_id': self.game_id,
            'total_decisions': len(self.decision_confidence),
            'average_confidence': avg_confidence,
            'success_rate': success_rate,
            'final_win_probability': self.world_view.win_probability,
            'actions_taken': len(self.game_actions),
            'world_view_summary': {
                'known_roles': len(self.world_view.known_roles),
                'trusted_players': len(self.world_view.trusted_players),
                'suspicious_players': len(self.world_view.suspicious_players),
                'voting_records': len(self.world_view.voting_history),
                'conversation_summaries': len(self.world_view.conversation_history)
            }
        }
    
    def save_agent_state(self) -> str:
        """Save the current agent state including world view"""
        # Save LLM checkpoint
        checkpoint_id = self.llm_trainer.save_checkpoint(agent_role=self.agent_role.value)
        
        # Export world view
        world_view_data = self.world_view.export_worldview()
        
        # Save additional agent data
        agent_data = {
            'world_view': world_view_data,
            'game_actions': [action.__dict__ if hasattr(action, '__dict__') else action for action in self.game_actions],
            'decision_confidence': self.decision_confidence,
            'action_outcomes': self.action_outcomes,
            'performance_summary': self.get_performance_summary()
        }
        
        # This could be saved to a separate agent state file
        self.logger.info(f"Saved agent state with checkpoint {checkpoint_id}")
        return checkpoint_id
    
    def load_agent_state(self, checkpoint_id: str = None) -> bool:
        """Load agent state from checkpoint"""
        success = self.llm_trainer.load_checkpoint(checkpoint_id, agent_role=self.agent_role.value)
        if success:
            self.logger.info(f"Loaded agent state from checkpoint")
        return success
    
    async def end_game_analysis(self, game_result: Dict[str, Any]):
        """Perform end-of-game analysis and learning"""
        performance = self.get_performance_summary()
        
        # Analyze what went well and what didn't
        analysis = {
            'game_result': game_result,
            'agent_performance': performance,
            'strategic_insights': self._extract_strategic_insights(),
            'learning_opportunities': self._identify_learning_opportunities()
        }
        
        # Train on game experience
        game_data = [{
            'agent_id': self.agent_id,
            'agent_role': self.agent_role.value,
            'actions': self.game_actions,
            'outcomes': self.action_outcomes,
            'final_result': game_result,
            'world_view': self.world_view.export_worldview()
        }]
        
        # Update LLM with game experience
        training_metrics = await self.llm_trainer.train_on_game_data(game_data)
        
        # Save checkpoint if training step threshold reached
        if self.llm_trainer.should_save_checkpoint():
            checkpoint_id = self.save_agent_state()
            analysis['checkpoint_saved'] = checkpoint_id
        
        self.logger.info(f"Completed end-game analysis: {analysis['strategic_insights']}")
        return analysis
    
    def _extract_strategic_insights(self) -> List[str]:
        """Extract strategic insights from the game"""
        insights = []
        
        # Analyze decision confidence patterns
        if self.decision_confidence:
            avg_confidence = sum(d['confidence'] for d in self.decision_confidence) / len(self.decision_confidence)
            if avg_confidence < 0.6:
                insights.append("Low decision confidence - need better strategic analysis")
            elif avg_confidence > 0.8:
                insights.append("High decision confidence - strategic analysis working well")
        
        # Analyze success rate
        if self.action_outcomes:
            successful_outcomes = sum(1 for outcome in self.action_outcomes if outcome['outcome'].get('success', False))
            success_rate = successful_outcomes / len(self.action_outcomes)
            if success_rate < 0.4:
                insights.append("Low success rate - need to improve decision making")
            elif success_rate > 0.7:
                insights.append("High success rate - decision making effective")
        
        # Analyze world view accuracy
        known_roles = len(self.world_view.known_roles)
        if known_roles < 3:
            insights.append("Limited role knowledge - need better investigation strategy")
        elif known_roles > 5:
            insights.append("Good role knowledge - investigation strategy effective")
        
        return insights
    
    def _identify_learning_opportunities(self) -> List[str]:
        """Identify areas for improvement"""
        opportunities = []
        
        # Check for patterns in failed decisions
        failed_decisions = [outcome for outcome in self.action_outcomes if not outcome['outcome'].get('success', False)]
        
        if len(failed_decisions) > len(self.action_outcomes) * 0.6:
            opportunities.append("High failure rate - review decision criteria")
        
        # Check for underutilized strategic factors
        if len(self.world_view.trusted_players) == 0:
            opportunities.append("No trusted players identified - improve trust building")
        
        if len(self.world_view.suspicious_players) == 0:
            opportunities.append("No suspicious players identified - improve threat detection")
        
        # Check conversation analysis
        if len(self.world_view.conversation_history) < self.world_view.current_round * 0.5:
            opportunities.append("Limited conversation analysis - improve social awareness")
        
        return opportunities
