import gym
from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from copy import deepcopy

from ..game.state import GameStateManager
from ..game.rules import VoteType, GamePhase, Role, Party

class SecretHitlerEnv(gym.Env):
    """
    OpenAI Gym environment for Secret Hitler game
    Supports multi-agent training with different objectives
    """
    
    def __init__(self, num_players: int = 7, ai_players: List[str] = None):
        super().__init__()
        
        self.num_players = num_players
        self.ai_players = ai_players or []
        self.game_manager = GameStateManager()
        self.current_game_id = None
        self.episode_step = 0
        self.max_episode_steps = 200  # Prevent infinite games
        
        # Action space depends on game phase
        # We'll use a discrete action space with different meanings per phase
        self.action_space = spaces.Discrete(20)  # Max action ID
        
        # Observation space: game state features
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(100,),  # Feature vector size
            dtype=np.float32
        )
        
        # Track game statistics for training
        self.game_stats = {
            'total_games': 0,
            'liberal_wins': 0,
            'fascist_wins': 0,
            'games_by_role': {'liberal': 0, 'fascist': 0, 'hitler': 0},
            'wins_by_role': {'liberal': 0, 'fascist': 0, 'hitler': 0},
            'average_game_length': 0,
            'action_distributions': {}
        }
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment for new game"""
        
        # Create new game
        self.current_game_id = self.game_manager.create_game()
        
        # Add AI players
        self.player_ids = []
        for i, ai_id in enumerate(self.ai_players):
            player_id = self.game_manager.add_player(self.current_game_id, f"AI_{ai_id}_{i}")
            self.player_ids.append((player_id, ai_id))
        
        # Add human players if needed
        while len(self.player_ids) < self.num_players:
            human_id = self.game_manager.add_player(
                self.current_game_id, 
                f"Human_{len(self.player_ids)}"
            )
            self.player_ids.append((human_id, "human"))
        
        # Start game
        self.game_manager.start_game(self.current_game_id)
        
        self.episode_step = 0
        
        # Return initial observation for first AI player
        return self._get_observation(self.player_ids[0][0])
    
    def step(self, action: int, player_id: str = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action for current player
        
        Args:
            action: Action ID to execute
            player_id: ID of player taking action (if None, use current player)
        
        Returns:
            observation: New game state
            reward: Reward for the action
            done: Whether game is finished
            info: Additional information
        """
        
        game_state = self.game_manager.get_game_state(self.current_game_id)
        if not game_state:
            return np.zeros(100), 0, True, {"error": "Game not found"}
        
        # Determine acting player
        if player_id is None:
            player_id = self._get_current_player()
        
        if not player_id:
            return self._get_observation(self.player_ids[0][0]), 0, False, {"waiting": True}
        
        # Store old state for reward calculation
        old_state = self.game_manager.get_player_view(self.current_game_id, player_id)
        
        # Execute action
        action_info = self._execute_action(action, player_id, game_state)
        
        # Get new state
        new_state = self.game_manager.get_player_view(self.current_game_id, player_id)
        new_game_state = self.game_manager.get_game_state(self.current_game_id)
        
        # Calculate reward
        reward = self._calculate_reward(old_state, new_state, action_info, player_id)
        
        # Check if game is done
        done = (
            new_game_state.phase == GamePhase.GAME_OVER or 
            self.episode_step >= self.max_episode_steps
        )
        
        if done:
            self._update_game_stats(new_game_state)
        
        self.episode_step += 1
        
        info = {
            'action_info': action_info,
            'game_phase': new_game_state.phase.value,
            'episode_step': self.episode_step,
            'valid_actions': self._get_valid_actions(player_id)
        }
        
        return self._get_observation(player_id), reward, done, info
    
    def _get_current_player(self) -> Optional[str]:
        """Get the current player who should act"""
        
        game_state = self.game_manager.get_game_state(self.current_game_id)
        if not game_state:
            return None
        
        # Return current president during election phase
        if game_state.phase == GamePhase.ELECTION:
            if not game_state.nominated_chancellor_id:
                return game_state.current_president_id
            else:
                # All players need to vote
                alive_players = [p for p in game_state.players if p.is_alive]
                for player in alive_players:
                    if not any(v.player_id == player.id for v in game_state.votes):
                        return player.id
        
        elif game_state.phase == GamePhase.LEGISLATIVE:
            if len(game_state.drawn_policies) == 3:
                return game_state.current_president_id
            elif len(game_state.drawn_policies) == 2:
                return game_state.nominated_chancellor_id
        
        return None
    
    def _execute_action(self, action: int, player_id: str, game_state) -> Dict[str, Any]:
        """Execute the given action for the player"""
        
        action_info = {'action_id': action, 'success': False, 'error': None}
        
        try:
            if game_state.phase == GamePhase.ELECTION:
                if not game_state.nominated_chancellor_id:
                    # Nominate chancellor
                    target_players = [
                        p for p in game_state.players 
                        if p.is_alive and p.id != player_id
                    ]
                    if action < len(target_players):
                        chancellor_id = target_players[action].id
                        self.game_manager.nominate_chancellor(
                            self.current_game_id, player_id, chancellor_id
                        )
                        action_info.update({
                            'action_type': 'nominate',
                            'target': chancellor_id,
                            'success': True
                        })
                else:
                    # Vote
                    vote = VoteType.JA if action == 0 else VoteType.NEIN
                    self.game_manager.cast_vote(self.current_game_id, player_id, vote)
                    action_info.update({
                        'action_type': 'vote',
                        'vote': vote.value,
                        'success': True
                    })
            
            elif game_state.phase == GamePhase.LEGISLATIVE:
                if len(game_state.drawn_policies) == 3:
                    # President discards policy
                    if action < len(game_state.drawn_policies):
                        policy_id = game_state.drawn_policies[action].id
                        self.game_manager.president_discard_policy(
                            self.current_game_id, player_id, policy_id
                        )
                        action_info.update({
                            'action_type': 'discard',
                            'policy_id': policy_id,
                            'success': True
                        })
                
                elif len(game_state.drawn_policies) == 2:
                    # Chancellor enacts policy
                    if action < len(game_state.drawn_policies):
                        policy_id = game_state.drawn_policies[action].id
                        self.game_manager.chancellor_enact_policy(
                            self.current_game_id, player_id, policy_id
                        )
                        action_info.update({
                            'action_type': 'enact',
                            'policy_id': policy_id,
                            'success': True
                        })
        
        except Exception as e:
            action_info['error'] = str(e)
        
        return action_info
    
    def _get_valid_actions(self, player_id: str) -> List[int]:
        """Get list of valid action IDs for the current player"""
        
        game_state = self.game_manager.get_game_state(self.current_game_id)
        if not game_state:
            return []
        
        valid_actions = []
        
        if game_state.phase == GamePhase.ELECTION:
            if not game_state.nominated_chancellor_id and game_state.current_president_id == player_id:
                # Can nominate any eligible player
                target_players = [
                    p for p in game_state.players 
                    if p.is_alive and p.id != player_id
                ]
                valid_actions = list(range(len(target_players)))
            
            elif game_state.nominated_chancellor_id:
                # Can vote JA (0) or NEIN (1)
                valid_actions = [0, 1]
        
        elif game_state.phase == GamePhase.LEGISLATIVE:
            if len(game_state.drawn_policies) == 3 and game_state.current_president_id == player_id:
                # Can discard any of the 3 policies
                valid_actions = [0, 1, 2]
            
            elif len(game_state.drawn_policies) == 2 and game_state.nominated_chancellor_id == player_id:
                # Can enact either of the 2 policies
                valid_actions = [0, 1]
        
        return valid_actions
    
    def _get_observation(self, player_id: str) -> np.ndarray:
        """Get observation vector for the given player"""
        
        player_view = self.game_manager.get_player_view(self.current_game_id, player_id)
        if not player_view:
            return np.zeros(100)
        
        features = []
        
        # Game state features
        features.extend([
            player_view.get('liberal_policies', 0) / 5.0,
            player_view.get('fascist_policies', 0) / 6.0,
            player_view.get('election_tracker', 0) / 3.0,
        ])
        
        # Phase encoding
        phases = ['lobby', 'election', 'legislative', 'executive', 'game_over']
        phase = player_view.get('phase', 'lobby')
        phase_encoding = [1.0 if phase == p else 0.0 for p in phases]
        features.extend(phase_encoding)
        
        # Player features
        players = player_view.get('players', [])
        max_players = 10
        
        for i in range(max_players):
            if i < len(players):
                player = players[i]
                features.extend([
                    1.0 if player.get('is_alive') else 0.0,
                    1.0 if player.get('is_president') else 0.0,
                    1.0 if player.get('is_chancellor') else 0.0,
                    1.0 if player['id'] == player_id else 0.0,  # Is this me?
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Role information
        role = player_view.get('your_role', 'liberal')
        role_encoding = {
            'liberal': [1.0, 0.0, 0.0],
            'fascist': [0.0, 1.0, 0.0],
            'hitler': [0.0, 0.0, 1.0]
        }
        features.extend(role_encoding.get(role, [1.0, 0.0, 0.0]))
        
        # Team knowledge
        fascist_team = player_view.get('fascist_team', [])
        features.append(len(fascist_team) / max_players)
        
        # Current government
        features.extend([
            1.0 if player_view.get('current_president_id') == player_id else 0.0,
            1.0 if player_view.get('nominated_chancellor_id') == player_id else 0.0,
        ])
        
        # Pad to fixed size
        while len(features) < 100:
            features.append(0.0)
        
        return np.array(features[:100], dtype=np.float32)
    
    def _calculate_reward(
        self, 
        old_state: Dict, 
        new_state: Dict, 
        action_info: Dict,
        player_id: str
    ) -> float:
        """Calculate reward for the action taken"""
        
        role = old_state.get('your_role', 'liberal')
        
        # Base reward weights by role
        role_weights = {
            'liberal': {'liberal_policy': 1.0, 'fascist_policy': -1.0, 'win': 10.0, 'lose': -10.0},
            'fascist': {'liberal_policy': -1.0, 'fascist_policy': 1.0, 'win': 10.0, 'lose': -10.0},
            'hitler': {'liberal_policy': -1.0, 'fascist_policy': 1.0, 'win': 15.0, 'lose': -15.0}
        }
        
        weights = role_weights.get(role, role_weights['liberal'])
        reward = 0.0
        
        # Policy rewards
        old_liberal = old_state.get('liberal_policies', 0)
        new_liberal = new_state.get('liberal_policies', 0)
        old_fascist = old_state.get('fascist_policies', 0)
        new_fascist = new_state.get('fascist_policies', 0)
        
        if new_liberal > old_liberal:
            reward += weights['liberal_policy']
        if new_fascist > old_fascist:
            reward += weights['fascist_policy']
        
        # Game ending rewards
        winner = new_state.get('winner')
        if winner:
            if (role == 'liberal' and winner == 'liberal') or \
               (role in ['fascist', 'hitler'] and winner == 'fascist'):
                reward += weights['win']
            else:
                reward += weights['lose']
        
        # Action-specific rewards
        if action_info.get('success'):
            reward += 0.1  # Small reward for valid actions
        else:
            reward -= 0.2  # Penalty for invalid actions
        
        # Strategic rewards based on action type
        action_type = action_info.get('action_type')
        
        if action_type == 'vote':
            vote = action_info.get('vote')
            # Reward strategic voting
            if role == 'liberal':
                # Liberals should generally be cautious
                if vote == 'nein':
                    reward += 0.05
            else:  # Fascist/Hitler
                # Fascists might vote more strategically
                if new_fascist > old_fascist:  # Government that passed fascist policy
                    if vote == 'ja':
                        reward += 0.1
        
        elif action_type == 'nominate':
            # Reward nominating based on role strategy
            if role in ['fascist', 'hitler']:
                # Fascists might want to nominate other fascists
                # This would require knowledge of other players' roles
                reward += 0.05
        
        # Phase completion reward
        old_phase = old_state.get('phase')
        new_phase = new_state.get('phase')
        if old_phase != new_phase:
            reward += 0.1  # Small reward for advancing the game
        
        return reward
    
    def _update_game_stats(self, game_state):
        """Update training statistics"""
        
        self.game_stats['total_games'] += 1
        
        if game_state.winner:
            if game_state.winner == Party.LIBERAL:
                self.game_stats['liberal_wins'] += 1
            else:
                self.game_stats['fascist_wins'] += 1
        
        # Update role-specific stats
        for player in game_state.players:
            if player.role:
                role_name = player.role.value
                self.game_stats['games_by_role'][role_name] += 1
                
                # Check if this player won
                player_won = False
                if game_state.winner == Party.LIBERAL and player.role == Role.LIBERAL:
                    player_won = True
                elif game_state.winner == Party.FASCIST and player.role in [Role.FASCIST, Role.HITLER]:
                    player_won = True
                
                if player_won:
                    self.game_stats['wins_by_role'][role_name] += 1
        
        # Update average game length
        total_games = self.game_stats['total_games']
        current_avg = self.game_stats['average_game_length']
        self.game_stats['average_game_length'] = (
            (current_avg * (total_games - 1) + self.episode_step) / total_games
        )
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        stats = deepcopy(self.game_stats)
        
        # Calculate win rates
        total_games = stats['total_games']
        if total_games > 0:
            stats['liberal_win_rate'] = stats['liberal_wins'] / total_games
            stats['fascist_win_rate'] = stats['fascist_wins'] / total_games
            
            for role in ['liberal', 'fascist', 'hitler']:
                games = stats['games_by_role'][role]
                if games > 0:
                    stats[f'{role}_win_rate'] = stats['wins_by_role'][role] / games
                else:
                    stats[f'{role}_win_rate'] = 0.0
        
        return stats
    
    def render(self, mode='human'):
        """Render the current game state"""
        
        game_state = self.game_manager.get_game_state(self.current_game_id)
        if not game_state:
            print("No active game")
            return
        
        print(f"\n=== Secret Hitler Game ===")
        print(f"Phase: {game_state.phase.value}")
        print(f"Liberal Policies: {game_state.liberal_policies}/5")
        print(f"Fascist Policies: {game_state.fascist_policies}/6")
        print(f"Election Tracker: {game_state.election_tracker}/3")
        
        print(f"\nPlayers:")
        for player in game_state.players:
            status = []
            if player.is_president:
                status.append("PRESIDENT")
            if player.is_chancellor:
                status.append("CHANCELLOR")
            if not player.is_alive:
                status.append("DEAD")
            
            status_str = f" ({', '.join(status)})" if status else ""
            print(f"  {player.name}{status_str}")
        
        if game_state.winner:
            print(f"\nGame Over! Winner: {game_state.winner.value}")
        
        print("=" * 30)