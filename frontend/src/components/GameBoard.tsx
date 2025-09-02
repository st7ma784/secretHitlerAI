import React, { useState, useEffect } from 'react';

interface Player {
  id: string;
  name: string;
  is_alive: boolean;
  is_president: boolean;
  is_chancellor: boolean;
}

interface Policy {
  id: string;
  type: 'liberal' | 'fascist';
}

interface GameState {
  game_id: string;
  phase: 'lobby' | 'election' | 'legislative' | 'executive' | 'game_over';
  liberal_policies: number;
  fascist_policies: number;
  election_tracker: number;
  current_president_id?: string;
  nominated_chancellor_id?: string;
  winner?: 'liberal' | 'fascist';
  players: Player[];
  your_role?: 'liberal' | 'fascist' | 'hitler';
  your_party?: 'liberal' | 'fascist';
  fascist_team?: Array<{id: string, name: string, role: string}>;
  drawn_policies?: Policy[];
}

interface GameBoardProps {
  gameId: string;
  playerId: string;
  onAction: (action: string, data: any) => void;
}

const GameBoard: React.FC<GameBoardProps> = ({ gameId, playerId, onAction }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [selectedChancellor, setSelectedChancellor] = useState<string>('');
  const [selectedPolicy, setSelectedPolicy] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchGameState();
    const interval = setInterval(fetchGameState, 2000); // Poll every 2 seconds
    return () => clearInterval(interval);
  }, [gameId, playerId]);

  const fetchGameState = async () => {
    try {
      const response = await fetch(`/api/games/${gameId}/state?player_id=${playerId}`);
      if (response.ok) {
        const data = await response.json();
        setGameState(data);
      }
    } catch (error) {
      console.error('Failed to fetch game state:', error);
    }
  };

  const handleNominateChancellor = async () => {
    if (!selectedChancellor) return;
    
    setLoading(true);
    try {
      await onAction('nominate', { chancellor_id: selectedChancellor });
      setSelectedChancellor('');
    } finally {
      setLoading(false);
    }
  };

  const handleVote = async (vote: 'ja' | 'nein') => {
    setLoading(true);
    try {
      await onAction('vote', { vote });
    } finally {
      setLoading(false);
    }
  };

  const handleDiscardPolicy = async () => {
    if (!selectedPolicy) return;
    
    setLoading(true);
    try {
      await onAction('discard', { policy_id: selectedPolicy });
      setSelectedPolicy('');
    } finally {
      setLoading(false);
    }
  };

  const handleEnactPolicy = async () => {
    if (!selectedPolicy) return;
    
    setLoading(true);
    try {
      await onAction('enact', { policy_id: selectedPolicy });
      setSelectedPolicy('');
    } finally {
      setLoading(false);
    }
  };

  if (!gameState) {
    return <div className=\"p-4\">Loading game state...</div>;
  }

  const isCurrentPlayer = (playerId: string) => playerId === gameState.current_president_id;
  const isNominatedChancellor = (playerId: string) => playerId === gameState.nominated_chancellor_id;
  const canNominate = gameState.phase === 'election' && gameState.current_president_id === playerId;
  const canVote = gameState.phase === 'election' && gameState.nominated_chancellor_id;
  const isPresident = gameState.current_president_id === playerId && gameState.phase === 'legislative';
  const isChancellor = gameState.nominated_chancellor_id === playerId && gameState.phase === 'legislative';

  return (
    <div className=\"min-h-screen bg-gray-900 text-white p-4\">
      <div className=\"max-w-6xl mx-auto\">
        {/* Game Header */}
        <div className=\"bg-gray-800 rounded-lg p-6 mb-6\">
          <h1 className=\"text-3xl font-bold mb-4\">Secret Hitler</h1>
          <div className=\"grid grid-cols-2 md:grid-cols-4 gap-4\">
            <div className=\"text-center\">
              <div className=\"text-sm text-gray-400\">Liberal Policies</div>
              <div className=\"text-2xl font-bold text-blue-400\">{gameState.liberal_policies}/5</div>
            </div>
            <div className=\"text-center\">
              <div className=\"text-sm text-gray-400\">Fascist Policies</div>
              <div className=\"text-2xl font-bold text-red-400\">{gameState.fascist_policies}/6</div>
            </div>
            <div className=\"text-center\">
              <div className=\"text-sm text-gray-400\">Election Tracker</div>
              <div className=\"text-2xl font-bold text-yellow-400\">{gameState.election_tracker}/3</div>
            </div>
            <div className=\"text-center\">
              <div className=\"text-sm text-gray-400\">Phase</div>
              <div className=\"text-lg font-semibold capitalize\">{gameState.phase}</div>
            </div>
          </div>
        </div>

        {/* Player Info */}
        <div className=\"bg-gray-800 rounded-lg p-6 mb-6\">
          <h2 className=\"text-xl font-bold mb-4\">Your Role</h2>
          <div className=\"flex gap-4\">
            <div>
              <span className=\"text-gray-400\">Role: </span>
              <span className={`font-semibold ${
                gameState.your_role === 'hitler' ? 'text-red-500' :
                gameState.your_party === 'fascist' ? 'text-red-400' : 'text-blue-400'
              }`}>
                {gameState.your_role?.toUpperCase()}
              </span>
            </div>
            <div>
              <span className=\"text-gray-400\">Party: </span>
              <span className={`font-semibold ${
                gameState.your_party === 'fascist' ? 'text-red-400' : 'text-blue-400'
              }`}>
                {gameState.your_party?.toUpperCase()}
              </span>
            </div>
          </div>
          
          {gameState.fascist_team && (
            <div className=\"mt-4\">
              <div className=\"text-gray-400 mb-2\">Fascist Team:</div>
              <div className=\"flex flex-wrap gap-2\">
                {gameState.fascist_team.map(player => (
                  <span key={player.id} className=\"bg-red-900 px-3 py-1 rounded text-sm\">
                    {player.name} ({player.role})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Players List */}
        <div className=\"bg-gray-800 rounded-lg p-6 mb-6\">
          <h2 className=\"text-xl font-bold mb-4\">Players</h2>
          <div className=\"grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4\">
            {gameState.players.map(player => (
              <div
                key={player.id}
                className={`p-4 rounded-lg border-2 ${
                  player.is_president ? 'border-yellow-500 bg-yellow-900/20' :
                  player.is_chancellor ? 'border-purple-500 bg-purple-900/20' :
                  isNominatedChancellor(player.id) ? 'border-blue-500 bg-blue-900/20' :
                  'border-gray-600 bg-gray-700'
                } ${!player.is_alive ? 'opacity-50' : ''}`}
              >
                <div className=\"font-semibold\">{player.name}</div>
                <div className=\"text-sm text-gray-400\">
                  {player.is_president && '🏛️ President'}
                  {player.is_chancellor && '📜 Chancellor'}
                  {isNominatedChancellor(player.id) && '🗳️ Nominated'}
                  {!player.is_alive && '💀 Dead'}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Game Actions */}
        <div className=\"bg-gray-800 rounded-lg p-6\">
          <h2 className=\"text-xl font-bold mb-4\">Actions</h2>
          
          {/* Chancellor Nomination */}
          {canNominate && (
            <div className=\"mb-6\">
              <h3 className=\"text-lg font-semibold mb-3\">Nominate Chancellor</h3>
              <div className=\"flex flex-wrap gap-2 mb-4\">
                {gameState.players
                  .filter(p => p.is_alive && p.id !== playerId)
                  .map(player => (
                    <button
                      key={player.id}
                      onClick={() => setSelectedChancellor(player.id)}
                      className={`px-4 py-2 rounded ${
                        selectedChancellor === player.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                      }`}
                    >
                      {player.name}
                    </button>
                  ))}
              </div>
              <button
                onClick={handleNominateChancellor}
                disabled={!selectedChancellor || loading}
                className=\"bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-2 rounded font-semibold\"
              >
                Nominate Chancellor
              </button>
            </div>
          )}

          {/* Voting */}
          {canVote && (
            <div className=\"mb-6\">
              <h3 className=\"text-lg font-semibold mb-3\">Vote on Government</h3>
              <p className=\"text-gray-400 mb-4\">
                President: {gameState.players.find(p => p.id === gameState.current_president_id)?.name}
                <br />
                Nominated Chancellor: {gameState.players.find(p => p.id === gameState.nominated_chancellor_id)?.name}
              </p>
              <div className=\"flex gap-4\">
                <button
                  onClick={() => handleVote('ja')}
                  disabled={loading}
                  className=\"bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-2 rounded font-semibold\"
                >
                  JA (Yes)
                </button>
                <button
                  onClick={() => handleVote('nein')}
                  disabled={loading}
                  className=\"bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-6 py-2 rounded font-semibold\"
                >
                  NEIN (No)
                </button>
              </div>
            </div>
          )}

          {/* Policy Actions */}
          {(isPresident || isChancellor) && gameState.drawn_policies && (
            <div className=\"mb-6\">
              <h3 className=\"text-lg font-semibold mb-3\">
                {isPresident ? 'Choose Policy to Discard' : 'Choose Policy to Enact'}
              </h3>
              <div className=\"flex flex-wrap gap-2 mb-4\">
                {gameState.drawn_policies.map(policy => (
                  <button
                    key={policy.id}
                    onClick={() => setSelectedPolicy(policy.id)}
                    className={`px-4 py-2 rounded border-2 ${
                      selectedPolicy === policy.id
                        ? 'border-white bg-gray-600'
                        : 'border-gray-500 bg-gray-700 hover:bg-gray-600'
                    } ${
                      policy.type === 'liberal' ? 'text-blue-400' : 'text-red-400'
                    }`}
                  >
                    {policy.type === 'liberal' ? '🕊️ Liberal' : '🔥 Fascist'}
                  </button>
                ))}
              </div>
              <button
                onClick={isPresident ? handleDiscardPolicy : handleEnactPolicy}
                disabled={!selectedPolicy || loading}
                className=\"bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-6 py-2 rounded font-semibold\"
              >
                {isPresident ? 'Discard Policy' : 'Enact Policy'}
              </button>
            </div>
          )}

          {/* Game Over */}
          {gameState.phase === 'game_over' && (
            <div className=\"text-center\">
              <h3 className=\"text-2xl font-bold mb-4\">Game Over!</h3>
              <p className={`text-xl ${
                gameState.winner === 'liberal' ? 'text-blue-400' : 'text-red-400'
              }`}>
                {gameState.winner === 'liberal' ? '🕊️ Liberals Win!' : '🔥 Fascists Win!'}
              </p>
            </div>
          )}

          {/* Waiting State */}
          {gameState.phase === 'election' && !canNominate && !canVote && (
            <div className=\"text-gray-400 text-center\">
              Waiting for other players...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GameBoard;