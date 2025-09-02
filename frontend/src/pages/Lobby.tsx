import React, { useState, useEffect } from 'react';

interface Player {
  id: string;
  name: string;
  is_alive: boolean;
  is_president: boolean;
  is_chancellor: boolean;
}

interface GameConfig {
  numBots: number;
  difficulty: 'easy' | 'medium' | 'hard';
  enableTraining: boolean;
  gamesPerSession?: number;
  trainingInterval?: number;
}

interface LobbyProps {
  onGameStart: (gameId: string, playerId: string, playerName: string) => void;
  onBackToModeSelection: () => void;
  gameConfig: GameConfig | null;
}

const Lobby: React.FC<LobbyProps> = ({ onGameStart, onBackToModeSelection, gameConfig }) => {
  const [gameId, setGameId] = useState('');
  const [playerName, setPlayerName] = useState('');
  const [currentGameId, setCurrentGameId] = useState<string>('');
  const [currentPlayerId, setCurrentPlayerId] = useState<string>('');
  const [players, setPlayers] = useState<Player[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [gameCreated, setGameCreated] = useState(false);

  useEffect(() => {
    if (currentGameId && gameCreated) {
      const interval = setInterval(fetchPlayers, 2000);
      return () => clearInterval(interval);
    }
  }, [currentGameId, gameCreated]);

  const fetchPlayers = async () => {
    if (!currentGameId) return;
    
    try {
      const response = await fetch(`/api/games/${currentGameId}/players`);
      if (response.ok) {
        const data = await response.json();
        setPlayers(data.players);
      }
    } catch (err) {
      console.error('Failed to fetch players:', err);
    }
  };

  const createGame = async () => {
    if (!playerName.trim()) {
      setError('Please enter your name');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // Create game
      const createResponse = await fetch('/api/games', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ game_id: gameId || undefined }),
      });

      if (!createResponse.ok) {
        throw new Error('Failed to create game');
      }

      const createData = await createResponse.json();
      const newGameId = createData.game_id;

      // Join the game
      const joinResponse = await fetch(`/api/games/${newGameId}/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ player_name: playerName.trim() }),
      });

      if (!joinResponse.ok) {
        throw new Error('Failed to join game');
      }

      const joinData = await joinResponse.json();

      setCurrentGameId(newGameId);
      setCurrentPlayerId(joinData.player_id);
      setGameCreated(true);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const joinGame = async () => {
    if (!gameId.trim() || !playerName.trim()) {
      setError('Please enter both game ID and your name');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`/api/games/${gameId.trim()}/join`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ player_name: playerName.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to join game');
      }

      const data = await response.json();

      setCurrentGameId(gameId.trim());
      setCurrentPlayerId(data.player_id);
      setGameCreated(true);
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const startGame = async () => {
    if (!currentGameId) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`/api/games/${currentGameId}/start`, {
        method: 'POST',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to start game');
      }

      onGameStart(currentGameId, currentPlayerId, playerName);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const copyGameId = () => {
    navigator.clipboard.writeText(currentGameId);
  };

  if (gameCreated) {
    return (
      <div className=\"min-h-screen bg-gray-900 text-white flex items-center justify-center\">
        <div className=\"bg-gray-800 rounded-lg p-8 max-w-2xl w-full mx-4\">
          <h1 className=\"text-3xl font-bold text-center mb-6\">Game Lobby</h1>
          
          <div className=\"bg-gray-700 rounded-lg p-4 mb-6\">
            <div className=\"flex items-center justify-between mb-2\">
              <span className=\"text-gray-400\">Game ID:</span>
              <div className=\"flex items-center gap-2\">
                <code className=\"bg-gray-600 px-3 py-1 rounded text-sm\">{currentGameId}</code>
                <button
                  onClick={copyGameId}
                  className=\"bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm\"
                >
                  Copy
                </button>
              </div>
            </div>
            <p className=\"text-sm text-gray-400\">
              Share this Game ID with other players so they can join!
            </p>
          </div>

          <div className=\"mb-6\">
            <h2 className=\"text-xl font-semibold mb-4\">Players ({players.length}/10)</h2>
            <div className=\"space-y-2\">
              {players.map((player, index) => (
                <div
                  key={player.id}
                  className=\"flex items-center justify-between bg-gray-700 rounded-lg p-3\"
                >
                  <span>{player.name}</span>
                  {player.id === currentPlayerId && (
                    <span className=\"text-blue-400 text-sm\">(You)</span>
                  )}
                </div>
              ))}
              {players.length === 0 && (
                <div className=\"text-gray-400 text-center py-4\">
                  Waiting for players to join...
                </div>
              )}
            </div>
          </div>

          <div className=\"text-center\">
            {players.length >= 5 ? (
              <button
                onClick={startGame}
                disabled={loading}
                className=\"bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-8 py-3 rounded-lg font-semibold text-lg\"
              >
                {loading ? 'Starting...' : 'Start Game'}
              </button>
            ) : (
              <div>
                <p className=\"text-gray-400 mb-4\">
                  Need at least 5 players to start ({5 - players.length} more needed)
                </p>
                <button
                  disabled
                  className=\"bg-gray-600 px-8 py-3 rounded-lg font-semibold text-lg cursor-not-allowed\"
                >
                  Start Game
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className=\"mt-4 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200\">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className=\"min-h-screen bg-gray-900 text-white flex items-center justify-center\">
      <div className=\"bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4\">
        <h1 className=\"text-3xl font-bold text-center mb-8\">Secret Hitler</h1>
        
        <div className=\"space-y-6\">
          <div>
            <label className=\"block text-sm font-medium text-gray-400 mb-2\">
              Your Name
            </label>
            <input
              type=\"text\"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              className=\"w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent\"
              placeholder=\"Enter your name\"
              maxLength={20}
            />
          </div>

          <div className=\"border-t border-gray-600 pt-6\">
            <h2 className=\"text-lg font-semibold mb-4\">Create New Game</h2>
            <div className=\"space-y-4\">
              <div>
                <label className=\"block text-sm font-medium text-gray-400 mb-2\">
                  Custom Game ID (optional)
                </label>
                <input
                  type=\"text\"
                  value={gameId}
                  onChange={(e) => setGameId(e.target.value)}
                  className=\"w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent\"
                  placeholder=\"Leave empty for random ID\"
                />
              </div>
              <button
                onClick={createGame}
                disabled={loading}
                className=\"w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold\"
              >
                {loading ? 'Creating...' : 'Create Game'}
              </button>
            </div>
          </div>

          <div className=\"border-t border-gray-600 pt-6\">
            <h2 className=\"text-lg font-semibold mb-4\">Join Existing Game</h2>
            <div className=\"space-y-4\">
              <div>
                <label className=\"block text-sm font-medium text-gray-400 mb-2\">
                  Game ID
                </label>
                <input
                  type=\"text\"
                  value={gameId}
                  onChange={(e) => setGameId(e.target.value)}
                  className=\"w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent\"
                  placeholder=\"Enter game ID\"
                />
              </div>
              <button
                onClick={joinGame}
                disabled={loading}
                className=\"w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold\"
              >
                {loading ? 'Joining...' : 'Join Game'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className=\"mt-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200\">
            {error}
          </div>
        )}

        <div className=\"mt-8 text-center text-sm text-gray-400\">
          <p>Secret Hitler is a social deduction game for 5-10 players.</p>
          <p className=\"mt-1\">Liberals vs Fascists in 1930s Germany.</p>
        </div>
      </div>
    </div>
  );
};

export default Lobby;