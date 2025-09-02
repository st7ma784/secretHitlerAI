import React, { useState, useEffect } from 'react';
import GameBoard from '../components/GameBoard';

interface GameProps {
  gameId: string;
  playerId: string;
  playerName: string;
}

const Game: React.FC<GameProps> = ({ gameId, playerId, playerName }) => {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    // Check connection to game
    checkGameConnection();
  }, [gameId, playerId]);

  const checkGameConnection = async () => {
    try {
      const response = await fetch(`/api/games/${gameId}/state?player_id=${playerId}`);
      if (response.ok) {
        setConnected(true);
        setError('');
      } else {
        setError('Failed to connect to game');
      }
    } catch (err) {
      setError('Network error');
    }
  };

  const handleGameAction = async (action: string, data: any) => {
    try {
      let endpoint = '';
      let body = {};

      switch (action) {
        case 'nominate':
          endpoint = `/api/games/${gameId}/nominate?player_id=${playerId}`;
          body = { chancellor_id: data.chancellor_id };
          break;
        case 'vote':
          endpoint = `/api/games/${gameId}/vote?player_id=${playerId}`;
          body = { vote: data.vote };
          break;
        case 'discard':
          endpoint = `/api/games/${gameId}/discard?player_id=${playerId}`;
          body = { policy_id: data.policy_id };
          break;
        case 'enact':
          endpoint = `/api/games/${gameId}/enact?player_id=${playerId}`;
          body = { policy_id: data.policy_id };
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Action failed');
      }

      // Action successful, game state will be updated on next poll
      setError('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  if (error) {
    return (
      <div className=\"min-h-screen bg-gray-900 text-white flex items-center justify-center\">
        <div className=\"text-center\">
          <h1 className=\"text-2xl font-bold mb-4 text-red-400\">Connection Error</h1>
          <p className=\"text-gray-400 mb-4\">{error}</p>
          <button
            onClick={checkGameConnection}
            className=\"bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded font-semibold\"
          >
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!connected) {
    return (
      <div className=\"min-h-screen bg-gray-900 text-white flex items-center justify-center\">
        <div className=\"text-center\">
          <h1 className=\"text-2xl font-bold mb-4\">Connecting to Game...</h1>
          <div className=\"animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto\"></div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <GameBoard 
        gameId={gameId} 
        playerId={playerId} 
        onAction={handleGameAction} 
      />
    </div>
  );
};

export default Game;