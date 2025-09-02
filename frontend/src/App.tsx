import React, { useState } from 'react';
import ModeSelection from './pages/ModeSelection';
import Lobby from './pages/Lobby';
import Game from './pages/Game';
import './App.css';

type AppState = 'mode-selection' | 'lobby' | 'game';

interface GameInfo {
  gameId: string;
  playerId: string;
  playerName: string;
}

interface GameConfig {
  numBots: number;
  difficulty: 'easy' | 'medium' | 'hard';
  enableTraining: boolean;
  gamesPerSession?: number;
  trainingInterval?: number;
}

function App() {
  const [appState, setAppState] = useState<AppState>('mode-selection');
  const [gameInfo, setGameInfo] = useState<GameInfo | null>(null);
  const [gameConfig, setGameConfig] = useState<GameConfig | null>(null);

  const handleModeSelect = (mode: 'train' | 'play', config: GameConfig) => {
    setGameConfig(config);
    if (mode === 'train') {
      // For training mode, we might want to show a different interface
      // For now, we'll just go to lobby
      setAppState('lobby');
    } else {
      setAppState('lobby');
    }
  };

  const handleGameStart = (gameId: string, playerId: string, playerName: string) => {
    setGameInfo({ gameId, playerId, playerName });
    setAppState('game');
  };

  const handleBackToLobby = () => {
    setGameInfo(null);
    setAppState('lobby');
  };

  const handleBackToModeSelection = () => {
    setGameInfo(null);
    setGameConfig(null);
    setAppState('mode-selection');
  };

  return (
    <div className="App">
      {appState === 'mode-selection' && (
        <ModeSelection onModeSelect={handleModeSelect} />
      )}
      
      {appState === 'lobby' && (
        <Lobby 
          onGameStart={handleGameStart} 
          onBackToModeSelection={handleBackToModeSelection}
          gameConfig={gameConfig}
        />
      )}
      
      {appState === 'game' && gameInfo && (
        <Game
          gameId={gameInfo.gameId}
          playerId={gameInfo.playerId}
          playerName={gameInfo.playerName}
          onBackToLobby={handleBackToLobby}
        />
      )}
    </div>
  );
}

export default App;