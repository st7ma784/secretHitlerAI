import React, { useState, useEffect } from 'react';

interface ModeSelectionProps {
  onModeSelect: (mode: 'train' | 'play', config: GameConfig) => void;
}

interface GameConfig {
  numBots: number;
  difficulty: 'easy' | 'medium' | 'hard';
  enableTraining: boolean;
  gamesPerSession?: number;
  trainingInterval?: number;
}

interface TrainingStatus {
  training_active: boolean;
  last_training_time: string | null;
  next_training_time: string | null;
  total_training_sessions: number;
  total_training_games: number;
  agent_count: number;
  live_game_buffer_size: number;
  experience_buffer_size: number;
}

const ModeSelection: React.FC<ModeSelectionProps> = ({ onModeSelect }) => {
  const [selectedMode, setSelectedMode] = useState<'train' | 'play' | null>(null);
  const [numBots, setNumBots] = useState(6);
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [enableTraining, setEnableTraining] = useState(true);
  const [gamesPerSession, setGamesPerSession] = useState(20);
  const [trainingInterval, setTrainingInterval] = useState(30);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchTrainingStatus();
    const interval = setInterval(fetchTrainingStatus, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch('/api/ai/training-status');
      if (response.ok) {
        const data = await response.json();
        setTrainingStatus(data);
      }
    } catch (err) {
      console.error('Failed to fetch training status:', err);
    }
  };

  const handleStartTraining = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai/start-training', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_agents: numBots,
          games_per_session: gamesPerSession,
          training_interval_minutes: trainingInterval,
          enable_live_learning: enableTraining
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to start training');
      }

      const config: GameConfig = {
        numBots,
        difficulty,
        enableTraining: true,
        gamesPerSession,
        trainingInterval
      };

      onModeSelect('train', config);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await fetch('/api/ai/stop-training', {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to stop training');
      }

      await fetchTrainingStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handlePlayMode = () => {
    const config: GameConfig = {
      numBots,
      difficulty,
      enableTraining: false
    };

    onModeSelect('play', config);
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const formatDuration = (minutes: number) => {
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Secret Hitler AI</h1>
        
        {/* Training Status Dashboard */}
        {trainingStatus && (
          <div className="bg-gray-800 rounded-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold mb-4">Training Status</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">Status</div>
                <div className={`text-lg font-semibold ${trainingStatus.training_active ? 'text-green-400' : 'text-red-400'}`}>
                  {trainingStatus.training_active ? 'Active' : 'Inactive'}
                </div>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">Total Sessions</div>
                <div className="text-lg font-semibold">{trainingStatus.total_training_sessions}</div>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">Total Games</div>
                <div className="text-lg font-semibold">{trainingStatus.total_training_games}</div>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">AI Agents</div>
                <div className="text-lg font-semibold">{trainingStatus.agent_count}</div>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">Last Training</div>
                <div className="text-sm">{formatDateTime(trainingStatus.last_training_time)}</div>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <div className="text-sm text-gray-400">Next Training</div>
                <div className="text-sm">{formatDateTime(trainingStatus.next_training_time)}</div>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Training Mode */}
          <div className="bg-gray-800 rounded-lg p-8">
            <h2 className="text-2xl font-semibold mb-6 text-center">Training Mode</h2>
            <p className="text-gray-300 mb-6 text-center">
              Train AI agents through self-play with continuous learning and WandB logging
            </p>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Number of AI Agents
                </label>
                <input
                  type="range"
                  min="4"
                  max="8"
                  value={numBots}
                  onChange={(e) => setNumBots(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm text-gray-400 mt-1">{numBots} agents</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Games per Training Session
                </label>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={gamesPerSession}
                  onChange={(e) => setGamesPerSession(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm text-gray-400 mt-1">{gamesPerSession} games</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Training Interval
                </label>
                <input
                  type="range"
                  min="15"
                  max="120"
                  step="15"
                  value={trainingInterval}
                  onChange={(e) => setTrainingInterval(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm text-gray-400 mt-1">{formatDuration(trainingInterval)}</div>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-400">
                  Enable Live Learning
                </label>
                <input
                  type="checkbox"
                  checked={enableTraining}
                  onChange={(e) => setEnableTraining(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                />
              </div>
            </div>

            <div className="space-y-3">
              {!trainingStatus?.training_active ? (
                <button
                  onClick={handleStartTraining}
                  disabled={loading}
                  className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold"
                >
                  {loading ? 'Starting Training...' : 'Start Training'}
                </button>
              ) : (
                <button
                  onClick={handleStopTraining}
                  disabled={loading}
                  className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold"
                >
                  {loading ? 'Stopping Training...' : 'Stop Training'}
                </button>
              )}
            </div>
          </div>

          {/* Play Mode */}
          <div className="bg-gray-800 rounded-lg p-8">
            <h2 className="text-2xl font-semibold mb-6 text-center">Play Mode</h2>
            <p className="text-gray-300 mb-6 text-center">
              Play games with AI bots of varying difficulty levels
            </p>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Number of AI Bots
                </label>
                <input
                  type="range"
                  min="3"
                  max="9"
                  value={numBots}
                  onChange={(e) => setNumBots(parseInt(e.target.value))}
                  className="w-full"
                />
                <div className="text-center text-sm text-gray-400 mt-1">{numBots} bots + you</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">
                  Bot Difficulty
                </label>
                <select
                  value={difficulty}
                  onChange={(e) => setDifficulty(e.target.value as 'easy' | 'medium' | 'hard')}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="easy">Easy - Basic strategies</option>
                  <option value="medium">Medium - Balanced play</option>
                  <option value="hard">Hard - Advanced tactics</option>
                </select>
              </div>
            </div>

            <button
              onClick={handlePlayMode}
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-6 py-3 rounded-lg font-semibold"
            >
              Start Playing
            </button>
          </div>
        </div>

        {/* WandB Integration Info */}
        <div className="bg-gray-800 rounded-lg p-6 mt-8">
          <h3 className="text-lg font-semibold mb-3">WandB Integration</h3>
          <p className="text-gray-300 text-sm mb-3">
            Training sessions are automatically logged to Weights & Biases for experiment tracking and visualization.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-700 p-3 rounded">
              <div className="font-medium text-blue-400">Game Metrics</div>
              <div className="text-gray-400">Win rates, game duration, player actions</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="font-medium text-green-400">Training Progress</div>
              <div className="text-gray-400">Session results, agent improvements</div>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <div className="font-medium text-purple-400">Model Performance</div>
              <div className="text-gray-400">Loss curves, learning rates, convergence</div>
            </div>
          </div>
        </div>

        {error && (
          <div className="mt-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModeSelection;
