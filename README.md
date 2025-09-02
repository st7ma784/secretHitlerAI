# Secret Hitler AI

A sophisticated AI training system for the Secret Hitler board game, featuring self-play training, WandB experiment tracking, and configurable bot difficulty levels.

## Features

### 🤖 AI Training System
- **Self-Play Training**: AI agents learn through continuous self-play sessions
- **Multiple Agent Types**: Diverse AI personalities (aggressive liberal, strategic fascist, hitler specialist, etc.)
- **Live Learning**: Real-time adaptation during gameplay
- **Performance Tracking**: Comprehensive metrics and improvement analysis

### 📊 WandB Integration
- **Experiment Tracking**: Automatic logging of training sessions and game metrics
- **Performance Visualization**: Win rates, game duration, player actions
- **Model Monitoring**: Loss curves, learning rates, convergence metrics
- **Game Analytics**: Detailed game summaries and player statistics

### 🎮 Training/Play Interface
- **Training Mode**: Configure and monitor AI training sessions
  - Adjustable number of AI agents (4-8)
  - Configurable games per session (10-50)
  - Training interval settings (15-120 minutes)
  - Live learning toggle
- **Play Mode**: Play against AI bots with varying difficulty
  - Easy, Medium, Hard difficulty levels
  - Configurable number of bots (3-9)
- **Real-time Status**: Live training status dashboard

## Architecture

### Backend (FastAPI)
- **Game Logic**: Complete Secret Hitler game implementation
- **AI Models**: LAMA and ARBO-based AI agents
- **Training Orchestrator**: Manages continuous learning and model updates
- **WandB Logger**: Comprehensive experiment tracking
- **REST API**: Training configuration and status endpoints

### Frontend
- **React/TypeScript**: Modern web interface (main game)
- **HTML/JavaScript**: Training interface (standalone)
- **Tailwind CSS**: Responsive, modern styling

## Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- WandB account (optional, for experiment tracking)

### Installation

1. **Backend Setup**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Frontend Setup**:
```bash
cd frontend
npm install
```

3. **WandB Configuration** (optional):
```bash
wandb login
```

### Running the Application

1. **Start Backend**:
```bash
cd backend
python -m api.game
```

2. **Start Frontend**:
```bash
cd frontend
npm start
```

3. **Access Training Interface**:
   - Open `http://localhost:3000/training-interface.html`
   - Or use the main React interface at `http://localhost:3000`

## API Endpoints

### Training Management
- `GET /api/ai/training-status` - Get current training status
- `POST /api/ai/start-training` - Start AI training
- `POST /api/ai/stop-training` - Stop AI training
- `POST /api/ai/configure-training` - Configure training parameters

### Game Management
- `POST /api/games` - Create new game
- `POST /api/games/{game_id}/join` - Join game
- `POST /api/games/{game_id}/start` - Start game
- `GET /api/games/{game_id}/players` - Get players

### AI Management
- `POST /api/games/{game_id}/ai-player` - Add AI player
- `GET /api/ai/performance` - Get AI performance metrics

## Training Configuration

### Agent Types
- **Aggressive Liberal**: High-temperature, liberal-focused strategies
- **Conservative Liberal**: Balanced liberal play
- **Strategic Fascist**: Advanced fascist tactics
- **Deceptive Fascist**: Deception-focused fascist play
- **Hitler Specialist**: Specialized Hitler gameplay
- **Balanced Player**: General-purpose strategies

### Training Parameters
- **Number of Agents**: 4-8 AI agents per training session
- **Games per Session**: 10-50 games before model updates
- **Training Interval**: 15-120 minutes between sessions
- **Live Learning**: Real-time adaptation from human games

## WandB Metrics

### Game Metrics
- Win rates by team (Liberal/Fascist/Hitler)
- Game duration and round counts
- Player action frequencies
- Confidence scores and decision quality

### Training Metrics
- Session completion rates
- Agent improvement over time
- Model convergence indicators
- Performance trends

### Model Metrics
- Loss curves and gradient norms
- Learning rate schedules
- Training step counts
- Model version tracking

## Development

### Project Structure
```
secretHitlerAI/
├── backend/
│   ├── api/           # FastAPI endpoints
│   ├── game/          # Game logic and rules
│   ├── models/        # AI models (LAMA, ARBO)
│   ├── training/      # Training orchestrator and WandB logger
│   └── requirements.txt
├── frontend/
│   ├── src/           # React components
│   ├── public/        # Static files and training interface
│   └── package.json
└── README.md
```

### Key Components
- **SelfTrainingOrchestrator**: Manages continuous AI training
- **WandBLogger**: Handles experiment tracking and metrics
- **AIPlayerManager**: Manages AI agent lifecycle
- **GameStateManager**: Handles game state and rules
- **Training Interface**: Standalone HTML interface for training control

## Documentation

Comprehensive documentation is available at: **[yourusername.github.io/secretHitlerAI](https://yourusername.github.io/secretHitlerAI)**

### Documentation Sections
- **Architecture Overview**: System design and components
- **API Reference**: Complete endpoint documentation
- **Training System**: ML training methodology and configuration
- **Docker Deployment**: Containerized deployment guide
- **Development Setup**: Local development environment
- **Usage Examples**: Practical code examples and tutorials

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8080 -d _build/html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation as needed
6. Submit a pull request

See our [Contributing Guide](https://yourusername.github.io/secretHitlerAI/contributing.html) for detailed information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Secret Hitler board game by Max Temkin, Mike Boxleiter, Tommy Maranges
- WandB for experiment tracking capabilities
- FastAPI and React communities for excellent frameworks
