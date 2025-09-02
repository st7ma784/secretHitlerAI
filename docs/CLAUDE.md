# Secret HITLER AI Game System Development Plan

## Overview
This document outlines the technical architecture and development roadmap for a Secret HITLER game system featuring AI agents trained using ARBO reinforcement learning on a LAMA model. The system will support both human and AI players with real-time chat functionality.

## Tech Stack

### Backend
- **FastAPI**: Python web framework for API endpoints
- **Redis**: In-memory database for game state and chat messages
- **PyTorch**: For LAMA model serving
- **Python**: Game logic and ARBO implementation
- **NumPy**: For numerical computations
- **TensorFlow**: Alternative model serving option

### Frontend
- **React**: Modern UI framework
- **Socket.IO**: Real-time communication
- **TailwindCSS**: Styling
- **TypeScript**: For type safety

### Model Serving
- **LAMA Model**: Base language model
- **ARBO RL**: Reinforcement learning framework
- **PyTorch Lightning**: Training infrastructure
- **ONNX Runtime**: For optimized model inference

## File Structure
```
secret_hitler_ai/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── game/             # Game logic
│   │   ├── rules.py      # Game rules implementation
│   │   ├── state.py      # Game state management
│   │   └── events.py     # Game event handling
│   ├── models/           # LAMA model code
│   │   ├── lama.py      # LAMA model wrapper
│   │   └── arbo.py      # ARBO RL implementation
│   └── training/         # ARBO training code
│       ├── data/         # Training data
│       └── scripts/      # Training scripts
├── frontend/
│   ├── src/
│   │   ├── components/   # React components
│   │   │   ├── GameBoard.tsx
│   │   │   ├── PlayerCard.tsx
│   │   │   └── ChatWindow.tsx
│   │   └── pages/        # Game pages
│   │       ├── Game.tsx
│   │       └── Lobby.tsx
│   └── public/
│       └── assets/       # Static assets
└── docs/
    └── CLAUDE.md         # This document
```

## Development Phases

### Phase 1: Core Game Logic (Weeks 1-2)
1. **Game Rules Implementation**
   - Create game state management system
   - Implement player roles (Liberal/Hitler/Fascist)
   - Set up turn-based gameplay mechanics
4. Game history tracking

## Development Timeline

### Week 1-2: Setup and Core Game
- Set up project structure
- Implement basic game rules
- Create initial UI
- Set up database schema

### Week 3-4: AI Integration
- Set up LAMA model
- Implement ARBO framework
- Create training environment
- Basic AI training

### Week 5-6: Multiplayer Features
- Add chat functionality
- Implement player matching
- Add spectator features
- Polish UI/UX

### Week 7: Testing and Optimization
- Playtesting
- Performance optimization
- Bug fixes
- Documentation

## Key Technical Challenges

1. **AI Decision Making**
   - Implement complex game state observation
   - Handle social deduction aspects
   - Balance short-term and long-term strategies

2. **Real-time Communication**
   - Efficient game state updates
   - Chat message handling
   - Player action synchronization

3. **Scalability**
   - Handle multiple concurrent games
   - Optimize database queries
   - Efficient memory management

## Required Files

### Backend
- `backend/api/game.py` - Game API endpoints
- `backend/game/state.py` - Game state management
- `backend/ai/model.py` - LAMA model integration
- `backend/ai/trainer.py` - ARBO implementation

### Frontend
- `frontend/src/components/GameBoard.jsx` - Main game interface
- `frontend/src/components/Chat.jsx` - Chat interface
- `frontend/src/components/PlayerList.jsx` - Player management

### Training
- `training/data/secret_hitler_env.py` - Gym environment
- `training/scripts/train.py` - Training script
- `training/config/params.yaml` - Training parameters

## Next Steps
1. Set up the project structure
2. Implement basic game rules
3. Create initial UI components
4. Begin AI model setup
