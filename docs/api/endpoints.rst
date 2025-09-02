API Endpoints
=============

This document provides comprehensive documentation for all REST API endpoints in the Secret Hitler AI system.

Base URL
--------

All API endpoints are prefixed with the base URL:

.. code-block:: text

   http://localhost:8000/api

Authentication
--------------

Currently, the API does not require authentication for local development. In production deployments, consider implementing:

- API key authentication
- JWT tokens
- OAuth 2.0

Training Endpoints
------------------

Training Status
~~~~~~~~~~~~~~~

Get the current training status and statistics.

.. http:get:: /ai/training-status

   **Response:**

   .. code-block:: json

      {
        "is_training": false,
        "current_session": null,
        "total_games_played": 1250,
        "agents_trained": 6,
        "last_training_time": "2024-01-15T10:30:00Z",
        "training_uptime": "2h 45m",
        "performance_metrics": {
          "average_win_rate": 0.67,
          "liberal_win_rate": 0.72,
          "fascist_win_rate": 0.61,
          "hitler_survival_rate": 0.45
        }
      }

   :statuscode 200: Training status retrieved successfully

Training Metrics
~~~~~~~~~~~~~~~~

Get detailed training metrics and performance data.

.. http:get:: /ai/training-metrics

   **Response:**

   .. code-block:: json

      {
        "training_loss": 0.234,
        "reward_score": 0.78,
        "policy_gradient": 0.012,
        "kl_divergence": 0.045,
        "value_function_loss": 0.156,
        "games_per_hour": 24.5,
        "training_efficiency": 0.89,
        "model_performance": {
          "liberal_accuracy": 0.84,
          "fascist_deception": 0.76,
          "hitler_stealth": 0.68
        },
        "recent_games": [
          {
            "game_id": "game_1234",
            "outcome": "liberal_victory",
            "duration_minutes": 12.5,
            "agents_performance": {...}
          }
        ]
      }

   :statuscode 200: Metrics retrieved successfully

Configure Training
~~~~~~~~~~~~~~~~~~

Update training configuration parameters.

.. http:post:: /ai/configure-training

   **Request Body:**

   .. code-block:: json

      {
        "num_agents": 6,
        "games_per_session": 20,
        "training_interval_minutes": 30,
        "enable_live_learning": true,
        "lora_rank": 16,
        "training_method": "rlhf",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "max_epochs": 10
      }

   **Response:**

   .. code-block:: json

      {
        "message": "Training configuration updated successfully",
        "config": {
          "num_agents": 6,
          "games_per_session": 20,
          "training_interval_minutes": 30,
          "enable_live_learning": true,
          "lora_rank": 16,
          "training_method": "rlhf"
        }
      }

   :statuscode 200: Configuration updated successfully
   :statuscode 400: Invalid configuration parameters

Start Training
~~~~~~~~~~~~~~

Start a new training session.

.. http:post:: /ai/start-training

   **Request Body:**

   .. code-block:: json

      {
        "session_name": "advanced_strategy_training",
        "max_games": 100,
        "target_performance": 0.8
      }

   **Response:**

   .. code-block:: json

      {
        "message": "Training started successfully",
        "session_id": "session_abc123",
        "estimated_duration": "2h 30m"
      }

   :statuscode 200: Training started successfully
   :statuscode 409: Training already in progress

Stop Training
~~~~~~~~~~~~~

Stop the current training session.

.. http:post:: /ai/stop-training

   **Response:**

   .. code-block:: json

      {
        "message": "Training stopped successfully",
        "session_summary": {
          "games_completed": 45,
          "training_time": "1h 23m",
          "final_performance": 0.76
        }
      }

   :statuscode 200: Training stopped successfully
   :statuscode 400: No training session in progress

Checkpoint Management
---------------------

List Checkpoints
~~~~~~~~~~~~~~~~

Get a list of available model checkpoints.

.. http:get:: /ai/checkpoints

   **Query Parameters:**

   - ``role`` (optional): Filter by agent role (liberal, fascist, hitler)
   - ``limit`` (optional): Maximum number of checkpoints to return (default: 50)

   **Response:**

   .. code-block:: json

      {
        "checkpoints": [
          {
            "id": "checkpoint_001",
            "role": "liberal",
            "timestamp": "2024-01-15T10:30:00Z",
            "performance": 0.82,
            "games_trained": 500,
            "file_size": "145MB",
            "metadata": {
              "training_method": "rlhf",
              "lora_rank": 16,
              "learning_rate": 1e-4
            }
          }
        ],
        "total_count": 12
      }

   :statuscode 200: Checkpoints retrieved successfully

Save Checkpoint
~~~~~~~~~~~~~~~

Save the current model state as a checkpoint.

.. http:post:: /ai/checkpoints

   **Request Body:**

   .. code-block:: json

      {
        "name": "milestone_checkpoint",
        "description": "High-performance liberal agent",
        "role": "liberal",
        "tags": ["milestone", "production-ready"]
      }

   **Response:**

   .. code-block:: json

      {
        "message": "Checkpoint saved successfully",
        "checkpoint_id": "checkpoint_002",
        "file_path": "/checkpoints/liberal_checkpoint_002.pt"
      }

   :statuscode 201: Checkpoint created successfully
   :statuscode 400: Invalid checkpoint parameters

Load Checkpoint
~~~~~~~~~~~~~~~

Load a specific checkpoint for training or inference.

.. http:post:: /ai/checkpoints/{checkpoint_id}/load

   **Response:**

   .. code-block:: json

      {
        "message": "Checkpoint loaded successfully",
        "checkpoint_info": {
          "id": "checkpoint_001",
          "role": "liberal",
          "performance": 0.82,
          "loaded_at": "2024-01-15T11:00:00Z"
        }
      }

   :statuscode 200: Checkpoint loaded successfully
   :statuscode 404: Checkpoint not found
   :statuscode 400: Failed to load checkpoint

Delete Checkpoint
~~~~~~~~~~~~~~~~~

Delete a specific checkpoint.

.. http:delete:: /ai/checkpoints/{checkpoint_id}

   **Response:**

   .. code-block:: json

      {
        "message": "Checkpoint deleted successfully"
      }

   :statuscode 200: Checkpoint deleted successfully
   :statuscode 404: Checkpoint not found

Game Management
---------------

Create Game
~~~~~~~~~~~

Create a new game session.

.. http:post:: /game/create

   **Request Body:**

   .. code-block:: json

      {
        "num_players": 6,
        "ai_players": 4,
        "difficulty": "medium",
        "game_mode": "training"
      }

   **Response:**

   .. code-block:: json

      {
        "game_id": "game_5678",
        "players": [
          {
            "id": "player_1",
            "name": "AI Agent 1",
            "type": "ai",
            "role": "hidden"
          }
        ],
        "status": "waiting_for_players"
      }

   :statuscode 201: Game created successfully

Join Game
~~~~~~~~~

Join an existing game session.

.. http:post:: /game/{game_id}/join

   **Request Body:**

   .. code-block:: json

      {
        "player_name": "Human Player",
        "player_type": "human"
      }

   **Response:**

   .. code-block:: json

      {
        "message": "Joined game successfully",
        "player_id": "player_2",
        "game_status": "ready_to_start"
      }

   :statuscode 200: Joined game successfully
   :statuscode 404: Game not found
   :statuscode 409: Game is full

Game State
~~~~~~~~~~

Get the current game state.

.. http:get:: /game/{game_id}/state

   **Response:**

   .. code-block:: json

      {
        "game_id": "game_5678",
        "phase": "election",
        "round": 2,
        "president": "player_1",
        "chancellor": null,
        "liberal_policies": 2,
        "fascist_policies": 1,
        "players": [
          {
            "id": "player_1",
            "name": "AI Agent 1",
            "alive": true,
            "investigated": false
          }
        ],
        "election_tracker": 1,
        "last_action": {
          "type": "nominate_chancellor",
          "player": "player_1",
          "target": "player_3"
        }
      }

   :statuscode 200: Game state retrieved successfully
   :statuscode 404: Game not found

Player Actions
~~~~~~~~~~~~~~

Submit a player action.

.. http:post:: /game/{game_id}/action

   **Request Body:**

   .. code-block:: json

      {
        "player_id": "player_1",
        "action_type": "vote",
        "parameters": {
          "vote": "ja"
        }
      }

   **Response:**

   .. code-block:: json

      {
        "message": "Action processed successfully",
        "game_state_updated": true,
        "next_phase": "legislative"
      }

   :statuscode 200: Action processed successfully
   :statuscode 400: Invalid action
   :statuscode 403: Action not allowed for this player

Agent Management
----------------

List Agents
~~~~~~~~~~~

Get a list of available AI agents.

.. http:get:: /agents

   **Response:**

   .. code-block:: json

      {
        "agents": [
          {
            "id": "agent_liberal_001",
            "role": "liberal",
            "performance": 0.84,
            "games_played": 1200,
            "win_rate": 0.78,
            "status": "active",
            "last_updated": "2024-01-15T10:30:00Z"
          }
        ]
      }

   :statuscode 200: Agents retrieved successfully

Agent Performance
~~~~~~~~~~~~~~~~~

Get detailed performance metrics for a specific agent.

.. http:get:: /agents/{agent_id}/performance

   **Response:**

   .. code-block:: json

      {
        "agent_id": "agent_liberal_001",
        "overall_stats": {
          "games_played": 1200,
          "win_rate": 0.78,
          "average_game_duration": "14.5 minutes"
        },
        "role_performance": {
          "as_president": {
            "times_elected": 245,
            "policies_enacted": 198,
            "success_rate": 0.81
          },
          "as_chancellor": {
            "times_elected": 189,
            "policies_enacted": 156,
            "success_rate": 0.83
          }
        },
        "strategic_metrics": {
          "deception_detection": 0.72,
          "trust_building": 0.86,
          "voting_accuracy": 0.79
        }
      }

   :statuscode 200: Performance data retrieved successfully
   :statuscode 404: Agent not found

System Information
------------------

Health Check
~~~~~~~~~~~~

Check system health and availability.

.. http:get:: /health

   **Response:**

   .. code-block:: json

      {
        "status": "healthy",
        "timestamp": "2024-01-15T11:00:00Z",
        "services": {
          "api": "healthy",
          "training": "healthy",
          "database": "healthy"
        },
        "version": "1.0.0"
      }

   :statuscode 200: System is healthy
   :statuscode 503: System is unhealthy

System Stats
~~~~~~~~~~~~

Get system resource usage and statistics.

.. http:get:: /stats

   **Response:**

   .. code-block:: json

      {
        "uptime": "5d 12h 34m",
        "memory_usage": {
          "used": "2.1GB",
          "total": "8GB",
          "percentage": 26.25
        },
        "cpu_usage": 15.7,
        "disk_usage": {
          "used": "45GB",
          "total": "100GB",
          "percentage": 45.0
        },
        "active_connections": 12,
        "total_requests": 15847
      }

   :statuscode 200: Statistics retrieved successfully

Error Handling
--------------

The API uses standard HTTP status codes and returns error information in JSON format:

.. code-block:: json

   {
     "error": {
       "code": "INVALID_PARAMETER",
       "message": "The 'num_agents' parameter must be between 5 and 10",
       "details": {
         "parameter": "num_agents",
         "provided_value": 15,
         "valid_range": [5, 10]
       }
     },
     "timestamp": "2024-01-15T11:00:00Z",
     "request_id": "req_abc123"
   }

Common Error Codes
~~~~~~~~~~~~~~~~~~

- ``400 Bad Request``: Invalid request parameters
- ``401 Unauthorized``: Authentication required
- ``403 Forbidden``: Insufficient permissions
- ``404 Not Found``: Resource not found
- ``409 Conflict``: Resource conflict (e.g., training already running)
- ``422 Unprocessable Entity``: Validation errors
- ``500 Internal Server Error``: Server error
- ``503 Service Unavailable``: Service temporarily unavailable

Rate Limiting
-------------

API endpoints are subject to rate limiting to prevent abuse:

- **Training endpoints**: 10 requests per minute
- **Game endpoints**: 100 requests per minute
- **Status endpoints**: 60 requests per minute

Rate limit headers are included in responses:

.. code-block:: text

   X-RateLimit-Limit: 60
   X-RateLimit-Remaining: 45
   X-RateLimit-Reset: 1642248000

WebSocket Endpoints
-------------------

For real-time updates, the system provides WebSocket endpoints:

Training Updates
~~~~~~~~~~~~~~~~

.. code-block:: text

   ws://localhost:8000/ws/training

Receives real-time training progress updates:

.. code-block:: json

   {
     "type": "training_progress",
     "data": {
       "games_completed": 15,
       "current_loss": 0.234,
       "estimated_time_remaining": "45 minutes"
     }
   }

Game Updates
~~~~~~~~~~~~

.. code-block:: text

   ws://localhost:8000/ws/game/{game_id}

Receives real-time game state updates:

.. code-block:: json

   {
     "type": "game_state_change",
     "data": {
       "phase": "voting",
       "current_player": "player_2",
       "action_required": "vote_on_government"
     }
   }
