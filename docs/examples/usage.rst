Usage Examples
==============

This section provides practical examples of using the Secret Hitler AI system for various scenarios.

Basic Training Example
----------------------

**Starting a Simple Training Session**

.. code-block:: python

   import requests
   import json

   # Configure training parameters
   config = {
       "num_agents": 6,
       "games_per_session": 20,
       "training_interval_minutes": 30,
       "enable_live_learning": True,
       "lora_rank": 16,
       "training_method": "rlhf"
   }

   # Update training configuration
   response = requests.post(
       "http://localhost:8000/api/ai/configure-training",
       json=config
   )
   print(f"Configuration updated: {response.json()}")

   # Start training
   response = requests.post(
       "http://localhost:8000/api/ai/start-training",
       json={"session_name": "basic_training", "max_games": 100}
   )
   print(f"Training started: {response.json()}")

**Monitoring Training Progress**

.. code-block:: python

   import time
   import requests

   def monitor_training():
       while True:
           # Get training status
           status_response = requests.get(
               "http://localhost:8000/api/ai/training-status"
           )
           status = status_response.json()
           
           if not status["is_training"]:
               print("Training completed!")
               break
               
           # Get training metrics
           metrics_response = requests.get(
               "http://localhost:8000/api/ai/training-metrics"
           )
           metrics = metrics_response.json()
           
           print(f"Games completed: {status['total_games_played']}")
           print(f"Training loss: {metrics['training_loss']:.4f}")
           print(f"Reward score: {metrics['reward_score']:.4f}")
           print("---")
           
           time.sleep(30)  # Check every 30 seconds

   monitor_training()

Advanced Training Configuration
-------------------------------

**Role-Specific Training**

.. code-block:: python

   # Configure different LoRA ranks for different roles
   liberal_config = {
       "num_agents": 6,
       "games_per_session": 50,
       "lora_rank": 16,  # Lower rank for simpler liberal strategy
       "training_method": "rlhf",
       "learning_rate": 1e-4,
       "target_role": "liberal"
   }

   fascist_config = {
       "num_agents": 6,
       "games_per_session": 50,
       "lora_rank": 32,  # Higher rank for complex deception
       "training_method": "rlhf",
       "learning_rate": 5e-5,
       "target_role": "fascist"
   }

   hitler_config = {
       "num_agents": 6,
       "games_per_session": 30,
       "lora_rank": 64,  # Highest rank for maximum complexity
       "training_method": "rlhf",
       "learning_rate": 2e-5,
       "target_role": "hitler"
   }

**Curriculum Learning Setup**

.. code-block:: python

   curriculum_stages = [
       {
           "stage": "beginner",
           "games_per_session": 10,
           "difficulty": "easy",
           "lora_rank": 8,
           "learning_rate": 1e-3
       },
       {
           "stage": "intermediate", 
           "games_per_session": 20,
           "difficulty": "medium",
           "lora_rank": 16,
           "learning_rate": 5e-4
       },
       {
           "stage": "advanced",
           "games_per_session": 50,
           "difficulty": "hard",
           "lora_rank": 32,
           "learning_rate": 1e-4
       }
   ]

   for stage in curriculum_stages:
       print(f"Starting {stage['stage']} training...")
       
       # Configure for this stage
       requests.post(
           "http://localhost:8000/api/ai/configure-training",
           json=stage
       )
       
       # Start training for this stage
       requests.post(
           "http://localhost:8000/api/ai/start-training",
           json={
               "session_name": f"{stage['stage']}_curriculum",
               "max_games": stage["games_per_session"]
           }
       )
       
       # Wait for completion (simplified)
       # In practice, you'd monitor status properly
       time.sleep(stage["games_per_session"] * 60)  # Rough estimate

Checkpoint Management
---------------------

**Saving and Loading Checkpoints**

.. code-block:: python

   # Save a checkpoint after good performance
   def save_milestone_checkpoint(performance_threshold=0.8):
       metrics = requests.get("http://localhost:8000/api/ai/training-metrics").json()
       
       if metrics["reward_score"] >= performance_threshold:
           checkpoint_data = {
               "name": f"milestone_{metrics['reward_score']:.3f}",
               "description": f"High-performance checkpoint with {metrics['reward_score']:.3f} reward",
               "role": "liberal",  # or determine dynamically
               "tags": ["milestone", "high-performance"]
           }
           
           response = requests.post(
               "http://localhost:8000/api/ai/checkpoints",
               json=checkpoint_data
           )
           
           print(f"Checkpoint saved: {response.json()}")
           return response.json()["checkpoint_id"]
       
       return None

   # Load a specific checkpoint
   def load_checkpoint(checkpoint_id):
       response = requests.post(
           f"http://localhost:8000/api/ai/checkpoints/{checkpoint_id}/load"
       )
       
       if response.status_code == 200:
           print(f"Checkpoint loaded successfully: {response.json()}")
           return True
       else:
           print(f"Failed to load checkpoint: {response.text}")
           return False

   # List available checkpoints
   def list_checkpoints(role=None):
       params = {"role": role} if role else {}
       response = requests.get(
           "http://localhost:8000/api/ai/checkpoints",
           params=params
       )
       
       checkpoints = response.json()["checkpoints"]
       for cp in checkpoints:
           print(f"ID: {cp['id']}, Role: {cp['role']}, Performance: {cp['performance']:.3f}")
       
       return checkpoints

**Checkpoint Cleanup Strategy**

.. code-block:: python

   def cleanup_old_checkpoints(keep_top_n=5, min_performance=0.6):
       # Get all checkpoints
       checkpoints = requests.get("http://localhost:8000/api/ai/checkpoints").json()["checkpoints"]
       
       # Sort by performance (descending)
       sorted_checkpoints = sorted(checkpoints, key=lambda x: x["performance"], reverse=True)
       
       # Keep top N and those above minimum performance
       to_keep = set()
       
       # Keep top N
       for cp in sorted_checkpoints[:keep_top_n]:
           to_keep.add(cp["id"])
       
       # Keep high-performance ones
       for cp in checkpoints:
           if cp["performance"] >= min_performance:
               to_keep.add(cp["id"])
       
       # Delete the rest
       for cp in checkpoints:
           if cp["id"] not in to_keep:
               response = requests.delete(f"http://localhost:8000/api/ai/checkpoints/{cp['id']}")
               if response.status_code == 200:
                   print(f"Deleted checkpoint {cp['id']} (performance: {cp['performance']:.3f})")

Game Interaction Examples
-------------------------

**Creating and Managing Games**

.. code-block:: python

   # Create a new game
   def create_training_game():
       game_config = {
           "num_players": 6,
           "ai_players": 6,  # All AI for training
           "difficulty": "medium",
           "game_mode": "training"
       }
       
       response = requests.post(
           "http://localhost:8000/game/create",
           json=game_config
       )
       
       game_data = response.json()
       print(f"Created game: {game_data['game_id']}")
       return game_data["game_id"]

   # Monitor game progress
   def monitor_game(game_id):
       while True:
           response = requests.get(f"http://localhost:8000/game/{game_id}/state")
           game_state = response.json()
           
           print(f"Game {game_id}: Phase {game_state['phase']}, Round {game_state['round']}")
           print(f"Liberal policies: {game_state['liberal_policies']}")
           print(f"Fascist policies: {game_state['fascist_policies']}")
           
           if game_state.get("game_over"):
               print(f"Game ended: {game_state['winner']} victory!")
               break
               
           time.sleep(5)

**Human vs AI Games**

.. code-block:: python

   # Create a mixed human/AI game
   def create_mixed_game():
       game_config = {
           "num_players": 6,
           "ai_players": 4,  # 4 AI, 2 human slots
           "difficulty": "hard",
           "game_mode": "competitive"
       }
       
       response = requests.post(
           "http://localhost:8000/game/create",
           json=game_config
       )
       
       game_id = response.json()["game_id"]
       
       # Join as human player
       join_response = requests.post(
           f"http://localhost:8000/game/{game_id}/join",
           json={
               "player_name": "Human Player 1",
               "player_type": "human"
           }
       )
       
       player_id = join_response.json()["player_id"]
       print(f"Joined game {game_id} as player {player_id}")
       
       return game_id, player_id

   # Make a human player action
   def make_player_action(game_id, player_id, action_type, parameters):
       action_data = {
           "player_id": player_id,
           "action_type": action_type,
           "parameters": parameters
       }
       
       response = requests.post(
           f"http://localhost:8000/game/{game_id}/action",
           json=action_data
       )
       
       return response.json()

Agent Performance Analysis
--------------------------

**Analyzing Agent Performance**

.. code-block:: python

   def analyze_agent_performance(agent_id):
       response = requests.get(f"http://localhost:8000/agents/{agent_id}/performance")
       performance = response.json()
       
       print(f"Agent {agent_id} Performance Analysis:")
       print(f"Games played: {performance['overall_stats']['games_played']}")
       print(f"Win rate: {performance['overall_stats']['win_rate']:.3f}")
       print(f"Average game duration: {performance['overall_stats']['average_game_duration']}")
       
       print("\\nRole Performance:")
       for role, stats in performance['role_performance'].items():
           print(f"  {role}: {stats['success_rate']:.3f} success rate")
       
       print("\\nStrategic Metrics:")
       for metric, value in performance['strategic_metrics'].items():
           print(f"  {metric}: {value:.3f}")
       
       return performance

   # Compare multiple agents
   def compare_agents(agent_ids):
       performances = {}
       for agent_id in agent_ids:
           performances[agent_id] = analyze_agent_performance(agent_id)
       
       print("\\nAgent Comparison:")
       print("Agent ID\\t\\tWin Rate\\tDeception Detection\\tTrust Building")
       print("-" * 70)
       
       for agent_id, perf in performances.items():
           win_rate = perf['overall_stats']['win_rate']
           deception = perf['strategic_metrics']['deception_detection']
           trust = perf['strategic_metrics']['trust_building']
           print(f"{agent_id}\\t\\t{win_rate:.3f}\\t\\t{deception:.3f}\\t\\t\\t{trust:.3f}")

Custom Training Scenarios
--------------------------

**Adversarial Training**

.. code-block:: python

   def setup_adversarial_training():
       # Train liberal agents against strong fascist agents
       liberal_session = {
           "session_name": "liberal_adversarial",
           "target_role": "liberal",
           "opponent_strength": "expert",
           "games_per_session": 100,
           "focus_areas": ["investigation", "trust_building", "coalition_formation"]
       }
       
       # Train fascist agents against strong liberal agents  
       fascist_session = {
           "session_name": "fascist_adversarial",
           "target_role": "fascist",
           "opponent_strength": "expert", 
           "games_per_session": 100,
           "focus_areas": ["deception", "misdirection", "chaos_creation"]
       }
       
       # Alternate between sessions
       for session in [liberal_session, fascist_session]:
           print(f"Starting {session['session_name']}...")
           requests.post("http://localhost:8000/api/ai/start-training", json=session)
           # Wait for completion...

**Meta-Learning Setup**

.. code-block:: python

   def setup_meta_learning():
       # Train agents to adapt to different opponent strategies
       meta_config = {
           "training_method": "meta_learning",
           "adaptation_steps": 5,
           "meta_batch_size": 10,
           "inner_learning_rate": 1e-3,
           "outer_learning_rate": 1e-4,
           "strategy_variations": [
               "aggressive_liberal",
               "passive_liberal", 
               "chaotic_fascist",
               "subtle_fascist",
               "paranoid_hitler",
               "confident_hitler"
           ]
       }
       
       requests.post("http://localhost:8000/api/ai/configure-training", json=meta_config)

Batch Processing and Automation
-------------------------------

**Automated Training Pipeline**

.. code-block:: python

   import schedule
   import time
   from datetime import datetime

   class TrainingPipeline:
       def __init__(self):
           self.current_stage = 0
           self.stages = [
               {"name": "foundation", "games": 50, "rank": 8},
               {"name": "intermediate", "games": 100, "rank": 16},
               {"name": "advanced", "games": 200, "rank": 32},
               {"name": "expert", "games": 500, "rank": 64}
           ]
       
       def run_stage(self, stage_idx):
           stage = self.stages[stage_idx]
           print(f"Starting {stage['name']} stage at {datetime.now()}")
           
           # Configure training
           config = {
               "games_per_session": stage["games"],
               "lora_rank": stage["rank"],
               "training_method": "rlhf"
           }
           
           requests.post("http://localhost:8000/api/ai/configure-training", json=config)
           
           # Start training
           session_data = {
               "session_name": f"pipeline_{stage['name']}",
               "max_games": stage["games"]
           }
           
           requests.post("http://localhost:8000/api/ai/start-training", json=session_data)
           
           # Save checkpoint after completion
           self.save_stage_checkpoint(stage["name"])
       
       def save_stage_checkpoint(self, stage_name):
           checkpoint_data = {
               "name": f"pipeline_{stage_name}",
               "description": f"Checkpoint from automated pipeline - {stage_name} stage",
               "tags": ["pipeline", stage_name, "automated"]
           }
           
           requests.post("http://localhost:8000/api/ai/checkpoints", json=checkpoint_data)
       
       def schedule_pipeline(self):
           # Schedule different stages
           schedule.every().day.at("02:00").do(self.run_stage, 0)  # Foundation
           schedule.every().day.at("08:00").do(self.run_stage, 1)  # Intermediate  
           schedule.every().day.at("14:00").do(self.run_stage, 2)  # Advanced
           schedule.every().day.at("20:00").do(self.run_stage, 3)  # Expert
           
           while True:
               schedule.run_pending()
               time.sleep(60)

   # Usage
   pipeline = TrainingPipeline()
   pipeline.schedule_pipeline()

**Batch Performance Evaluation**

.. code-block:: python

   def batch_evaluate_agents():
       # Get all agents
       agents_response = requests.get("http://localhost:8000/agents")
       agents = agents_response.json()["agents"]
       
       results = []
       
       for agent in agents:
           # Get detailed performance
           perf_response = requests.get(f"http://localhost:8000/agents/{agent['id']}/performance")
           performance = perf_response.json()
           
           # Calculate composite score
           composite_score = (
               performance['overall_stats']['win_rate'] * 0.4 +
               performance['strategic_metrics']['deception_detection'] * 0.2 +
               performance['strategic_metrics']['trust_building'] * 0.2 +
               performance['strategic_metrics']['voting_accuracy'] * 0.2
           )
           
           results.append({
               "agent_id": agent["id"],
               "role": agent["role"],
               "win_rate": performance['overall_stats']['win_rate'],
               "composite_score": composite_score,
               "games_played": performance['overall_stats']['games_played']
           })
       
       # Sort by composite score
       results.sort(key=lambda x: x["composite_score"], reverse=True)
       
       print("Agent Performance Ranking:")
       print("Rank\\tAgent ID\\t\\tRole\\t\\tWin Rate\\tComposite Score\\tGames")
       print("-" * 80)
       
       for i, result in enumerate(results, 1):
           print(f"{i}\\t{result['agent_id']}\\t{result['role']}\\t\\t{result['win_rate']:.3f}\\t\\t{result['composite_score']:.3f}\\t\\t{result['games_played']}")
       
       return results

Integration Examples
--------------------

**WebSocket Real-time Updates**

.. code-block:: python

   import websocket
   import json

   def on_training_update(ws, message):
       data = json.loads(message)
       if data["type"] == "training_progress":
           progress = data["data"]
           print(f"Training progress: {progress['games_completed']} games, "
                 f"Loss: {progress['current_loss']:.4f}")

   def on_error(ws, error):
       print(f"WebSocket error: {error}")

   def on_close(ws, close_status_code, close_msg):
       print("WebSocket connection closed")

   # Connect to training updates
   ws = websocket.WebSocketApp(
       "ws://localhost:8000/ws/training",
       on_message=on_training_update,
       on_error=on_error,
       on_close=on_close
   )
   
   ws.run_forever()

**Custom Metrics Collection**

.. code-block:: python

   import csv
   from datetime import datetime

   class MetricsCollector:
       def __init__(self, output_file="training_metrics.csv"):
           self.output_file = output_file
           self.initialize_csv()
       
       def initialize_csv(self):
           with open(self.output_file, 'w', newline='') as file:
               writer = csv.writer(file)
               writer.writerow([
                   "timestamp", "training_loss", "reward_score", 
                   "liberal_win_rate", "fascist_win_rate", "games_completed"
               ])
       
       def collect_metrics(self):
           # Get current metrics
           metrics_response = requests.get("http://localhost:8000/api/ai/training-metrics")
           status_response = requests.get("http://localhost:8000/api/ai/training-status")
           
           metrics = metrics_response.json()
           status = status_response.json()
           
           # Write to CSV
           with open(self.output_file, 'a', newline='') as file:
               writer = csv.writer(file)
               writer.writerow([
                   datetime.now().isoformat(),
                   metrics["training_loss"],
                   metrics["reward_score"],
                   metrics["model_performance"]["liberal_accuracy"],
                   metrics["model_performance"]["fascist_deception"],
                   status["total_games_played"]
               ])
       
       def start_collection(self, interval=60):
           while True:
               try:
                   self.collect_metrics()
                   print(f"Metrics collected at {datetime.now()}")
               except Exception as e:
                   print(f"Error collecting metrics: {e}")
               
               time.sleep(interval)

   # Usage
   collector = MetricsCollector()
   collector.start_collection(interval=30)  # Collect every 30 seconds
