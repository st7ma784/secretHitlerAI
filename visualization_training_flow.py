import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create figure with subplots
fig = plt.figure(figsize=(20, 16))

# Define colors
colors = {
    'training': '#FF6B6B',
    'prompt': '#4ECDC4',
    'model': '#45B7D1',
    'game': '#96CEB4',
    'feedback': '#FFEAA7',
    'data': '#DDA0DD'
}

# Create main training loop diagram
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_title('Training Loop & Model Updates', fontsize=14, fontweight='bold')

# Training components
components = [
    {'name': 'Game\nSession', 'pos': (2, 8), 'color': colors['game']},
    {'name': 'Experience\nCollection', 'pos': (5, 8), 'color': colors['data']},
    {'name': 'LoRA+RLHF\nTraining', 'pos': (8, 8), 'color': colors['training']},
    {'name': 'Model\nUpdate', 'pos': (8, 5), 'color': colors['model']},
    {'name': 'Checkpoint\nSave', 'pos': (5, 5), 'color': colors['feedback']},
    {'name': 'Deploy\nUpdated AI', 'pos': (2, 5), 'color': colors['prompt']}
]

# Draw components
for comp in components:
    box = FancyBboxPatch((comp['pos'][0]-0.7, comp['pos'][1]-0.5), 1.4, 1, 
                         boxstyle="round,pad=0.1", facecolor=comp['color'], 
                         edgecolor='black', linewidth=1.5)
    ax1.add_patch(box)
    ax1.text(comp['pos'][0], comp['pos'][1], comp['name'], ha='center', va='center', 
             fontsize=9, fontweight='bold')

# Draw arrows for training loop
arrows = [
    ((2.7, 8), (4.3, 8)),    # Game -> Experience
    ((5.7, 8), (7.3, 8)),    # Experience -> Training
    ((8, 7.5), (8, 5.5)),    # Training -> Model
    ((7.3, 5), (5.7, 5)),    # Model -> Checkpoint
    ((4.3, 5), (2.7, 5)),    # Checkpoint -> Deploy
    ((2, 5.5), (2, 7.5))     # Deploy -> Game (complete loop)
]

for start, end in arrows:
    ax1.annotate('', xy=end, xytext=start, 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# Create prompt generation flow
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_title('Per-Turn Prompt Generation', fontsize=14, fontweight='bold')

prompt_flow = [
    {'name': 'Game State\nUpdate', 'pos': (2, 9), 'color': colors['game']},
    {'name': 'World View\nAnalysis', 'pos': (5, 9), 'color': colors['data']},
    {'name': 'Strategic\nContext', 'pos': (8, 9), 'color': colors['prompt']},
    {'name': 'Role-Specific\nGoals', 'pos': (1, 6), 'color': colors['model']},
    {'name': 'Player\nProfiles', 'pos': (3.5, 6), 'color': colors['data']},
    {'name': 'Threat\nAssessment', 'pos': (6, 6), 'color': colors['feedback']},
    {'name': 'Opportunity\nIdentification', 'pos': (8.5, 6), 'color': colors['training']},
    {'name': 'Custom Prompt\nGeneration', 'pos': (5, 3), 'color': colors['prompt']},
    {'name': 'LLM\nInference', 'pos': (2, 1), 'color': colors['model']},
    {'name': 'Action\nDecision', 'pos': (8, 1), 'color': colors['game']}
]

# Draw prompt flow components
for comp in prompt_flow:
    box = FancyBboxPatch((comp['pos'][0]-0.6, comp['pos'][1]-0.4), 1.2, 0.8, 
                         boxstyle="round,pad=0.05", facecolor=comp['color'], 
                         edgecolor='black', linewidth=1)
    ax2.add_patch(box)
    ax2.text(comp['pos'][0], comp['pos'][1], comp['name'], ha='center', va='center', 
             fontsize=8, fontweight='bold')

# Draw prompt flow arrows
prompt_arrows = [
    ((2.6, 9), (4.4, 9)),     # Game State -> World View
    ((5.6, 9), (7.4, 9)),     # World View -> Strategic Context
    ((8, 8.6), (8.5, 6.4)),   # Strategic -> Opportunities
    ((7.4, 9), (6, 6.4)),     # Strategic -> Threats
    ((4.4, 9), (3.5, 6.4)),   # World View -> Player Profiles
    ((2, 8.6), (1, 6.4)),     # Game State -> Role Goals
    # All converge to prompt generation
    ((1, 5.6), (4.2, 3.6)),
    ((3.5, 5.6), (4.6, 3.6)),
    ((6, 5.6), (5.4, 3.6)),
    ((8.5, 5.6), (5.8, 3.6)),
    # From prompt to inference and decision
    ((4.4, 3), (2.6, 1.4)),
    ((5.6, 3), (7.4, 1.4))
]

for start, end in prompt_arrows:
    ax2.annotate('', xy=end, xytext=start, 
                arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))

ax2.set_xticks([])
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# Create role-specific training visualization
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_title('Role-Specific Model Training', fontsize=14, fontweight='bold')

# Draw three role models
roles = [
    {'name': 'Liberal\nAgent', 'pos': (2, 7), 'color': '#87CEEB'},
    {'name': 'Fascist\nAgent', 'pos': (5, 7), 'color': '#CD5C5C'},
    {'name': 'Hitler\nAgent', 'pos': (8, 7), 'color': '#8B0000'}
]

for role in roles:
    box = FancyBboxPatch((role['pos'][0]-0.8, role['pos'][1]-0.6), 1.6, 1.2, 
                         boxstyle="round,pad=0.1", facecolor=role['color'], 
                         edgecolor='black', linewidth=2)
    ax3.add_patch(box)
    ax3.text(role['pos'][0], role['pos'][1], role['name'], ha='center', va='center', 
             fontsize=10, fontweight='bold', color='white')

# Shared training data
shared_box = FancyBboxPatch((3.5, 3.5), 3, 1, 
                           boxstyle="round,pad=0.1", facecolor=colors['data'], 
                           edgecolor='black', linewidth=2)
ax3.add_patch(shared_box)
ax3.text(5, 4, 'Shared Training Data\n(Game Outcomes, Actions, Rewards)', 
         ha='center', va='center', fontsize=9, fontweight='bold')

# Individual checkpoints
checkpoints = [
    {'name': 'Liberal\nCheckpoint', 'pos': (2, 1), 'color': '#87CEEB'},
    {'name': 'Fascist\nCheckpoint', 'pos': (5, 1), 'color': '#CD5C5C'},
    {'name': 'Hitler\nCheckpoint', 'pos': (8, 1), 'color': '#8B0000'}
]

for cp in checkpoints:
    box = FancyBboxPatch((cp['pos'][0]-0.7, cp['pos'][1]-0.4), 1.4, 0.8, 
                         boxstyle="round,pad=0.05", facecolor=cp['color'], 
                         edgecolor='black', linewidth=1)
    ax3.add_patch(box)
    ax3.text(cp['pos'][0], cp['pos'][1], cp['name'], ha='center', va='center', 
             fontsize=8, fontweight='bold', color='white')

# Training flow arrows
for i, role in enumerate(roles):
    # From shared data to each role
    ax3.annotate('', xy=(role['pos'][0], 6.4), xytext=(5, 4.5), 
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    # From each role to its checkpoint
    ax3.annotate('', xy=(checkpoints[i]['pos'][0], 1.4), xytext=(role['pos'][0], 6.4), 
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

ax3.set_xticks([])
ax3.set_yticks([])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

# Create training metrics visualization
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Training Progress Metrics', fontsize=14, fontweight='bold')

# Simulate training progress data
episodes = np.arange(0, 100, 5)
reward_liberal = 0.3 + 0.4 * (1 - np.exp(-episodes/20)) + np.random.normal(0, 0.02, len(episodes))
reward_fascist = 0.2 + 0.5 * (1 - np.exp(-episodes/25)) + np.random.normal(0, 0.02, len(episodes))
reward_hitler = 0.1 + 0.6 * (1 - np.exp(-episodes/30)) + np.random.normal(0, 0.03, len(episodes))

ax4.plot(episodes, reward_liberal, 'o-', color='#87CEEB', linewidth=2, label='Liberal Agent', markersize=4)
ax4.plot(episodes, reward_fascist, 's-', color='#CD5C5C', linewidth=2, label='Fascist Agent', markersize=4)
ax4.plot(episodes, reward_hitler, '^-', color='#8B0000', linewidth=2, label='Hitler Agent', markersize=4)

ax4.set_xlabel('Training Episodes', fontsize=12)
ax4.set_ylabel('Average Reward', fontsize=12)
ax4.set_ylim(0, 1)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Add performance annotations
ax4.annotate('Liberal: Trust-building\nand detection skills', 
             xy=(80, reward_liberal[-3]), xytext=(60, 0.8),
             arrowprops=dict(arrowstyle='->', color='#87CEEB', lw=1.5),
             fontsize=9, ha='center')

ax4.annotate('Fascist: Deception\nand coordination', 
             xy=(85, reward_fascist[-2]), xytext=(40, 0.3),
             arrowprops=dict(arrowstyle='->', color='#CD5C5C', lw=1.5),
             fontsize=9, ha='center')

ax4.annotate('Hitler: Identity hiding\nand timing', 
             xy=(90, reward_hitler[-1]), xytext=(70, 0.1),
             arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.5),
             fontsize=9, ha='center')

plt.tight_layout()
plt.suptitle('Secret Hitler AI: Language Model Tuning and Prompt Generation System', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/home/user/CascadeProjects/secretHitlerAI/docs/training_flow_diagram.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()

print("Training flow visualization created!")
print("\nKey Components Illustrated:")
print("1. Continuous Training Loop - Shows how models are updated from game experience")
print("2. Per-Turn Prompt Generation - Demonstrates dynamic context building")
print("3. Role-Specific Training - Illustrates specialized model adaptations")
print("4. Training Progress - Shows performance improvements over time")
