# Run this diagnostic:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.wrappers import make_crafter_env

env = make_crafter_env(seed=42)
obs, info = env.reset()

print("="*70)
print("CRAFTER DIAGNOSTIC")
print("="*70)
print(f"Observation shape: {obs.shape}")
print(f"Observation type: {obs.dtype}")
print(f"Info keys: {info.keys()}")
print(f"Info contents: {info}")
print("="*70)

# Step and check again
obs, reward, done, truncated, info = env.step(0)
print(f"After step - Info keys: {info.keys()}")
print(f"After step - Info: {info}")

# Visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow(obs.transpose(1, 2, 0))  # CHW -> HWC
plt.title("Crafter Observation (CHW → HWC)")
plt.axis('off')
plt.savefig('crafter_observation.png', dpi=150, bbox_inches='tight')
print("\nSaved visualization to crafter_observation.png")

# Quick check - add this to your diagnostic
env = make_crafter_env(seed=42)
print("\nAction Space Analysis:")
print(f"Number of actions: {env.action_space.n}")
print("Action mapping (from Crafter docs):")
actions = ['noop', 'move_left', 'move_right', 'move_up', 'move_down', 
           'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
           'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 
           'make_iron_pickaxe', 'make_wood_sword', 'make_stone_sword', 
           'make_iron_sword']
for i, action in enumerate(actions):
    print(f"  {i}: {action}")