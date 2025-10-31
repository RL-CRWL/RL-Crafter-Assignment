"""
Script to explore and understand the Crafter environment
"""
import numpy as np
from src.utils.wrappers import make_crafter_env

def explore_environment():
    """Explore the Crafter environment structure"""
    
    # Create environment using the provided wrapper
    env = make_crafter_env()
    
    print("="*60)
    print("CRAFTER ENVIRONMENT ANALYSIS")
    print("="*60)
    
    # Observation space
    print("\n1. OBSERVATION SPACE:")
    print(f"   Type: {env.observation_space}")
    print(f"   Shape: {env.observation_space.shape}")
    print(f"   Data type: {env.observation_space.dtype}")
    
    # Action space
    print("\n2. ACTION SPACE:")
    print(f"   Type: {env.action_space}")
    print(f"   Number of actions: {env.action_space.n}")
    
    # Action meanings (from Crafter documentation)
    actions = [
        "noop", "move_left", "move_right", "move_up", "move_down",
        "do", "sleep", "place_stone", "place_table", "place_furnace",
        "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
        "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
        "make_iron_sword"
    ]
    print("\n3. ACTIONS:")
    for i, action in enumerate(actions):
        print(f"   {i}: {action}")
    
    # Reset and get initial observation
    obs, info = env.reset()
    print("\n4. INITIAL STATE:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation range: [{obs.min()}, {obs.max()}]")
    print(f"   Info keys: {info.keys() if info else 'None'}")
    
    # Run a few steps
    print("\n5. SAMPLE EPISODE (10 steps):")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
        if step < 3:  # Show first 3 steps in detail
            print(f"   Step {step+1}: action={actions[action]}, reward={reward}, done={done}")
        
        if done:
            print(f"   Episode ended at step {step+1}")
            break
    
    print(f"\n   Total reward: {total_reward}")
    
    # Test episode statistics
    print("\n6. RUNNING TEST EPISODE (max 100 steps)...")
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    print(f"   Episode length: {episode_length}")
    print(f"   Episode reward: {episode_reward}")
    
    env.close()
    print("\n" + "="*60)
    print("Environment exploration complete!")
    print("="*60)

if __name__ == "__main__":
    explore_environment()