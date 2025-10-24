from stable_baselines3 import PPO
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env
import imageio

# Import the custom components from your LSTM implementation
# You'll need to import these from your PPO_lstm_achievements.py file
from agents.PPO_improv2 import AchievementTrackingWrapper, LSTMActorCriticPolicy, CrafterLSTMExtractor

# Create base environment and wrap it with achievement tracking
base_env = make_crafter_env()
env = AchievementTrackingWrapper(base_env)

# Load the LSTM model with custom policy
model = PPO.load("../../results/ppo_improv_2/ppo_lstm_model", env=env, custom_objects={
    'policy_class': LSTMActorCriticPolicy,
    'features_extractor_class': CrafterLSTMExtractor
})

n_episodes = 10 

gif_dir = "ppo_improv2_gifs"
os.makedirs(gif_dir, exist_ok=True)

for episode in range(n_episodes):
    frames = []
    obs, info = env.reset()
    
    # Reset LSTM hidden state at the start of each episode
    if hasattr(model.policy.features_extractor, 'reset_hidden_state'):
        model.policy.features_extractor.reset_hidden_state()
    
    done = False
    total_reward = 0
    achievement_count = 0
    
    while not done:
        frame = env.render()  # In newer gymnasium, mode parameter is deprecated
        if frame is not None:
            frames.append(frame)
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    
    # Count achievements at end of episode
    achievement_count = np.sum(obs['achievements'])
    
    imageio.mimsave(f"{gif_dir}/ppo_episode_{episode+1}.gif", frames, duration=0.1)
    print(f"Episode {episode+1}: Reward={total_reward:.2f}, Achievements={int(achievement_count)}")

env.close()
print(f"\nSaved {n_episodes} episode GIFs to {gif_dir}/")
