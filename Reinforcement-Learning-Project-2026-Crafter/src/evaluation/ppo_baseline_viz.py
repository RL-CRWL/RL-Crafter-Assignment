from stable_baselines3 import PPO
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env
import imageio


env = make_crafter_env()
model = PPO.load("../../results/PPO/ppo_baseline/ppo_baseline_model", env=env)

n_episodes = 10 

gif_dir = "ppo_gifs"
os.makedirs(gif_dir, exist_ok=True)

for episode in range(n_episodes):
    frames = []
    obs, info = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    imageio.mimsave(f"{gif_dir}/ppo_episode_{episode+1}.gif", frames, duration=0.1)

env.close()
print(f"Saved {n_episodes} episode GIFs to {gif_dir}/")
