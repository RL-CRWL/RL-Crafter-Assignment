from stable_baselines3 import PPO
import numpy as np
import imageio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrapper_ppo import make_crafter_env

# === Load environment and model ===
env = make_crafter_env()
model_path = 'Reinforcement-Learning-Project-2026-Crafter/results/PPO/models/ppo_improv_1/ppo_rnd_model.zip'
model = PPO.load(model_path, env=env)

# === Parameters ===
n_eval_episodes = 30   # Run more episodes to find highlights
gif_dir = "Reinforcement-Learning-Project-2026-Crafter/results/gifs/ppo_rnd_highlights"
os.makedirs(gif_dir, exist_ok=True)

episode_data = []  # store (reward, length, frames, maybe achievements)
print(f"Evaluating {n_eval_episodes} episodes...")

for ep in range(n_eval_episodes):
    obs, info = env.reset()
    frames, done = [], False
    total_reward, steps = 0, 0

    while not done:
        # In Gymnasium render() returns rgb_array by default, no mode argument needed
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    # Try to extract achievements if available (Crafter env often has them)
    achievements = None
    if isinstance(obs, dict) and 'achievements' in obs:
        achievements = int(np.sum(obs['achievements']))

    episode_data.append((total_reward, steps, frames, achievements))
    msg = f"Episode {ep+1}: Reward={total_reward:.1f}, Steps={steps}"
    if achievements is not None:
        msg += f", Achievements={achievements}"
    print(msg)

env.close()

# === Select highlights ===
rewards = [r for (r, _, _, _) in episode_data]
lengths = [l for (_, l, _, _) in episode_data]
achievements = [a for (_, _, _, a) in episode_data if a is not None]

best_reward_idx = int(np.argmax(rewards))
median_idx = int(np.argsort(rewards)[len(rewards)//2])
longest_idx = int(np.argmax(lengths))

selected_indices = sorted(set([best_reward_idx, median_idx, longest_idx]))
print("\nSelected highlight episodes:", selected_indices)

# === Save highlight GIFs ===
for idx in selected_indices:
    reward, steps, frames, ach = episode_data[idx]
    label = f"R{reward:.1f}_S{steps}"
    if ach is not None:
        label += f"_A{ach}"
    gif_path = os.path.join(gif_dir, f"highlight_ep{idx+1}_{label}.gif")
    imageio.mimsave(gif_path, frames, duration=0.08)
    print(f"Saved {gif_path}")

print(f"\nSaved highlight GIFs to {gif_dir}/")
