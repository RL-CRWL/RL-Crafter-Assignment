import os, sys
from sb3_contrib import RecurrentPPO
import numpy as np
import imageio

# ===== Imports =====
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.PPO_improv2 import AchievementTrackingWrapper
from utils.wrapper_ppo import make_crafter_env

# ===== Load environment =====
base_env = make_crafter_env()
env = AchievementTrackingWrapper(base_env)

# ===== Load trained RecurrentPPO model =====
model_path = "../../results/ppo_improv_2/recurrent_ppo_model"
model = RecurrentPPO.load(model_path, env=env)

# ===== Parameters =====
n_eval_episodes = 30
gif_dir = "recurrent_ppo_highlights"
os.makedirs(gif_dir, exist_ok=True)

episode_data = []  # store (reward, achievements, frames)
print(f"Evaluating {n_eval_episodes} episodes...\n")

# ===== Evaluate episodes =====
for ep in range(n_eval_episodes):
    obs, info = env.reset()
    state, episode_start, done = None, True, False
    frames, total_reward = [], 0

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = done
        total_reward += reward

    # ===== Track achievements =====
    achieved_indices = np.where(obs["achievements"] > 0)[0]
    achieved_names = [env.achievement_names[i] for i in achieved_indices]
    num_achievements = len(achieved_names)

    episode_data.append((total_reward, num_achievements, achieved_names, frames))

    print(f"Episode {ep+1}: Reward={total_reward:.1f}, Achievements={num_achievements}")
    if achieved_names:
        print("   → " + ", ".join(achieved_names))
    else:
        print("   → No achievements unlocked.")

env.close()

# ===== Compute stats =====
rewards = np.array([r for (r, _, _, _) in episode_data])
ach_counts = np.array([a for (_, a, _, _) in episode_data])

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
mean_ach = np.mean(ach_counts)
std_ach = np.std(ach_counts)

best_reward_idx = int(np.argmax(rewards))
best_ach_idx = int(np.argmax(ach_counts))
median_idx = int(np.argsort(rewards)[len(rewards)//2])
selected_indices = sorted(set([best_reward_idx, best_ach_idx, median_idx]))

# ===== Save highlight GIFs =====
print("\nSelected highlight episodes:", [i + 1 for i in selected_indices])
for idx in selected_indices:
    reward, ach, achieved_names, frames = episode_data[idx]
    gif_path = os.path.join(gif_dir, f"highlight_ep{idx+1}_R{reward:.1f}_A{ach}.gif")
    imageio.mimsave(gif_path, frames, duration=0.08)
    print(f"Saved {gif_path}")

# ===== Summary Table =====
print("\n" + "="*60)
print(" EVALUATION SUMMARY ".center(60, "="))
print("="*60)
print(f"Total Episodes Evaluated : {n_eval_episodes}")
print(f"Mean Reward             : {mean_reward:.2f} ± {std_reward:.2f}")
print(f"Mean Achievements       : {mean_ach:.2f} ± {std_ach:.2f}")
print("-"*60)
print(f"Best Reward Episode     : {best_reward_idx+1} (Reward={rewards[best_reward_idx]:.2f})")
print(f"Most Achievements Ep.   : {best_ach_idx+1} (Achievements={ach_counts[best_ach_idx]})")
print(f"Median Reward Episode   : {median_idx+1} (Reward={rewards[median_idx]:.2f})")
print("="*60)
print(f"\nSaved highlight GIFs to: {gif_dir}/")


