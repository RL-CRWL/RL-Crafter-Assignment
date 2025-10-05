"""
Baseline DQN implementation for Crafter (COMS4061A / COMS7071A)
Author: Wendy Maboa
Description:
    - Uses Stable Baselines3 DQN with CnnPolicy
    - Trains on CrafterPartial-v1 (64x64 RGB input)
    - Logs training progress and saves models
"""

import gymnasium as gym
import crafter
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os

# setup
LOG_DIR = "./results/dqn_baseline_logs/"
MODEL_DIR = "./results/models/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#  Creating an environment
def make_env():
    env = gym.make("CrafterPartial-v1", render_mode=None)
    env = Monitor(env)  # Tracks rewards, episode lengths, etc.
    return env

env = make_vec_env(make_env, n_envs=4)  # parallel envs = faster training

# DQN model definition
model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=10000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log=LOG_DIR,
    device="auto"
)

# Training
TIMESTEPS = 1_000_000  # You can reduce to 100_000 for testing
print(f"🚀 Training DQN for {TIMESTEPS} timesteps...")
model.learn(total_timesteps=TIMESTEPS, log_interval=100, progress_bar=True)

# Save the model
model.save(os.path.join(MODEL_DIR, "dqn_crafter_baseline"))
print("✅ Model saved successfully!")

# Evaluation
print("🎯 Evaluating model...")
eval_env = make_env()
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
print(f"✅ Mean reward over 10 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

# Cleanup
eval_env.close()
env.close()
print("🏁 Training and evaluation completed successfully.")
