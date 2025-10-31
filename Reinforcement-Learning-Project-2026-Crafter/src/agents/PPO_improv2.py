"""
Recurrent PPO (LSTM) with Achievement Tracking for Crafter Environment
Now uses sb3_contrib.RecurrentPPO for true temporal learning.
Saves CSVs for metrics and generates plots from them.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
import csv

# Local import for Crafter env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrapper_ppo import make_crafter_env


# ===========================
#   Reward & Observation Wrappers
# ===========================

class RewardShapingWrapper(gym.Wrapper):
    """Adds reward shaping to encourage exploration and achievement completion."""
    def __init__(self, env):
        super().__init__(env)
        self.previous_achievements = np.zeros(env.num_achievements, dtype=np.float32)
        self.step_count = 0

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.previous_achievements = obs['achievements'].copy()
        self.step_count = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_achievements = obs['achievements']
        new_achievements = current_achievements - self.previous_achievements

        # Bonuses
        achievement_bonus = np.sum(new_achievements) * 5.0
        survival_bonus = 0.01
        exploration_bonus = 0.001

        shaped_reward = reward + achievement_bonus + survival_bonus + exploration_bonus
        self.previous_achievements = current_achievements.copy()
        self.step_count += 1
        return obs, shaped_reward, terminated, truncated, info


class AchievementTrackingWrapper(gym.Wrapper):
    """Tracks Crafter achievements as part of the observation."""
    def __init__(self, env):
        super().__init__(env)
        self.achievement_names = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
            'place_stone', 'place_table', 'wake_up'
        ]
        self.num_achievements = len(self.achievement_names)
        original_obs_space = env.observation_space

        self.observation_space = spaces.Dict({
            'image': original_obs_space,
            'achievements': spaces.Box(low=0, high=1, shape=(self.num_achievements,), dtype=np.float32)
        })

        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'achievements' in info:
            for i, name in enumerate(self.achievement_names):
                if name in info['achievements'] and info['achievements'][name]:
                    self.achievements[i] = 1.0
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, image_obs):
        return {'image': image_obs, 'achievements': self.achievements.copy()}


# ===========================
#   Metrics Callback with CSV
# ===========================

class CrafterMetricsCallback(BaseCallback):
    """Tracks rewards and achievements; saves CSVs periodically."""
    def __init__(self, eval_env, eval_freq=10000, save_dir='results', verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.achievement_counts = []

    def _on_step(self) -> bool:
        # Collect training info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])

        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_reward, achievement_count = self.evaluate_agent()
            self.eval_rewards.append(eval_reward)
            self.achievement_counts.append(achievement_count)

            if self.verbose > 0:
                print(f"\nEvaluation at step {self.n_calls}:")
                print(f"  Mean reward = {eval_reward:.2f}")
                print(f"  Mean achievements = {achievement_count:.2f}")

            # Save CSVs
            self._save_csvs()

        return True

    def evaluate_agent(self, n_episodes=5):
        rewards, achievement_counts = [], []
        for _ in range(n_episodes):
            obs, info = self.eval_env.reset()
            state, episode_start, done = None, True, False
            total_reward = 0
            while not done:
                action, state = self.model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=True
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_start = done
                total_reward += reward
            rewards.append(total_reward)
            achievement_counts.append(np.sum(obs['achievements']))
        return np.mean(rewards), np.mean(achievement_counts)

    def _save_csvs(self):
        # Training data
        if len(self.episode_rewards) > 0:
            df_train = pd.DataFrame({
                'reward': self.episode_rewards,
                'length': self.episode_lengths
            })
            df_train.to_csv(os.path.join(self.save_dir, 'training_data.csv'), index=False)

        # Evaluation data
        if len(self.eval_rewards) > 0:
            df_eval = pd.DataFrame({
                'eval_reward': self.eval_rewards,
                'eval_achievements': self.achievement_counts
            })
            df_eval.to_csv(os.path.join(self.save_dir, 'evaluation_data.csv'), index=False)


# ===========================
#   PPO-LSTM Agent
# ===========================

class PPOLSTMAgent:
    """Recurrent PPO agent with LSTM policy for Crafter."""
    def __init__(self,
                 learning_rate=3e-4,
                 n_steps=4096,
                 batch_size=128,
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 ent_coef=0.02,
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 use_reward_shaping=True,
                 device='auto',
                 seed=42,
                 save_dir='results/ppo_run'):
        self.seed = seed
        self.use_reward_shaping = use_reward_shaping
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        base_train_env = make_crafter_env()
        base_eval_env = make_crafter_env()
        self.train_env = AchievementTrackingWrapper(base_train_env)
        self.eval_env = AchievementTrackingWrapper(base_eval_env)

        if use_reward_shaping:
            self.train_env = RewardShapingWrapper(self.train_env)

        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 100)

        self.hyperparameters = dict(
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )

        self.model = RecurrentPPO(
            "MultiInputLstmPolicy",
            self.train_env,
            verbose=1,
            seed=seed,
            device=device,
            **self.hyperparameters
        )

        self.metrics_callback = CrafterMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=10000,
            save_dir=self.save_dir,
            verbose=1
        )

    def train(self, total_timesteps=1_000_000, eval_freq=10_000):
        print("=" * 60)
        print("Starting Recurrent PPO (LSTM) Training with Achievement Tracking")
        print("=" * 60)
        for k, v in self.hyperparameters.items():
            print(f"{k}: {v}")
        print("=" * 60)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )

        model_path = os.path.join(self.save_dir, 'recurrent_ppo_model.zip')
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")

        # Generate plots from CSVs
        self.save_training_plots()

    def evaluate(self, n_episodes=10):
        """Evaluate trained RecurrentPPO agent."""
        print(f"\nEvaluating RecurrentPPO agent over {n_episodes} episodes...")
        rewards, lengths, achievements = [], [], []

        for ep in range(n_episodes):
            obs, info = self.eval_env.reset()
            state, episode_start, done = None, True, False
            total_r, length = 0, 0

            while not done:
                action, state = self.model.predict(
                    obs, state=state, episode_start=episode_start, deterministic=True
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_start = done
                total_r += reward
                length += 1

            rewards.append(total_r)
            lengths.append(length)
            achievements.append(np.sum(obs['achievements']))

            print(f"Episode {ep+1}: Reward={total_r:.2f}, Steps={length}, Achievements={int(np.sum(obs['achievements']))}")

        print("\n" + "=" * 60)
        print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
        print(f"Mean Achievements: {np.mean(achievements):.2f} ± {np.std(achievements):.2f}")
        print("=" * 60)

    def save_training_plots(self):
        """Read CSVs and generate plots for training and evaluation metrics."""
        training_csv = os.path.join(self.save_dir, 'training_data.csv')
        eval_csv = os.path.join(self.save_dir, 'evaluation_data.csv')

        # Training plots
        if os.path.exists(training_csv):
            df_train = pd.read_csv(training_csv)
            plt.figure(figsize=(10, 6))
            plt.plot(df_train['reward'], alpha=0.3, label='Episode Reward')
            if len(df_train) > 100:
                ma = np.convolve(df_train['reward'], np.ones(100)/100, mode='valid')
                plt.plot(ma, color='red', label='100-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards (Recurrent PPO)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'training_rewards.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # Evaluation plots
        if os.path.exists(eval_csv):
            df_eval = pd.read_csv(eval_csv)
            plt.figure(figsize=(10, 6))
            plt.plot(df_eval['eval_reward'], label='Eval Reward')
            plt.plot(df_eval['eval_achievements'], label='Eval Achievements')
            plt.xlabel('Evaluation Step')
            plt.ylabel('Value')
            plt.title('Evaluation Metrics (Recurrent PPO)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.save_dir, 'evaluation_metrics.png'), dpi=150, bbox_inches='tight')
            plt.close()


# ===========================
#   MAIN ENTRY POINT
# ===========================

def main():
    agent = PPOLSTMAgent(
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_reward_shaping=True,
        seed=42,
        save_dir='Reinforcement-Learning-Project-2026-Crafter/results/PPO/models/ppo_improv_2'
    )

    agent.train(
        total_timesteps=1_000_000,
        eval_freq=10_000
    )

    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()

