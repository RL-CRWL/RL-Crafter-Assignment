"""
PPO (Proximal Policy Optimization) Baseline Implementation for Crafter Environment
Uses Stable Baselines3's PPO algorithm.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_crafter_env

class CrafterMetricsCallback(BaseCallback):
    """
    Custom callback to track Crafter-specific metrics during PPO training.
    """
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []

    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_reward = self.evaluate_agent()
            self.eval_rewards.append(eval_reward)
            if self.verbose > 0:
                print(f"\nEvaluation at step {self.n_calls}: Mean reward = {eval_reward:.2f}")
        return True

    def evaluate_agent(self, n_episodes=5):
        """Run evaluation episodes"""
        rewards = []
        for _ in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)

class PPOAgent:
    """
    PPO Agent for Crafter Environment - Baseline Implementation
    """
    def __init__(self,
                 learning_rate=3e-4,
                 n_steps=2048,
                 batch_size=64,
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 ent_coef=0.01,
                 device='auto',
                 seed=42):
        self.seed = seed
        # Create environments
        self.train_env = make_crafter_env()
        self.eval_env = make_crafter_env()
        # Set seeds
        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 100)

        # PPO hyperparameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef
        }

        # Create PPO model
        self.model = PPO(
            'CnnPolicy', # CNN policy for image observations
            self.train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            device=device,
            seed=seed
        )

        self.metrics_callback = None

    def train(self, total_timesteps=500000, eval_freq=10000, save_dir='results/ppo_baseline'):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        # Setup callback
        self.metrics_callback = CrafterMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        print("="*60)
        print("Starting PPO Training")
        print("="*60)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Evaluation frequency: {eval_freq}")
        print(f"Save directory: {save_dir}")
        print("\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f" {key}: {value}")
        print("="*60)
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        # Save model
        model_path = os.path.join(save_dir, 'ppo_baseline_model.zip')
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        # Save training metrics
        self.save_training_plots(save_dir)
        return self.model

    def save_training_plots(self, save_dir):
        """Save training metric plots"""
        if self.metrics_callback is None:
            return
        # Plot episode rewards
        if len(self.metrics_callback.episode_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_callback.episode_rewards, alpha=0.3)
            # Moving average
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                moving_avg = np.convolve(
                    self.metrics_callback.episode_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                plt.plot(moving_avg, linewidth=2, label=f'{window}-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'training_rewards.png'), dpi=150, bbox_inches='tight')
            plt.close()
        # Plot evaluation rewards
        if len(self.metrics_callback.eval_rewards) > 0:
            plt.figure(figsize=(10, 6))
            eval_steps = np.arange(len(self.metrics_callback.eval_rewards)) * self.metrics_callback.eval_freq
            plt.plot(eval_steps, self.metrics_callback.eval_rewards, marker='o')
            plt.xlabel('Training Steps')
            plt.ylabel('Mean Evaluation Reward')
            plt.title('Evaluation Performance')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'eval_rewards.png'), dpi=150, bbox_inches='tight')
            plt.close()

    def evaluate(self, n_episodes=10):
        """Evaluate the trained agent"""
        print(f"\nEvaluating agent over {n_episodes} episodes...")
        episode_rewards = []
        episode_lengths = []
        for episode in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"Episode {episode+1}/{n_episodes}: Reward={episode_reward:.2f}, Length={episode_length}")
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        print("\n" + "="*60)
        print("Evaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print("="*60)
        return metrics

    def load(self, model_path):
        """Load a saved PPO model"""
        self.model = PPO.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")

def main():
    """Main training script"""
    # Create PPO agent
    agent = PPOAgent(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42
    )
    # Train
    agent.train(
        total_timesteps=500000,
        eval_freq=10000,
        save_dir='results/ppo_baseline'
    )
    # Evaluate
    agent.evaluate(n_episodes=10)

if __name__ == "__main__":
    main()
