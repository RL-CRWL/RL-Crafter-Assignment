"""
DQN (Deep Q-Network) Baseline Implementation for Crafter Environment

This implementation uses Stable Baselines3 for the base DQN algorithm.
"""

import os
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from datetime import datetime

from utils.wrappers import make_crafter_env


class CrafterMetricsCallback(BaseCallback):
    """
    Custom callback to track Crafter-specific metrics during training
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
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)


class DQNAgent:
    """
    DQN Agent for Crafter Environment - Baseline Implementation
    """
    
    def __init__(self, 
                 env_name="CrafterPartial-v1",
                 learning_rate=1e-4,
                 buffer_size=100000,
                 learning_starts=10000,
                 batch_size=32,
                 gamma=0.99,
                 target_update_interval=10000,
                 exploration_fraction=0.1,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05,
                 train_freq=4,
                 gradient_steps=1,
                 device='auto',
                 seed=42):
        """
        Initialize DQN agent with hyperparameters
        
        Args:
            env_name: Crafter environment name
            learning_rate: Learning rate for optimizer
            buffer_size: Size of replay buffer
            learning_starts: Steps before training starts
            batch_size: Minibatch size for training
            gamma: Discount factor
            target_update_interval: Steps between target network updates
            exploration_fraction: Fraction of training for epsilon decay
            exploration_initial_eps: Initial epsilon for exploration
            exploration_final_eps: Final epsilon for exploration
            train_freq: Update frequency (in steps)
            gradient_steps: Gradient steps per update
            device: Device to use ('auto', 'cuda', 'cpu')
            seed: Random seed
        """
        
        self.env_name = env_name
        self.seed = seed
        
        # Create environments
        self.train_env = make_crafter_env()
        self.eval_env = make_crafter_env()
        
        # Set seeds
        self.train_env.seed(seed)
        self.eval_env.seed(seed + 100)
        
        # DQN hyperparameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'learning_starts': learning_starts,
            'batch_size': batch_size,
            'gamma': gamma,
            'target_update_interval': target_update_interval,
            'exploration_fraction': exploration_fraction,
            'exploration_initial_eps': exploration_initial_eps,
            'exploration_final_eps': exploration_final_eps,
            'train_freq': train_freq,
            'gradient_steps': gradient_steps,
        }
        
        # Create DQN model
        self.model = DQN(
            'CnnPolicy',  # CNN policy for image observations
            self.train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            verbose=1,
            device=device,
            seed=seed
        )
        
        self.metrics_callback = None
        
    def train(self, total_timesteps=500000, eval_freq=10000, save_dir='results/dqn_baseline'):
        """
        Train the DQN agent
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_dir: Directory to save results
        """
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup callback
        self.metrics_callback = CrafterMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("="*60)
        print("Starting DQN Training")
        print("="*60)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Evaluation frequency: {eval_freq}")
        print(f"Save directory: {save_dir}")
        print("\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        # Save model
        model_path = os.path.join(save_dir, 'dqn_baseline_model.zip')
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
    
    def evaluate(self, n_episodes=10, render=False):
        """
        Evaluate the trained agent
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render (not supported in Crafter)
        
        Returns:
            Dictionary with evaluation metrics
        """
        
        print(f"\nEvaluating agent over {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
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
        """Load a saved model"""
        self.model = DQN.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    """Main training script"""
    
    # Create DQN agent
    agent = DQNAgent(
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        seed=42
    )
    
    # Train
    agent.train(
        total_timesteps=500000,
        eval_freq=10000,
        save_dir='results/dqn_baseline'
    )
    
    # Evaluate
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()