"""
PPO with Random Network Distillation (RND) for Crafter Environment
Implements exploration bonus based on prediction error of random features.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import matplotlib.pyplot as plt
import csv
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_crafter_env


class RNDNetwork(nn.Module):
    """Random Network for generating target features"""
    def __init__(self, input_shape, output_size=512):
        super().__init__()
        # Adaptive CNN for processing image observations (works with small inputs)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            n_flatten = self.cnn(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, output_size),
            nn.ReLU()
        )
        
        # Fix random network weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class PredictorNetwork(nn.Module):
    """Predictor Network for learning to predict random features"""
    def __init__(self, input_shape, output_size=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            n_flatten = self.cnn(sample).shape[1]
        
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x


class RNDModule:
    """RND Module for computing intrinsic rewards"""
    def __init__(self, obs_shape, device='cuda', learning_rate=1e-4):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.obs_shape = obs_shape
        
        # Initialize networks
        self.target_network = RNDNetwork(obs_shape).to(self.device)
        self.predictor_network = PredictorNetwork(obs_shape).to(self.device)
        
        # Optimizer for predictor
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=learning_rate)
        
        # Running statistics for normalization
        self.reward_rms_mean = 0.0
        self.reward_rms_var = 1.0
        self.reward_rms_count = 1e-4
        
    def compute_intrinsic_reward(self, obs):
        """Compute intrinsic reward based on prediction error"""
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs).to(self.device)
            else:
                obs_tensor = obs
            
            # Ensure correct shape: (batch, channels, height, width)
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            target_features = self.target_network(obs_tensor)
            predicted_features = self.predictor_network(obs_tensor)
            
            # Prediction error as intrinsic reward
            intrinsic_reward = torch.mean((target_features - predicted_features) ** 2, dim=1)
            
            return intrinsic_reward.cpu().numpy()
    
    def update(self, obs_batch):
        """Update predictor network"""
        if isinstance(obs_batch, np.ndarray):
            obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        
        # Forward pass
        target_features = self.target_network(obs_batch)
        predicted_features = self.predictor_network(obs_batch)
        
        # Compute loss
        loss = torch.mean((target_features - predicted_features) ** 2)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def normalize_reward(self, reward):
        """Normalize intrinsic reward using running statistics"""
        # Update running statistics
        batch_mean = np.mean(reward)
        batch_var = np.var(reward)
        batch_count = len(reward)
        
        delta = batch_mean - self.reward_rms_mean
        total_count = self.reward_rms_count + batch_count
        
        new_mean = self.reward_rms_mean + delta * batch_count / total_count
        m_a = self.reward_rms_var * self.reward_rms_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.reward_rms_count * batch_count / total_count
        new_var = M2 / total_count
        
        self.reward_rms_mean = new_mean
        self.reward_rms_var = new_var
        self.reward_rms_count = total_count
        
        # Normalize
        normalized_reward = reward / (np.sqrt(self.reward_rms_var) + 1e-8)
        return normalized_reward


class RNDRewardWrapper(gym.Wrapper):
    """Wrapper that adds RND intrinsic rewards to environment rewards"""
    def __init__(self, env, rnd_module, intrinsic_coef=1.0):
        super().__init__(env)
        self.rnd_module = rnd_module
        self.intrinsic_coef = intrinsic_coef
        self.last_obs = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute intrinsic reward
        intrinsic_reward = self.rnd_module.compute_intrinsic_reward(obs)
        normalized_intrinsic = self.rnd_module.normalize_reward(intrinsic_reward)
        
        # Update RND predictor
        self.rnd_module.update(np.expand_dims(obs, 0))
        
        # Combine rewards
        total_reward = reward + self.intrinsic_coef * normalized_intrinsic[0]
        
        # Store intrinsic reward in info for logging
        info['intrinsic_reward'] = normalized_intrinsic[0]
        info['extrinsic_reward'] = reward
        
        self.last_obs = obs
        return obs, total_reward, terminated, truncated, info


class RNDCallback(BaseCallback):
    """Callback for RND training and evaluation"""
    def __init__(self, eval_env, eval_freq=10000, csv_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.csv_path = csv_path
        
        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_steps = []
        self.intrinsic_rewards = []
        self.extrinsic_rewards = []
        
        # Initialize CSV file
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'step', 'mean_reward', 'std_reward', 
                               'min_reward', 'max_reward', 'mean_length', 'std_length'])
    
    def _on_step(self) -> bool:
        # Collect intrinsic/extrinsic rewards from episode info
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_metrics = self.evaluate_agent()
            self.eval_rewards.append(eval_metrics['mean_reward'])
            self.eval_steps.append(self.n_calls)
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {self.n_calls}:")
                print(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"  Mean Length: {eval_metrics['mean_length']:.2f}")
                print(f"{'='*60}")
            
            # Save to CSV
            if self.csv_path:
                with open(self.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(),
                        self.n_calls,
                        eval_metrics['mean_reward'],
                        eval_metrics['std_reward'],
                        eval_metrics['min_reward'],
                        eval_metrics['max_reward'],
                        eval_metrics['mean_length'],
                        eval_metrics['std_length']
                    ])
        
        return True
    
    def evaluate_agent(self, n_episodes=5):
        """Run evaluation episodes"""
        rewards = []
        lengths = []
        
        for _ in range(n_episodes):
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
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths)
        }


class PPORNDAgent:
    """PPO Agent with Random Network Distillation"""
    def __init__(self,
                 learning_rate=3e-4,
                 n_steps=2048,
                 batch_size=64,
                 n_epochs=10,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 ent_coef=0.01,
                 intrinsic_reward_coef=1.0,
                 rnd_learning_rate=1e-4,
                 device='auto',
                 seed=42):
        self.seed = seed
        
        # Create environments
        self.train_env = make_crafter_env()
        self.eval_env = make_crafter_env()
        
        # Set seeds
        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 100)
        
        # Get observation shape
        obs, _ = self.train_env.reset()
        obs_shape = obs.shape
        
        # Initialize RND module
        device_str = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        self.rnd_module = RNDModule(
            obs_shape=obs_shape,
            device=device_str,
            learning_rate=rnd_learning_rate
        )
        
        # Wrap training environment with RND rewards
        self.train_env = RNDRewardWrapper(
            self.train_env,
            self.rnd_module,
            intrinsic_coef=intrinsic_reward_coef
        )
        
        # PPO hyperparameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'intrinsic_reward_coef': intrinsic_reward_coef,
            'rnd_learning_rate': rnd_learning_rate
        }
        
        # Create PPO model
        self.model = PPO(
            'CnnPolicy',
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
        
        self.callback = None
    
    def train(self, total_timesteps=500000, eval_freq=10000, 
              save_dir='results/ppo_rnd'):
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # CSV path for evaluation results
        csv_path = os.path.join(save_dir, 'evaluation_results.csv')
        
        # Setup callback
        self.callback = RNDCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            csv_path=csv_path,
            verbose=1
        )
        
        print("="*60)
        print("Starting PPO + RND Training")
        print("="*60)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Evaluation frequency: {eval_freq}")
        print(f"Save directory: {save_dir}")
        print(f"Evaluation CSV: {csv_path}")
        print("\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            log_interval=100
        )
        
        # Save model
        model_path = os.path.join(save_dir, 'ppo_rnd_model.zip')
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save RND networks
        torch.save(self.rnd_module.predictor_network.state_dict(),
                  os.path.join(save_dir, 'rnd_predictor.pt'))
        
        # Save training plots
        self.save_training_plots(save_dir)
        
        return self.model
    
    def save_training_plots(self, save_dir):
        """Save training metric plots"""
        if self.callback is None:
            return
        
        # Plot episode rewards
        if len(self.callback.episode_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback.episode_rewards, alpha=0.3, label='Episode Rewards')
            window = 100
            if len(self.callback.episode_rewards) >= window:
                moving_avg = np.convolve(
                    self.callback.episode_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                plt.plot(moving_avg, linewidth=2, label=f'{window}-episode MA')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time (PPO + RND)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'training_rewards.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot evaluation rewards
        if len(self.callback.eval_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback.eval_steps, self.callback.eval_rewards, 
                    marker='o', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Mean Evaluation Reward')
            plt.title('Evaluation Performance (PPO + RND)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'eval_rewards.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot intrinsic rewards
        if len(self.callback.intrinsic_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback.intrinsic_rewards, alpha=0.3)
            window = 1000
            if len(self.callback.intrinsic_rewards) >= window:
                moving_avg = np.convolve(
                    self.callback.intrinsic_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                plt.plot(moving_avg, linewidth=2, label=f'{window}-step MA')
            plt.xlabel('Step')
            plt.ylabel('Intrinsic Reward')
            plt.title('RND Intrinsic Rewards')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'intrinsic_rewards.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot extrinsic rewards
        if len(self.callback.extrinsic_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.callback.extrinsic_rewards, alpha=0.3)
            window = 1000
            if len(self.callback.extrinsic_rewards) >= window:
                moving_avg = np.convolve(
                    self.callback.extrinsic_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                plt.plot(moving_avg, linewidth=2, label=f'{window}-step MA')
            plt.xlabel('Step')
            plt.ylabel('Extrinsic Reward')
            plt.title('Environment Rewards (Extrinsic)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'extrinsic_rewards.png'), 
                       dpi=150, bbox_inches='tight')
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
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, Length={episode_length}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths)
        }
        
        print("\n" + "="*60)
        print("Final Evaluation Results:")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print("="*60)
        
        return metrics
    
    def load(self, model_path, rnd_path=None):
        """Load saved models"""
        self.model = PPO.load(model_path, env=self.train_env)
        print(f"PPO model loaded from {model_path}")
        
        if rnd_path and os.path.exists(rnd_path):
            self.rnd_module.predictor_network.load_state_dict(
                torch.load(rnd_path)
            )
            print(f"RND predictor loaded from {rnd_path}")


def main():
    """Main training script"""
    # Create PPO + RND agent
    agent = PPORNDAgent(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        intrinsic_reward_coef=1.0,  # Weight for intrinsic rewards
        rnd_learning_rate=1e-4,
        seed=42
    )
    
    # Train
    agent.train(
        total_timesteps=100_000,
        eval_freq=10000,
        save_dir='results/ppo_rnd'
    )
    
    # Final evaluation
    agent.evaluate(n_episodes=20)


if __name__ == "__main__":
    main()