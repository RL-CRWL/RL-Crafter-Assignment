"""
PPO with LSTM and Achievement Tracking for Crafter Environment
Extends baseline PPO with memory and explicit achievement state tracking.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Type
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_crafter_env


class RewardShapingWrapper(gym.Wrapper):
    """
    Adds reward shaping to encourage better exploration and achievement completion.
    """
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
        
        # Reward shaping based on new achievements
        current_achievements = obs['achievements']
        new_achievements = current_achievements - self.previous_achievements
        
        # Big bonus for unlocking new achievements
        achievement_bonus = np.sum(new_achievements) * 5.0
        
        # Small survival bonus to encourage staying alive
        survival_bonus = 0.01
        
        # Penalize staying in one place (encourage exploration)
        # This is a simple heuristic - you could track position if available
        exploration_bonus = 0.001
        
        # Combine rewards
        shaped_reward = reward + achievement_bonus + survival_bonus + exploration_bonus
        
        self.previous_achievements = current_achievements.copy()
        self.step_count += 1
        
        return obs, shaped_reward, terminated, truncated, info


class AchievementTrackingWrapper(gym.Wrapper):
    """
    Wrapper that tracks and exposes Crafter achievements as part of the observation.
    """
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
        
        # Create new observation space that includes achievements
        original_obs_space = env.observation_space
        self.observation_space = spaces.Dict({
            'image': original_obs_space,
            'achievements': spaces.Box(
                low=0, high=1, shape=(self.num_achievements,), dtype=np.float32
            )
        })
        
        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)
        return self._get_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update achievements from info
        if 'achievements' in info:
            for i, name in enumerate(self.achievement_names):
                if name in info['achievements'] and info['achievements'][name]:
                    self.achievements[i] = 1.0
        
        return self._get_obs(obs), reward, terminated, truncated, info
    
    def _get_obs(self, image_obs):
        return {
            'image': image_obs,
            'achievements': self.achievements.copy()
        }


class CrafterLSTMExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that combines CNN for images and MLP for achievements,
    then processes through an LSTM for temporal awareness.
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        lstm_hidden_size: int = 256,
    ):
        super().__init__(observation_space, features_dim)
        
        # CNN for processing image observations
        image_shape = observation_space['image'].shape
        n_input_channels = image_shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_image = torch.zeros(1, *image_shape)
            cnn_output_size = self.cnn(sample_image).shape[1]
        
        # MLP for processing achievement vector
        achievement_dim = observation_space['achievements'].shape[0]
        self.achievement_mlp = nn.Sequential(
            nn.Linear(achievement_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Combine CNN and achievement features
        combined_size = cnn_output_size + 64
        
        # LSTM for temporal processing
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = nn.LSTM(combined_size, lstm_hidden_size, batch_first=True)
        
        # Final linear layer to get to features_dim
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        
        # Hidden state buffers (will be managed during rollout)
        self.lstm_hidden = None
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process image through CNN
        image_features = self.cnn(observations['image'])
        
        # Process achievements through MLP
        achievement_features = self.achievement_mlp(observations['achievements'])
        
        # Combine features
        combined = torch.cat([image_features, achievement_features], dim=1)
        
        # Process through LSTM
        batch_size = combined.shape[0]
        
        if self.lstm_hidden is None or self.lstm_hidden[0].shape[1] != batch_size:
            self.lstm_hidden = (
                torch.zeros(1, batch_size, self.lstm_hidden_size, device=combined.device),
                torch.zeros(1, batch_size, self.lstm_hidden_size, device=combined.device)
            )
        
        # Detach hidden state to prevent backprop through time across batches
        self.lstm_hidden = (self.lstm_hidden[0].detach(), self.lstm_hidden[1].detach())
        
        # Reshape for LSTM: (batch, seq_len=1, features)
        lstm_input = combined.unsqueeze(1)
        lstm_out, self.lstm_hidden = self.lstm(lstm_input, self.lstm_hidden)
        
        # Remove sequence dimension and pass through final linear layer
        lstm_out = lstm_out.squeeze(1)
        output = self.linear(lstm_out)
        
        return output
    
    def reset_hidden_state(self):
        """Reset LSTM hidden state (call at episode boundaries)"""
        self.lstm_hidden = None


class LSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCritic policy that uses our LSTM feature extractor.
    """
    def __init__(self, *args, **kwargs):
        # Set our custom feature extractor
        kwargs['features_extractor_class'] = CrafterLSTMExtractor
        kwargs['features_extractor_kwargs'] = {
            'features_dim': 256,
            'lstm_hidden_size': 256
        }
        super().__init__(*args, **kwargs)


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
        self.achievement_counts = []

    def _on_step(self) -> bool:
        # Log training metrics
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
        
        return True

    def evaluate_agent(self, n_episodes=5):
        """Run evaluation episodes"""
        rewards = []
        achievement_counts = []
        
        for _ in range(n_episodes):
            obs, info = self.eval_env.reset()
            
            # Reset LSTM hidden state for new episode
            if hasattr(self.model.policy.features_extractor, 'reset_hidden_state'):
                self.model.policy.features_extractor.reset_hidden_state()
            
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            rewards.append(episode_reward)
            # Count achievements from final observation
            achievement_counts.append(np.sum(obs['achievements']))
        
        return np.mean(rewards), np.mean(achievement_counts)


class PPOLSTMAgent:
    """
    PPO Agent with LSTM memory and Achievement Tracking for Crafter Environment
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
                 vf_coef=0.5,
                 max_grad_norm=0.5,
                 use_reward_shaping=True,
                 device='auto',
                 seed=42):
        self.seed = seed
        self.use_reward_shaping = use_reward_shaping
        
        # Create environments with achievement tracking
        base_train_env = make_crafter_env()
        base_eval_env = make_crafter_env()
        
        self.train_env = AchievementTrackingWrapper(base_train_env)
        self.eval_env = AchievementTrackingWrapper(base_eval_env)
        
        # Add reward shaping wrapper if enabled
        if use_reward_shaping:
            self.train_env = RewardShapingWrapper(self.train_env)
        
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
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'max_grad_norm': max_grad_norm
        }

        # Create PPO model with LSTM policy
        self.model = PPO(
            LSTMActorCriticPolicy,
            self.train_env,
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
            verbose=1,
            device=device,
            seed=seed
        )

        self.metrics_callback = None

    def train(self, total_timesteps=500000, eval_freq=10000, save_dir='results/ppo_improv_2'):
        # Create save directory (relative to project root)
        # Go up two levels from src/agents/ to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(script_dir)  # src/
        project_root = os.path.dirname(src_dir)  # project root
        full_save_dir = os.path.join(project_root, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        
        # Setup callback
        self.metrics_callback = CrafterMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("="*60)
        print("Starting PPO Training with LSTM + Achievement Tracking")
        print("="*60)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Evaluation frequency: {eval_freq}")
        print(f"Save directory: {full_save_dir}")
        print("\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print("\nArchitecture:")
        print("  - CNN for visual processing")
        print("  - MLP for achievement tracking")
        print("  - LSTM for temporal memory")
        print("  - 22 tracked achievements")
        if self.use_reward_shaping:
            print("  - Reward shaping enabled (achievement bonuses)")
        print("="*60)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        # Save model
        model_path = os.path.join(full_save_dir, 'ppo_lstm_model.zip')
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save training metrics
        self.save_training_plots(full_save_dir)
        
        return self.model

    def save_training_plots(self, save_dir):
        """Save training metric plots"""
        if self.metrics_callback is None:
            print("Warning: No metrics callback found, skipping plots")
            return
        
        print(f"\nGenerating training plots...")
        print(f"Episode rewards collected: {len(self.metrics_callback.episode_rewards)}")
        print(f"Eval rewards collected: {len(self.metrics_callback.eval_rewards)}")
        
        # Plot episode rewards
        if len(self.metrics_callback.episode_rewards) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics_callback.episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                moving_avg = np.convolve(
                    self.metrics_callback.episode_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                plt.plot(moving_avg, linewidth=2, label=f'{window}-episode MA', color='red')
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('Training Rewards Over Time (LSTM + Achievements)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plot_path = os.path.join(save_dir, 'training_rewards.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved training rewards plot to {plot_path}")
        else:
            print("Warning: No episode rewards to plot")
        
        # Plot evaluation rewards and achievements
        if len(self.metrics_callback.eval_rewards) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            eval_steps = np.arange(len(self.metrics_callback.eval_rewards)) * self.metrics_callback.eval_freq
            
            # Rewards
            ax1.plot(eval_steps, self.metrics_callback.eval_rewards, marker='o', color='blue')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('Mean Evaluation Reward')
            ax1.set_title('Evaluation Performance')
            ax1.grid(True, alpha=0.3)
            
            # Achievements
            ax2.plot(eval_steps, self.metrics_callback.achievement_counts, marker='s', color='green')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Mean Achievements Unlocked')
            ax2.set_title('Achievement Progress')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(save_dir, 'eval_metrics.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved evaluation metrics plot to {plot_path}")
        else:
            print("Warning: No evaluation rewards to plot")

    def evaluate(self, n_episodes=10):
        """Evaluate the trained agent"""
        print(f"\nEvaluating agent over {n_episodes} episodes...")
        episode_rewards = []
        episode_lengths = []
        episode_achievements = []
        
        for episode in range(n_episodes):
            obs, info = self.eval_env.reset()
            
            # Reset LSTM hidden state
            if hasattr(self.model.policy.features_extractor, 'reset_hidden_state'):
                self.model.policy.features_extractor.reset_hidden_state()
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
            
            achievement_count = np.sum(obs['achievements'])
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_achievements.append(achievement_count)
            
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Achievements={int(achievement_count)}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'mean_achievements': np.mean(episode_achievements),
            'std_achievements': np.std(episode_achievements)
        }
        
        print("\n" + "="*60)
        print("Evaluation Results (LSTM + Achievement Tracking):")
        print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print(f"Mean Achievements: {metrics['mean_achievements']:.2f} ± {metrics['std_achievements']:.2f}")
        print("="*60)
        
        return metrics

    def load(self, model_path):
        """Load a saved PPO model"""
        self.model = PPO.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    """Main training script"""
    # Create PPO agent with LSTM and achievement tracking
    # IMPROVED HYPERPARAMETERS FOR BETTER EXPLORATION AND LEARNING
    agent = PPOLSTMAgent(
        learning_rate=3e-4,
        n_steps=4096,           # Increased from 2048 - more experience per update
        batch_size=128,         # Increased from 64 - more stable gradients
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,          # Increased from 0.01 - MORE EXPLORATION
        vf_coef=0.5,            # Added - balances value/policy learning
        max_grad_norm=0.5,      # Added - prevents exploding gradients
        use_reward_shaping=True, # Enable achievement bonuses
        seed=42
    )
    
    # Train for longer to see better results
    agent.train(
        total_timesteps=1000000,  # Increased from 500k
        eval_freq=10000,
        save_dir='results/ppo_improv_2'
    )
    
    # Evaluate
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()