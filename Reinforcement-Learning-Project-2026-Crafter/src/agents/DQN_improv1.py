"""
DQN Improvement 1 SIMPLIFIED: Conservative CNN Enhancement

PROBLEM IDENTIFIED:
- Deep CNN (4 layers) is too hard to train → poor performance
- Need incremental improvement, not radical change

NEW APPROACH:
- Keep baseline's 3-layer CNN structure
- Add ONLY observation normalization (proven to work)
- Slightly increase capacity (more filters)
- Keep everything else stable

Goal: Beat baseline by making minimal, well-tested changes
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env


class SimplifiedImprovedCNN(BaseFeaturesExtractor):
    """
    SIMPLIFIED improved CNN - minimal changes from baseline
    
    Changes from baseline:
    1. Slightly more filters (32→48, 64→96, 64→96)
    2. Better initialization
    3. That's it! Keep it simple.
    """
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        # Same structure as baseline, just slightly wider
        self.cnn = nn.Sequential(
            # 64x64x3 -> 32x32x48 (was 32)
            nn.Conv2d(n_input_channels, 48, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # 32x32x48 -> 9x9x96 (was 64)
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            # 9x9x96 -> 7x7x96 (was 64)
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flatten size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
        # Better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


class SimplifiedMetricsCallback(BaseCallback):
    """Simplified callback - same as before but cleaner"""
    
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.achievements_unlocked = []
        self.best_eval_reward = -np.inf
        
    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
        
        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_agent()
            
            self.eval_rewards.append(metrics['mean_reward'])
            self.eval_lengths.append(metrics['mean_length'])
            self.achievements_unlocked.append(metrics['unique_achievements'])
            
            if metrics['mean_reward'] > self.best_eval_reward:
                self.best_eval_reward = metrics['mean_reward']
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Eval @ step {self.n_calls:,}")
                print(f"  Reward:      {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"  Survival:    {metrics['mean_length']:.1f} steps")
                print(f"  Achievements: {metrics['unique_achievements']}/22")
                print(f"  Best:        {self.best_eval_reward:.2f}")
                print(f"{'='*60}\n")
        
        return True
    
    def evaluate_agent(self, n_episodes=5):
        rewards = []
        lengths = []
        all_achievements = defaultdict(int)
        
        for ep in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked:
                            episode_achievements.add(achievement)
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            for ach in episode_achievements:
                all_achievements[ach] += 1
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'unique_achievements': len(all_achievements)
        }


class DQNImprovement1Simplified:
    """
    DQN Improvement 1 SIMPLIFIED: Conservative Approach
    
    Philosophy: Make minimal, well-tested changes
    - Observation normalization (proven to help)
    - Slightly wider CNN (not deeper!)
    - Everything else identical to baseline
    """
    
    def __init__(self,
                 learning_rate=1e-4,        # Same as baseline
                 buffer_size=100000,        # Same as baseline  
                 learning_starts=10000,      # Same as baseline
                 batch_size=32,             # Same as baseline
                 gamma=0.99,                # Same as baseline
                 target_update_interval=10000,  # Same as baseline
                 exploration_fraction=0.1,   # Same as baseline
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05,
                 train_freq=4,
                 gradient_steps=1,
                 device='auto',
                 seed=42):
        
        self.seed = seed
        
        # ONLY CHANGE: Add observation normalization
        print("\n🔧 Creating environments with normalization...")
        self.train_env = make_crafter_env(
            seed=seed,
            preprocess_type='normalize'
        )
        self.eval_env = make_crafter_env(
            seed=seed + 100,
            preprocess_type='normalize'
        )
        
        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 100)
        
        # Device
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"💻 Using device: {self.device}")
        
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
        
        # Use simplified CNN
        policy_kwargs = dict(
            features_extractor_class=SimplifiedImprovedCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        print("\n🏗️ Building SIMPLIFIED DQN...")
        print("  ✓ Observation normalization")
        print("  ✓ Slightly wider CNN (48-96-96 vs 32-64-64)")
        print("  ✓ Better initialization")
        print("  ✓ All other hyperparameters = baseline")
        
        self.model = DQN(
            'CnnPolicy',
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
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=self.device,
            seed=seed
        )
        
        self.metrics_callback = None
        
    def train(self, total_timesteps=500000, eval_freq=10000, 
              save_dir='results/dqn_improvement1_simplified'):
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_callback = SimplifiedMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("STARTING DQN IMPROVEMENT 1 (SIMPLIFIED) TRAINING")
        print("="*70)
        print("\n🎯 Goal: Beat baseline with minimal, proven changes")
        print("\n📋 Changes from baseline:")
        print("  1. Observation normalization (0-255 → 0-1)")
        print("  2. Slightly wider CNN (50% more filters)")
        print("  3. Better weight initialization")
        print("  4. Everything else IDENTICAL to baseline")
        
        print(f"\n⚙️ Hyperparameters (same as baseline):")
        for key, value in self.hyperparameters.items():
            print(f"  {key:25s}: {value}")
        print("="*70 + "\n")
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        # Save
        model_path = os.path.join(save_dir, 'model.zip')
        self.model.save(model_path)
        print(f"\n💾 Model saved to {model_path}")
        
        self.save_training_plots(save_dir)
        
        return self.model
    
    def save_training_plots(self, save_dir):
        if self.metrics_callback is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Improvement 1 (Simplified) - Training Progress', 
                    fontsize=16, fontweight='bold')
        
        # Rewards
        ax = axes[0, 0]
        if len(self.metrics_callback.episode_rewards) > 0:
            ax.plot(self.metrics_callback.episode_rewards, alpha=0.2, color='blue')
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                ma = np.convolve(self.metrics_callback.episode_rewards,
                               np.ones(window)/window, mode='valid')
                ax.plot(ma, linewidth=2, color='red', label=f'{window}-ep MA')
            ax.axhline(y=3.68, color='green', linestyle='--', linewidth=2, 
                      label='Baseline (3.68)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Eval rewards
        ax = axes[0, 1]
        if len(self.metrics_callback.eval_rewards) > 0:
            steps = np.arange(len(self.metrics_callback.eval_rewards)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.eval_rewards, marker='o', 
                   linewidth=2, markersize=6)
            ax.axhline(y=3.68, color='green', linestyle='--', linewidth=2,
                      label='Baseline (3.68)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Evaluation Reward')
            ax.set_title('Evaluation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Survival
        ax = axes[1, 0]
        if len(self.metrics_callback.eval_lengths) > 0:
            steps = np.arange(len(self.metrics_callback.eval_lengths)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.eval_lengths, marker='s',
                   linewidth=2, color='orange', markersize=6)
            ax.axhline(y=199, color='green', linestyle='--', linewidth=2,
                      label='Baseline (199)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Survival Time')
            ax.set_title('Survival Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Achievements
        ax = axes[1, 1]
        if len(self.metrics_callback.achievements_unlocked) > 0:
            steps = np.arange(len(self.metrics_callback.achievements_unlocked)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.achievements_unlocked, marker='^',
                   linewidth=2, color='green', markersize=6)
            ax.axhline(y=22, color='red', linestyle='--', linewidth=2,
                      label='Total (22)')
            ax.axhline(y=8, color='orange', linestyle='--', linewidth=2,
                      label='Baseline (8)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Unique Achievements')
            ax.set_title('Achievement Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved")
    
    def evaluate(self, n_episodes=10):
        print(f"\n{'='*70}")
        print(f"EVALUATING DQN IMPROVEMENT 1 (SIMPLIFIED)")
        print(f"{'='*70}\n")
        
        episode_rewards = []
        episode_lengths = []
        all_achievements = defaultdict(int)
        
        for episode in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked:
                            episode_achievements.add(achievement)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            for ach in episode_achievements:
                all_achievements[ach] += 1
            
            print(f"Episode {episode+1:2d}/{n_episodes}: "
                  f"Reward={episode_reward:.1f}, Length={episode_length:3d}, "
                  f"Achievements={len(episode_achievements)}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'total_unique_achievements': len(all_achievements)
        }
        
        print("\n" + "="*70)
        print("COMPARISON:")
        print("="*70)
        print(f"                      Reward    Survival")
        print(f"Baseline:             3.68      199.0")
        print(f"Simplified Improv 1:  {metrics['mean_reward']:.2f}      {metrics['mean_length']:.1f}")
        print("="*70 + "\n")
        
        return metrics
    
    def load(self, model_path):
        self.model = DQN.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    agent = DQNImprovement1Simplified(seed=42)
    agent.train(total_timesteps=500000, eval_freq=10000)
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()