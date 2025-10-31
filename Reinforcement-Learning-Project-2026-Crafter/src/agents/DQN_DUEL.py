"""
DQN Improvement 2: Frame Stacking for Temporal Context

PREVIOUS PROBLEM:
- Curriculum learning approaches (extended exploration) consistently underperform
- Both 30% and 20% exploration led to worse results than Improvement 1

NEW APPROACH - FRAME STACKING:
Why this should work:
1. Single frames lack temporal information (can't see movement/velocity)
2. Stacking 4 frames gives agent context about entity motion
3. Helps agent predict: "Is that zombie moving toward me?"
4. Should improve survival and achievement success rates

This is a DIFFERENT type of improvement - architectural rather than training-focused

Based on proven CNN from Improvement 1 + frame stacking
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env

import torch.nn.functional as F
import torch.nn as nn

class DuelingCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Shared convolutional layers
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_space.shape[0], 64, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Get CNN output size
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # Shared fully connected
        self.shared_fc = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Linear(256, 1)
        
        # Advantage stream  
        self.advantage_stream = nn.Linear(256, features_dim)
        
    def forward(self, observations):
        features = self.cnn(observations)
        shared_out = self.shared_fc(features)
        
        value = self.value_stream(shared_out)
        advantage = self.advantage_stream(shared_out)
        
        # Dueling combination
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
    
class FrameStackCNN(BaseFeaturesExtractor):
    """
    CNN designed for frame-stacked observations (4 frames × 3 channels = 12 input channels)
    
    Same architecture as Improvement 1, but handles 12 input channels instead of 3
    """
    
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # Should be 12 (4 frames × 3 RGB)
        
        print(f"  📺 Frame-stacked CNN: {n_input_channels} input channels (4 frames × 3 RGB)")
        
        self.cnn = nn.Sequential(
            # Same structure as Improvement 1, just more input channels
            nn.Conv2d(n_input_channels, 48, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(48, 96, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate flatten size
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


class FrameStackMetricsCallback(BaseCallback):
    """Metrics callback with frame stacking awareness"""
    
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.achievements_unlocked = []
        self.geometric_means = []
        self.best_eval_reward = -np.inf
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_agent(n_episodes=10)
            
            self.eval_rewards.append(metrics['mean_reward'])
            self.eval_lengths.append(metrics['mean_length'])
            self.achievements_unlocked.append(metrics['unique_achievements'])
            self.geometric_means.append(metrics['geometric_mean'])
            
            if metrics['mean_reward'] > self.best_eval_reward:
                self.best_eval_reward = metrics['mean_reward']
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Eval @ step {self.n_calls:,}")
                print(f"  Reward:       {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"  Survival:     {metrics['mean_length']:.1f} steps")
                print(f"  Achievements: {metrics['unique_achievements']}/22")
                print(f"  Geom Mean:    {metrics['geometric_mean']:.1f}%")
                print(f"  Best:         {self.best_eval_reward:.2f}")
                print(f"{'='*60}\n")
        
        return True
    
    def evaluate_agent(self, n_episodes=10):
        rewards = []
        lengths = []
        all_achievements = defaultdict(int)
        achievement_rates = defaultdict(int)
        
        for ep in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Handle vectorized info
                actual_info = info[0] if isinstance(info, list) else info
                
                if 'achievements' in actual_info:
                    for achievement, unlocked in actual_info['achievements'].items():
                        if unlocked:
                            episode_achievements.add(achievement)
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
            
            for ach in episode_achievements:
                all_achievements[ach] += 1
        
        # Calculate geometric mean
        for ach, count in all_achievements.items():
            achievement_rates[ach] = count / n_episodes
        
        if achievement_rates:
            geometric_mean = np.exp(np.mean(np.log([rate + 1e-10 for rate in achievement_rates.values()]))) * 100
        else:
            geometric_mean = 0.0
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'unique_achievements': len(all_achievements),
            'geometric_mean': geometric_mean
        }


class DQNImprovement3DuelCNN:
    """
    DQN Improvement 2: Frame Stacking
    
    Key improvement: Stack 4 frames to provide temporal context
    
    Benefits:
    - Agent can see motion/velocity of entities
    - Better prediction of threats (zombies, skeletons)
    - Improved decision making for combat and exploration
    
    All other hyperparameters SAME as Improvement 1 (the working one!)
    """
    
    def __init__(self,
                 learning_rate=1e-4,
                 buffer_size=75000,         # SAME as Improvement 1
                 learning_starts=10000,     # SAME as Improvement 1
                 batch_size=32,
                 gamma=0.99,
                 target_update_interval=10000,
                 exploration_fraction=0.1,  # SAME as Improvement 1
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.05, # SAME as Improvement 1
                 train_freq=4,
                 gradient_steps=1,
                 n_stack=4,                 # NEW: Stack 4 frames
                 device='auto',
                 seed=42):
        
        self.seed = seed
        self.n_stack = n_stack
        
        print("\n🔧 Creating frame-stacked environment...")
        print(f"  * Frame stacking: {n_stack} frames")
        print(f"  * Observation normalization: ON")
        print(f"  * Input channels: {n_stack * 3} (was 3)")
        
        # Create base environment
        def make_env():
            env = make_crafter_env(
                seed=seed,
                preprocess_type='normalize'
            )
            return env
        
        # Wrap in DummyVecEnv for frame stacking
        self.train_env = DummyVecEnv([make_env])
        self.train_env = VecFrameStack(self.train_env, n_stack=n_stack)
        
        # Eval environment
        def make_eval_env():
            env = make_crafter_env(
                seed=seed + 100,
                preprocess_type='normalize'
            )
            return env
        
        self.eval_env = DummyVecEnv([make_eval_env])
        self.eval_env = VecFrameStack(self.eval_env, n_stack=n_stack)
        
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n💻 Using device: {self.device}")
        
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
            'n_stack': n_stack,
        }
        
        # Use frame-stack aware CNN
        policy_kwargs = dict(
            features_extractor_class=DuelingCNN,
            features_extractor_kwargs=dict(features_dim=256),
        )
        
        print("\n🏗️ Building DQN with frame stacking...")
        print("  * SAME hyperparameters as Improvement 1 (proven to work!)")
        print("  * ONLY change: 4-frame stacking for temporal awareness")
        print("  * Expected: Better combat/exploration decisions")
        
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
              save_dir='results/dqn_improvement2_framestack'):
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_callback = FrameStackMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("STARTING DQN IMPROVEMENT 2 (FRAME STACKING) TRAINING")
        print("="*70)
        print("\n🎯 Goal: Add temporal awareness through frame stacking")
        print("\n📋 Key Change from Improvement 1:")
        print("  1. Frame stacking: 4 frames (provides motion/velocity info)")
        print("  2. Everything else IDENTICAL to Improvement 1")
        print("\n🧠 Why this should work:")
        print("  • Agent can see entity movement (zombies approaching)")
        print("  • Better combat decisions (when to fight/flee)")
        print("  • Improved exploration (track where you came from)")
        
        print(f"\n⚙️ Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key:25s}: {value}")
        print("="*70 + "\n")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        model_path = os.path.join(save_dir, 'model.zip')
        self.model.save(model_path)
        print(f"\n💾 Model saved to {model_path}")
        
        self.save_training_plots(save_dir)
        
        return self.model
    
    def save_training_plots(self, save_dir):
        if self.metrics_callback is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Improvement 2 (Frame Stacking) - Training Progress',
                    fontsize=16, fontweight='bold')
        
        # Training rewards
        ax = axes[0, 0]
        if len(self.metrics_callback.episode_rewards) > 0:
            ax.plot(self.metrics_callback.episode_rewards, alpha=0.2, color='blue')
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                ma = np.convolve(self.metrics_callback.episode_rewards,
                               np.ones(window)/window, mode='valid')
                ax.plot(ma, linewidth=2, color='red', label=f'{window}-ep MA')
            ax.axhline(y=3.26, color='green', linestyle='--', linewidth=2,
                      label='Baseline (3.26)')
            ax.axhline(y=4.40, color='orange', linestyle='--', linewidth=2,
                      label='Improv1 (4.40)')
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
            ax.axhline(y=3.26, color='green', linestyle='--', linewidth=2,
                      label='Baseline (3.26)')
            ax.axhline(y=4.40, color='orange', linestyle='--', linewidth=2,
                      label='Improv1 (4.40)')
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
            ax.axhline(y=177.7, color='green', linestyle='--', linewidth=2,
                      label='Baseline (177.7)')
            ax.axhline(y=193.5, color='orange', linestyle='--', linewidth=2,
                      label='Improv1 (193.5)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Survival Time')
            ax.set_title('Survival Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Geometric mean
        ax = axes[1, 1]
        if len(self.metrics_callback.geometric_means) > 0:
            steps = np.arange(len(self.metrics_callback.geometric_means)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.geometric_means, marker='^',
                   linewidth=2, color='purple', markersize=6)
            ax.axhline(y=21.79, color='green', linestyle='--', linewidth=2,
                      label='Baseline (21.79%)')
            ax.axhline(y=34.61, color='orange', linestyle='--', linewidth=2,
                      label='Improv1 (34.61%)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Geometric Mean (%)')
            ax.set_title('Achievement Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved")
    
    def evaluate(self, n_episodes=50):
        print(f"\n{'='*70}")
        print(f"EVALUATING DQN IMPROVEMENT 2 (FRAME STACKING)")
        print(f"{'='*70}\n")
        
        episode_rewards = []
        episode_lengths = []
        all_achievements = defaultdict(int)
        
        for episode in range(n_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                
                episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                episode_length += 1
                
                # Handle vectorized info
                actual_info = info[0] if isinstance(info, list) else info
                
                if 'achievements' in actual_info:
                    for achievement, unlocked in actual_info['achievements'].items():
                        if unlocked:
                            episode_achievements.add(achievement)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            for ach in episode_achievements:
                all_achievements[ach] += 1
            
            if (episode + 1) % 10 == 0:
                print(f"Progress: {episode+1}/{n_episodes} | "
                      f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                      f"Avg Length: {np.mean(episode_lengths):.1f}")
        
        # Calculate geometric mean
        achievement_rates = {ach: count / n_episodes for ach, count in all_achievements.items()}
        if achievement_rates:
            geometric_mean = np.exp(np.mean(np.log([rate + 1e-10 for rate in achievement_rates.values()]))) * 100
        else:
            geometric_mean = 0.0
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'total_unique_achievements': len(all_achievements),
            'geometric_mean': geometric_mean
        }
        
        print("\n" + "="*70)
        print("FINAL COMPARISON:")
        print("="*70)
        print(f"                              Reward    Survival    Achievements")
        print(f"Baseline:                     3.26      177.7       9/22")
        print(f"Improvement 1:                4.40      193.5       9/22")
        print(f"Improvement 3 (DuelCNN):   {metrics['mean_reward']:.2f}      {metrics['mean_length']:.1f}       {metrics['total_unique_achievements']}/22")
        print("="*70 + "\n")
        
        return metrics
    
    def load(self, model_path):
        self.model = DQN.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    agent = DuelingCNN(seed=42)
    agent.train(total_timesteps=500000, eval_freq=10000)
    agent.evaluate(n_episodes=50)


if __name__ == "__main__":
    main()