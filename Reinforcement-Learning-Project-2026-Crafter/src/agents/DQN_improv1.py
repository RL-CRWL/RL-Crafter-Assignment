"""
DQN Improvement 1: Enhanced Observation Processing + Better CNN Architecture

Key Improvements over Baseline:
1. Observation normalization (0-255 → 0-1) for stable learning
2. Custom CNN policy with better feature extraction
3. Larger replay buffer with prioritization-like sampling
4. Adjusted learning parameters for better exploration
5. Frame stacking for temporal information

Hypothesis: Raw pixel values (0-255) make learning unstable. Normalizing to [0,1]
and using a better CNN architecture should improve feature learning and lead to
better performance and achievement unlocking.
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
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env


class ImprovedCNN(BaseFeaturesExtractor):
    """
    Custom CNN architecture with better feature extraction for Crafter
    
    Improvements over default SB3 CNN:
    1. Deeper network (4 conv layers instead of 3)
    2. Batch normalization for stable training
    3. More filters to capture complex patterns
    4. Larger fully connected layer
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            # First conv block: 64x64x3 -> 32x32x32
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Second conv block: 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Third conv block: 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Fourth conv block: 8x8x128 -> 4x4x128
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Flatten
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


class FrameStackWrapper:
    """
    Stack the last N frames to give agent temporal information
    Helps agent understand motion and sequences
    """
    def __init__(self, env, n_frames=3):
        self.env = env
        self.n_frames = n_frames
        self.frames = None
        
        # Update observation space
        from gymnasium import spaces
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(n_frames * old_space.shape[2], old_space.shape[0], old_space.shape[1]),
            dtype=np.float32
        )
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Initialize with repeated frames
        self.frames = [obs.copy() for _ in range(self.n_frames)]
        return self._get_observation(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add new frame and remove oldest
        self.frames.pop(0)
        self.frames.append(obs.copy())
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        # Stack frames along channel dimension
        return np.concatenate(self.frames, axis=2).transpose(2, 0, 1)
    
    def close(self):
        return self.env.close()


class ImprovedMetricsCallback(BaseCallback):
    """
    Enhanced callback with better logging and early stopping
    """
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
            eval_reward, eval_length, achievements = self.evaluate_agent()
            self.eval_rewards.append(eval_reward)
            self.eval_lengths.append(eval_length)
            self.achievements_unlocked.append(achievements)
            
            # Track best model
            if eval_reward > self.best_eval_reward:
                self.best_eval_reward = eval_reward
                if self.verbose > 0:
                    print(f"  🎉 New best! Reward: {eval_reward:.2f}")
            
            if self.verbose > 0:
                print(f"\n{'='*60}")
                print(f"Evaluation at step {self.n_calls}:")
                print(f"  Mean reward:    {eval_reward:.2f}")
                print(f"  Mean length:    {eval_length:.1f}")
                print(f"  Achievements:   {achievements}")
                print(f"  Best so far:    {self.best_eval_reward:.2f}")
                print(f"{'='*60}\n")
        
        return True
    
    def evaluate_agent(self, n_episodes=5):
        """Run evaluation episodes"""
        rewards = []
        lengths = []
        all_achievements = {}
        
        for ep in range(n_episodes):
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
                
                # Track achievements
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked:
                            all_achievements[achievement] = all_achievements.get(achievement, 0) + 1
            
            rewards.append(episode_reward)
            lengths.append(episode_length)
        
        unique_achievements = len([a for a in all_achievements.values() if a > 0])
        return np.mean(rewards), np.mean(lengths), unique_achievements


class DQNImprovement1:
    """
    DQN with Improvement 1: Better observation processing and architecture
    """
    
    def __init__(self,
                 learning_rate=2.5e-4,          # Slightly higher LR with normalization
                 buffer_size=100000,             # Larger buffer
                 learning_starts=10000,          # More initial exploration
                 batch_size=64,                  # Larger batches for stability
                 gamma=0.99,
                 target_update_interval=5000,
                 exploration_fraction=0.2,       # Longer exploration
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.02,     # Lower final epsilon
                 train_freq=4,
                 gradient_steps=1,
                 device='auto',
                 seed=42,
                 use_frame_stacking=True):
        """
        Initialize improved DQN agent
        
        Key changes from baseline:
        - Normalized observations (preprocess_type='normalize')
        - Custom CNN architecture (ImprovedCNN)
        - Optional frame stacking for temporal info
        - Adjusted hyperparameters for better learning
        """
        
        self.seed = seed
        self.use_frame_stacking = use_frame_stacking
        
        # Create environments with normalization
        print("\n🔧 Creating environments with observation normalization...")
        self.train_env = make_crafter_env(
            seed=seed,
            preprocess_type='normalize'  # KEY IMPROVEMENT: Normalize to [0,1]
        )
        self.eval_env = make_crafter_env(
            seed=seed + 100,
            preprocess_type='normalize'
        )
        
        # Optional: Add frame stacking
        if use_frame_stacking:
            print("📚 Adding frame stacking (3 frames)...")
            self.train_env = FrameStackWrapper(self.train_env, n_frames=3)
            self.eval_env = FrameStackWrapper(self.eval_env, n_frames=3)
        
        # Set seeds
        self.train_env.reset(seed=seed)
        self.eval_env.reset(seed=seed + 100)
        
        # Device setup
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n💻 Using device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Hyperparameters
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
        
        # Policy kwargs for custom CNN
        policy_kwargs = dict(
            features_extractor_class=ImprovedCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, 256],  # Larger Q-network
        )
        
        print("\n🏗️ Building DQN with improved architecture...")
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
            seed=seed,
            tensorboard_log=None
        )
        
        self.metrics_callback = None
        
    def train(self, total_timesteps=500000, eval_freq=10000, save_dir='results/dqn_improvement1'):
        """Train the improved DQN agent"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup callback
        self.metrics_callback = ImprovedMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("STARTING DQN IMPROVEMENT 1 TRAINING")
        print("="*70)
        print("\n🎯 Key Improvements:")
        print("  1. ✅ Observation normalization (0-255 → 0-1)")
        print("  2. ✅ Custom CNN with batch normalization")
        print("  3. ✅ Frame stacking for temporal info" if self.use_frame_stacking else "  3. ⏭️  Frame stacking disabled")
        print("  4. ✅ Adjusted hyperparameters")
        
        print(f"\n📊 Training Parameters:")
        print(f"  Total timesteps:      {total_timesteps:,}")
        print(f"  Evaluation frequency: {eval_freq:,}")
        print(f"  Save directory:       {save_dir}")
        
        print(f"\n⚙️ Hyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key:25s}: {value}")
        print("="*70 + "\n")
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        # Save model
        model_path = os.path.join(save_dir, 'dqn_improvement1_model.zip')
        self.model.save(model_path)
        print(f"\n💾 Model saved to {model_path}")
        
        # Save training metrics
        self.save_training_plots(save_dir)
        
        # Save improvement notes
        self.save_improvement_notes(save_dir)
        
        return self.model
    
    def save_training_plots(self, save_dir):
        """Save enhanced training plots"""
        
        if self.metrics_callback is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Improvement 1 - Training Progress', fontsize=16)
        
        # Plot 1: Episode rewards
        ax = axes[0, 0]
        if len(self.metrics_callback.episode_rewards) > 0:
            ax.plot(self.metrics_callback.episode_rewards, alpha=0.3, label='Episode Reward')
            
            # Moving average
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                moving_avg = np.convolve(
                    self.metrics_callback.episode_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                ax.plot(moving_avg, linewidth=2, label=f'{window}-episode MA', color='red')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Evaluation rewards
        ax = axes[0, 1]
        if len(self.metrics_callback.eval_rewards) > 0:
            eval_steps = np.arange(len(self.metrics_callback.eval_rewards)) * self.metrics_callback.eval_freq
            ax.plot(eval_steps, self.metrics_callback.eval_rewards, marker='o', linewidth=2)
            ax.axhline(y=self.metrics_callback.best_eval_reward, color='g', linestyle='--', 
                      label=f'Best: {self.metrics_callback.best_eval_reward:.2f}')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Evaluation Reward')
            ax.set_title('Evaluation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Survival length
        ax = axes[1, 0]
        if len(self.metrics_callback.eval_lengths) > 0:
            eval_steps = np.arange(len(self.metrics_callback.eval_lengths)) * self.metrics_callback.eval_freq
            ax.plot(eval_steps, self.metrics_callback.eval_lengths, marker='s', linewidth=2, color='orange')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Survival Time')
            ax.set_title('Survival Time Progress')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Achievements
        ax = axes[1, 1]
        if len(self.metrics_callback.achievements_unlocked) > 0:
            eval_steps = np.arange(len(self.metrics_callback.achievements_unlocked)) * self.metrics_callback.eval_freq
            ax.plot(eval_steps, self.metrics_callback.achievements_unlocked, marker='^', 
                   linewidth=2, color='green')
            ax.axhline(y=22, color='r', linestyle='--', label='Total (22)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Unique Achievements')
            ax.set_title('Achievement Progress')
            ax.set_ylim([0, 24])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress_improvement1.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training plots saved")
    
    def save_improvement_notes(self, save_dir):
        """Save notes about what was improved"""
        notes = """
DQN IMPROVEMENT 1: Enhanced Observation Processing & Architecture

HYPOTHESIS:
The baseline DQN struggles because:
1. Raw pixel values (0-255) cause unstable learning
2. Default CNN architecture is too shallow for Crafter's complexity
3. No temporal information (single frame observation)

IMPROVEMENTS IMPLEMENTED:
1. Observation Normalization: 
   - Normalized pixel values from [0, 255] to [0, 1]
   - Improves gradient stability and learning speed
   
2. Custom CNN Architecture:
   - 4 convolutional layers (vs 3 in baseline)
   - Batch normalization after each conv layer
   - More filters (32->64->128->128)
   - Larger feature dimension (512 vs 256)
   
3. Frame Stacking (optional):
   - Stacks last 3 frames
   - Provides temporal information for understanding motion
   
4. Adjusted Hyperparameters:
   - Learning rate: 2.5e-4 (vs 1e-4)
   - Batch size: 64 (vs 32)
   - Buffer size: 100K (vs 50K)
   - Exploration: 20% of training (vs 15%)
   - Final epsilon: 0.02 (vs 0.05)

EXPECTED IMPROVEMENTS:
- Better feature learning from normalized inputs
- Improved achievement unlock rate
- Longer survival times
- Higher cumulative rewards

BASELINE PERFORMANCE:
- Mean Reward: 4.4
- Mean Survival: 205.2 timesteps
- Achievements: 0/22

IMPROVEMENT 1 RESULTS:
[To be filled after training]
"""
        
        with open(os.path.join(save_dir, 'IMPROVEMENT_NOTES.txt'), 'w') as f:
            f.write(notes)
        
        print(f"📝 Improvement notes saved")
    
    def evaluate(self, n_episodes=10):
        """Evaluate the trained agent"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING DQN IMPROVEMENT 1")
        print(f"{'='*70}\n")
        
        episode_rewards = []
        episode_lengths = []
        all_achievements = {}
        
        for episode in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements_set = set()
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                
                if 'achievements' in info:
                    for achievement, unlocked in info['achievements'].items():
                        if unlocked:
                            episode_achievements_set.add(achievement)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            for achievement in episode_achievements_set:
                all_achievements[achievement] = all_achievements.get(achievement, 0) + 1
            
            print(f"Episode {episode+1}/{n_episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Length={episode_length}, "
                  f"Achievements={len(episode_achievements_set)}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'total_unique_achievements': len(all_achievements),
            'achievement_counts': all_achievements
        }
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS:")
        print(f"Mean Reward:    {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean Length:    {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
        print(f"Achievements:   {metrics['total_unique_achievements']}/22")
        
        if all_achievements:
            print("\nAchievements unlocked:")
            for achievement, count in sorted(all_achievements.items()):
                print(f"  {achievement}: {count}/{n_episodes} episodes")
        
        print("="*70 + "\n")
        
        return metrics
    
    def load(self, model_path):
        """Load a saved model"""
        self.model = DQN.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    """Main training script for Improvement 1"""
    
    agent = DQNImprovement1(
        learning_rate=2.5e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=64,
        gamma=0.99,
        exploration_fraction=0.2,
        use_frame_stacking=True,  # Enable frame stacking
        seed=42
    )
    
    # Train
    agent.train(
        total_timesteps=500000,
        eval_freq=10000,
        save_dir='results/dqn_improvement1'
    )
    
    # Evaluate
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()