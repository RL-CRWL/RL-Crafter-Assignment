"""
DQN Improvement 2: Strategic Reward Shaping + Stable Architecture

Key Improvements over Improvement 1:
1. Strategic reward shaping for achievement progression
2. Survival milestone bonuses
3. Health/hunger management rewards
4. Death penalty to discourage reckless behavior
5. Same stable CNN architecture (no batch norm)

Goal: Beat both baseline AND improv1 by balancing survival + achievements

Expected Results:
- Reward: 5.0-6.0+ (beat baseline 4.4 and improv1 4.1)
- Survival: 200+ steps (match/beat baseline)
- Achievements: 8-12/22 (beat improv1's 6)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrappers import make_crafter_env


class StableCNN(BaseFeaturesExtractor):
    """
    Stable CNN architecture (same as fixed improv1)
    No batch normalization for RL stability
    """
    
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, observations):
        return self.linear(self.cnn(observations))


class RewardShapingWrapper(gym.Wrapper):
    """
    Strategic reward shaping for Crafter
    
    Designed to balance survival with achievement progression
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        self.unlocked_achievements = set()
        self.episode_achievements = set()
        
        # Achievement rewards (tiered by difficulty)
        self.achievement_rewards = {
            # Tier 1: Basic survival (encourage early learning)
            'collect_drink': 0.5,
            'collect_sapling': 0.5,
            'place_plant': 1.0,
            
            # Tier 2: Resource gathering
            'collect_wood': 1.5,
            'collect_stone': 10.0,
            'collect_coal': 12.0,
            'collect_iron': 15.0,
            'collect_diamond': 20.0,
            
            # Tier 3: Building
            'place_table': 2.0,
            'place_furnace': 15.0,
            'place_stone': 12.0,
            
            # Tier 4: Crafting tools
            'make_wood_pickaxe': 15.0,
            'make_stone_pickaxe': 18.0,
            'make_iron_pickaxe': 20.0,
            'make_wood_sword': 15.0,
            'make_stone_sword': 18.0,
            'make_iron_sword': 20.0,
            
            # Tier 5: Combat & food
            'defeat_zombie': 20.0,
            'defeat_skeleton': 25.0,
            'eat_cow': 10.0,
            'eat_plant': 8.0,
            
            # Special
            'wake_up': 3.0,
        }
        
        self.steps_survived = 0
        self.last_health = None
        self.last_food = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_achievements = set()
        self.steps_survived = 0
        self.last_health = None
        self.last_food = None
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward  # Start with base survival reward (+1)
        self.steps_survived += 1
        
        # Achievement bonuses
        if 'achievements' in info:
            for achievement, unlocked in info['achievements'].items():
                if unlocked and achievement not in self.episode_achievements:
                    self.episode_achievements.add(achievement)
                    
                    # First time globally? Extra bonus!
                    if achievement not in self.unlocked_achievements:
                        self.unlocked_achievements.add(achievement)
                        bonus = self.achievement_rewards.get(achievement, 10.0)
                        shaped_reward += bonus + 5.0  # +5 for first unlock
                    else:
                        # Still reward repeating achievements (learning)
                        bonus = self.achievement_rewards.get(achievement, 10.0) * 0.5
                        shaped_reward += bonus
        
        # Survival milestone bonuses
        if self.steps_survived % 100 == 0:
            shaped_reward += 0.2
        
        # Health/hunger management
        if 'player_health' in info and 'player_food' in info:
            health = info['player_health']
            food = info['player_food']
            
            # Small bonus for maintaining health/food
            if health > 7:
                shaped_reward += 0.1
            if food > 7:
                shaped_reward += 0.1
            
            # Penalty for taking damage
            if self.last_health is not None and health < self.last_health:
                shaped_reward -= 1.0
            
            self.last_health = health
            self.last_food = food
        
        # Death penalty
        if terminated:
            shaped_reward -= 0.5
        
        return obs, shaped_reward, terminated, truncated, info


class EnhancedMetricsCallback(BaseCallback):
    """Enhanced callback with detailed tracking"""
    
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.eval_lengths = []
        self.achievements_unlocked = []
        self.achievement_details = []
        
        self.best_eval_reward = -np.inf
        self.best_achievements = 0
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate_agent()
            
            self.eval_rewards.append(metrics['mean_reward'])
            self.eval_lengths.append(metrics['mean_length'])
            self.achievements_unlocked.append(metrics['unique_achievements'])
            self.achievement_details.append(metrics['achievement_counts'])
            
            if metrics['mean_reward'] > self.best_eval_reward:
                self.best_eval_reward = metrics['mean_reward']
            if metrics['unique_achievements'] > self.best_achievements:
                self.best_achievements = metrics['unique_achievements']
            
            if self.verbose > 0:
                print(f"\n{'='*70}")
                print(f"Evaluation at step {self.n_calls:,}")
                print(f"{'='*70}")
                print(f"  Reward:        {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
                print(f"  Survival:      {metrics['mean_length']:.1f} steps")
                print(f"  Achievements:  {metrics['unique_achievements']}/22")
                print(f"  Best Reward:   {self.best_eval_reward:.2f}")
                print(f"  Best Achvmnts: {self.best_achievements}/22")
                
                if metrics['top_achievements']:
                    print(f"  Top unlocked:")
                    for ach, count in metrics['top_achievements'][:3]:
                        print(f"    - {ach}: {count}/5")
                
                print(f"{'='*70}\n")
        
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
            'unique_achievements': len(all_achievements),
            'achievement_counts': dict(all_achievements),
            'top_achievements': sorted(all_achievements.items(), 
                                      key=lambda x: x[1], reverse=True)
        }


class DQNImprovement2:
    """
    DQN Improvement 2: Strategic Reward Shaping
    """
    
    def __init__(self,
                 learning_rate=1e-4,
                 buffer_size=75000,
                 learning_starts=10000,
                 batch_size=32,
                 gamma=0.99,
                 target_update_interval=5000,
                 exploration_fraction=0.25,
                 exploration_initial_eps=1.0,
                 exploration_final_eps=0.01,
                 train_freq=4,
                 gradient_steps=1,
                 device='auto',
                 seed=42):
        
        self.seed = seed
        
        print("\n[Setup] Creating environments...")
        print("  * Observation normalization: ON")
        print("  * Strategic reward shaping: ON")
        
        # Base env with normalization
        base_env = make_crafter_env(
            seed=seed,
            preprocess_type='normalize'
        )
        # Add reward shaping
        self.train_env = RewardShapingWrapper(base_env)
        
        # Eval env WITHOUT reward shaping (fair comparison)
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
        
        print(f"\n[Hardware] Using device: {self.device}")
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
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
        
        policy_kwargs = dict(
            features_extractor_class=StableCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[256, 256],
        )
        
        print("\n[Model] Building DQN with improvements...")
        print("  * Stable CNN (no batch norm)")
        print("  * Reward shaping for achievements")
        print("  * Extended exploration (25%)")
        
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
              save_dir='results/dqn_improvement2'):
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_callback = EnhancedMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("\n" + "="*70)
        print("STARTING DQN IMPROVEMENT 2 TRAINING")
        print("="*70)
        print("\n[Goal] Beat baseline (4.4) AND improv1 (4.1)")
        print("\n[Key Improvements]")
        print("  1. Strategic reward shaping")
        print("  2. Achievement tier bonuses")
        print("  3. Survival milestone rewards")
        print("  4. Death penalty")
        print("  5. Health/hunger management")
        
        print(f"\n[Training Parameters]")
        print(f"  Total timesteps:      {total_timesteps:,}")
        print(f"  Evaluation frequency: {eval_freq:,}")
        print(f"  Save directory:       {save_dir}")
        
        print(f"\n[Hyperparameters]")
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
        model_path = os.path.join(save_dir, 'dqn_improvement2_model.zip')
        self.model.save(model_path)
        print(f"\n[Saved] Model: {model_path}")
        
        self.save_training_plots(save_dir)
        self.save_improvement_notes(save_dir)
        
        return self.model
    
    def save_training_plots(self, save_dir):
        if self.metrics_callback is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Improvement 2 - Training Progress', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Training rewards
        ax = axes[0, 0]
        if len(self.metrics_callback.episode_rewards) > 0:
            ax.plot(self.metrics_callback.episode_rewards, alpha=0.2, 
                   color='blue', label='Episode')
            
            window = 100
            if len(self.metrics_callback.episode_rewards) >= window:
                ma = np.convolve(
                    self.metrics_callback.episode_rewards,
                    np.ones(window)/window,
                    mode='valid'
                )
                ax.plot(ma, linewidth=2, color='red', label=f'{window}-ep MA')
            
            ax.axhline(y=4.4, color='green', linestyle='--', 
                      linewidth=2, label='Baseline (4.4)')
            ax.axhline(y=4.1, color='orange', linestyle='--', 
                      linewidth=2, label='Improv1 (4.1)')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Training Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Evaluation rewards
        ax = axes[0, 1]
        if len(self.metrics_callback.eval_rewards) > 0:
            steps = np.arange(len(self.metrics_callback.eval_rewards)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.eval_rewards, marker='o', 
                   linewidth=2, markersize=6)
            ax.axhline(y=4.4, color='green', linestyle='--', 
                      linewidth=2, label='Baseline (4.4)')
            ax.axhline(y=4.1, color='orange', linestyle='--', 
                      linewidth=2, label='Improv1 (4.1)')
            ax.axhline(y=self.metrics_callback.best_eval_reward, color='purple',
                      linestyle=':', linewidth=2, 
                      label=f'Best: {self.metrics_callback.best_eval_reward:.2f}')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Evaluation Reward')
            ax.set_title('Evaluation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Survival time
        ax = axes[1, 0]
        if len(self.metrics_callback.eval_lengths) > 0:
            steps = np.arange(len(self.metrics_callback.eval_lengths)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.eval_lengths, marker='s',
                   linewidth=2, color='orange', markersize=6)
            ax.axhline(y=205, color='green', linestyle='--', 
                      linewidth=2, label='Baseline (205)')
            ax.axhline(y=176, color='orange', linestyle='--', 
                      linewidth=2, label='Improv1 (176)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Mean Survival Time')
            ax.set_title('Survival Progress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Achievements
        ax = axes[1, 1]
        if len(self.metrics_callback.achievements_unlocked) > 0:
            steps = np.arange(len(self.metrics_callback.achievements_unlocked)) * \
                    self.metrics_callback.eval_freq
            ax.plot(steps, self.metrics_callback.achievements_unlocked, marker='^',
                   linewidth=2, color='green', markersize=6)
            ax.axhline(y=22, color='red', linestyle='--', 
                      linewidth=2, label='Total (22)')
            ax.axhline(y=6, color='orange', linestyle='--', 
                      linewidth=2, label='Improv1 (6)')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Unique Achievements')
            ax.set_title('Achievement Progress')
            ax.set_ylim([0, 24])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_progress_improv2.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Saved] Training plots")
    
    def save_improvement_notes(self, save_dir):
        notes = """
DQN IMPROVEMENT 2: Strategic Reward Shaping

ANALYSIS OF PREVIOUS RESULTS:
Baseline:        4.4 reward | 205 survival | 0/22 achievements
Old Improv1:     3.1 reward | 163 survival | 7/22 achievements  (FAILED)
Fixed Improv1:   4.1 reward | 176 survival | 6/22 achievements  (PARTIAL SUCCESS)

KEY INSIGHT FROM IMPROV1:
- Agent learned achievements BUT at cost of survival
- Peak performance (4.9) beat baseline, but average (4.1) slightly lower
- Trade-off: exploration for achievements vs survival optimization
- Need to align incentives: survival + achievements = maximum reward

IMPROVEMENTS IN VERSION 2:

1. STRATEGIC REWARD SHAPING:
   Tiered achievement rewards:
   - Tier 1 (Basic): collect_drink +5, place_plant +8
   - Tier 2 (Resources): collect_wood +10, collect_coal +12
   - Tier 3 (Building): place_table +15, place_furnace +15
   - Tier 4 (Tools): make_pickaxe +15-20, make_sword +15-20
   - Tier 5 (Combat): defeat_zombie +20, defeat_skeleton +25
   
   First unlock bonus: +5 additional reward
   Repeat unlock: 50% of base reward (encourages consistency)

2. SURVIVAL INCENTIVES:
   - Milestone bonuses: +2 reward every 100 steps
   - Health maintenance: +0.1 if health >7
   - Hunger maintenance: +0.1 if food >7
   - Damage penalty: -1.0 when taking damage
   - Death penalty: -5.0 to discourage reckless play

3. BALANCED OBJECTIVES:
   Instead of: Survival OR Achievements
   Now: Survival AND Achievements = Maximum Reward
   
   Agent learns that:
   - Long survival + achievements = best reward
   - Short survival + achievements = medium reward
   - Long survival + no achievements = baseline reward
   - Short survival + no achievements = low reward

4. KEPT FROM IMPROV1:
   - Stable CNN architecture (no batch norm)
   - Observation normalization
   - Extended exploration (25%)
   - Conservative learning rate (1e-4)

EXPECTED IMPROVEMENTS:
Mean reward:    >5.0 (beat both baseline and improv1)
Survival:       >200 (match/beat baseline)
Achievements:   >8 (significantly beat improv1's 6)
Stability:      High (stable architecture + aligned incentives)

TARGET PERFORMANCE:
Conservative: 5.0-5.5 reward, 200+ survival, 8-10 achievements
Optimistic:   5.5-6.5 reward, 220+ survival, 12-15 achievements

KEY INNOVATION:
This improvement doesn't change the algorithm or architecture.
Instead, it changes the REWARD SIGNAL to align with our goals.
This is a fundamental RL principle: good reward design is critical!
"""
        
        with open(os.path.join(save_dir, 'IMPROVEMENT_NOTES.txt'), 'w', 
                 encoding='utf-8') as f:
            f.write(notes)
        
        print(f"[Saved] Improvement notes")
    
    def evaluate(self, n_episodes=10):
        print(f"\n{'='*70}")
        print(f"EVALUATING DQN IMPROVEMENT 2")
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
                  f"Reward={episode_reward:.1f}, "
                  f"Length={episode_length:3d}, "
                  f"Achievements={len(episode_achievements)}")
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'total_unique_achievements': len(all_achievements),
            'achievement_counts': dict(all_achievements)
        }
        
        print("\n" + "="*70)
        print("COMPARISON:")
        print("="*70)
        print(f"                    Reward    Survival    Achievements")
        print(f"Baseline:           4.4       205.2       0/22")
        print(f"Improvement 1:      4.1       176.2       6/22")
        print(f"Improvement 2:      {metrics['mean_reward']:.1f}       {metrics['mean_length']:.1f}       {metrics['total_unique_achievements']}/22")
        
        print("\n" + "="*70)
        if metrics['mean_reward'] > 4.4:
            print("[SUCCESS] Beat baseline reward!")
        if metrics['mean_reward'] > 4.1:
            print("[SUCCESS] Beat improvement 1 reward!")
        if metrics['mean_length'] > 200:
            print("[SUCCESS] Beat baseline survival!")
        if metrics['total_unique_achievements'] > 6:
            print("[SUCCESS] Beat improvement 1 achievements!")
        print("="*70)
        
        if all_achievements:
            print(f"\nAchievements unlocked:")
            for achievement, count in sorted(all_achievements.items(), 
                                            key=lambda x: x[1], reverse=True):
                bar = '#' * count + '-' * (n_episodes - count)
                print(f"  {achievement:20s} [{bar}] {count:2d}/{n_episodes}")
        
        print("="*70 + "\n")
        
        return metrics
    
    def load(self, model_path):
        self.model = DQN.load(model_path, env=self.train_env)
        print(f"Model loaded from {model_path}")


def main():
    """Main training script for Improvement 2"""
    
    print("\n" + "="*70)
    print("DQN IMPROVEMENT 2: STRATEGIC REWARD SHAPING")
    print("="*70)
    
    agent = DQNImprovement2(
        learning_rate=1e-4,
        buffer_size=75000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.25,
        exploration_final_eps=0.01,
        seed=42
    )
    
    # Train
    agent.train(
        total_timesteps=500000,
        eval_freq=10000,
        save_dir='results/dqn_improvement2'
    )
    
    # Evaluate
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()