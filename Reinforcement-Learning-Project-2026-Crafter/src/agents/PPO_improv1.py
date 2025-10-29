"""
PPO with Reward Shaping Implementation for Crafter Environment
Uses Stable Baselines3's PPO algorithm with advanced reward shaping.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.wrappers import make_crafter_env


# ===========================
#   REWARD SHAPING WRAPPERS
# ===========================

class EnhancedRewardShapingWrapper(gym.Wrapper):
    """
    Advanced reward shaping for Crafter with multiple components:
    1. Achievement bonuses (scaled by difficulty)
    2. Progressive achievement rewards
    3. Survival incentives with decay
    4. Exploration bonuses
    5. Health management rewards
    """
    
    def __init__(self, env, config=None):
        super().__init__(env)
        
        # Default configuration (can be customized)
        self.config = {
            'achievement_bonus': 10.0,          # Base bonus for achievements
            'survival_bonus': 0.01,             # Per-step survival reward
            'exploration_bonus': 0.005,         # Exploration incentive
            'health_penalty_scale': 0.02,       # Penalty for losing health
            'progression_multiplier': 1.5,      # Multiplier for key achievements
            'diversity_bonus': 2.0,             # Bonus for unique achievements
            'survival_decay': 0.9999,           # Decay rate for survival bonus
            'death_penalty': 5.0,               # Penalty for dying
        }
        if config:
            self.config.update(config)
        
        # Achievement difficulty tiers (1=easiest, 5=hardest)
        self.achievement_tiers = {
            # Tier 1: Basic collection
            'collect_wood': 1, 'collect_stone': 1, 'collect_sapling': 1,
            'collect_drink': 1, 'eat_plant': 1, 'wake_up': 1,
            # Tier 2: Simple crafting
            'make_wood_pickaxe': 2, 'make_wood_sword': 2, 'place_table': 2,
            'place_plant': 2, 'place_stone': 2,
            # Tier 3: Intermediate
            'collect_coal': 3, 'collect_iron': 3, 'make_stone_pickaxe': 3,
            'make_stone_sword': 3, 'eat_cow': 3, 'place_furnace': 3,
            # Tier 4: Advanced
            'make_iron_pickaxe': 4, 'make_iron_sword': 4,
            # Tier 5: Hardest
            'defeat_zombie': 5, 'defeat_skeleton': 5, 'collect_diamond': 5,
        }
        
        # Track achievement names from Crafter
        self.achievement_names = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
            'place_stone', 'place_table', 'wake_up'
        ]
        
        # State tracking
        self.previous_achievements = {}
        self.total_achievements = 0
        self.step_count = 0
        self.previous_health = 9.0  # Crafter starts with 9 health
        self.achievement_history = deque(maxlen=100)
        self.current_survival_bonus = self.config['survival_bonus']
    
    def reset(self, **kwargs):
        """Reset all tracking variables."""
        obs, info = super().reset(**kwargs)
        
        # Initialize achievement tracking from info
        if 'achievements' in info:
            self.previous_achievements = info['achievements'].copy()
            self.total_achievements = sum(info['achievements'].values())
        else:
            self.previous_achievements = {name: False for name in self.achievement_names}
            self.total_achievements = 0
        
        self.step_count = 0
        self.previous_health = 9.0
        self.achievement_history.clear()
        self.current_survival_bonus = self.config['survival_bonus']
        
        return obs, info
    
    def step(self, action):
        """Apply reward shaping to environment step."""
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        current_achievements = info.get('achievements', {})
        
        # === 1. Achievement Rewards (Scaled by Difficulty) ===
        achievement_reward = 0.0
        
        for achievement_name in self.achievement_names:
            prev = self.previous_achievements.get(achievement_name, False)
            curr = current_achievements.get(achievement_name, False)
            
            # Check if this is a NEW achievement
            if curr and not prev:
                tier = self.achievement_tiers.get(achievement_name, 3)
                
                # Scale reward by difficulty tier
                base_bonus = self.config['achievement_bonus']
                tier_multiplier = tier * 0.5  # Higher tier = more reward
                achievement_reward += base_bonus * tier_multiplier
                
                self.achievement_history.append(achievement_name)
                
                # Progression bonus: extra reward for gateway achievements
                if self._is_progression_achievement(achievement_name):
                    achievement_reward *= self.config['progression_multiplier']
        
        # === 2. Achievement Diversity Bonus ===
        diversity_bonus = 0.0
        unique_achievements = sum(current_achievements.values())
        if unique_achievements > self.total_achievements:
            diversity_bonus = self.config['diversity_bonus']
            self.total_achievements = unique_achievements
        
        # === 3. Survival Bonus (with decay) ===
        survival_reward = self.current_survival_bonus if not terminated else 0.0
        self.current_survival_bonus *= self.config['survival_decay']
        
        # === 4. Exploration Bonus ===
        exploration_reward = self.config['exploration_bonus']
        
        # === 5. Health Management ===
        health_reward = 0.0
        current_health = info.get('health', 9.0)
        if current_health < self.previous_health:
            health_lost = self.previous_health - current_health
            health_reward = -health_lost * self.config['health_penalty_scale']
        self.previous_health = current_health
        
        # === 6. Death Penalty ===
        death_penalty = -self.config['death_penalty'] if terminated else 0.0
        
        # === Combine All Rewards ===
        shaped_reward = (
            base_reward +
            achievement_reward +
            diversity_bonus +
            survival_reward +
            exploration_reward +
            health_reward +
            death_penalty
        )
        
        # Update tracking
        self.previous_achievements = current_achievements.copy()
        self.step_count += 1
        
        # Store detailed reward breakdown (useful for debugging)
        info['reward_breakdown'] = {
            'base': float(base_reward),
            'achievement': float(achievement_reward),
            'diversity': float(diversity_bonus),
            'survival': float(survival_reward),
            'exploration': float(exploration_reward),
            'health': float(health_reward),
            'death': float(death_penalty),
            'total': float(shaped_reward)
        }
        
        return obs, shaped_reward, terminated, truncated, info
    
    def _is_progression_achievement(self, achievement_name):
        """Check if achievement enables significant progression."""
        progression_achievements = {
            'make_wood_pickaxe',  # Enables stone collection
            'place_table',         # Enables crafting
            'make_stone_pickaxe',  # Enables coal/iron collection
            'place_furnace',       # Enables iron smelting
            'make_iron_pickaxe',   # Enables diamond collection
            'make_wood_sword',     # Enables combat
        }
        return achievement_name in progression_achievements


class SimpleRewardShapingWrapper(gym.Wrapper):
    """
    Simpler reward shaping focusing on core mechanics.
    Use this if the enhanced version seems too complex or unstable.
    """
    
    def __init__(self, env, achievement_bonus=5.0, survival_bonus=0.01, exploration_bonus=0.005):
        super().__init__(env)
        self.achievement_bonus = achievement_bonus
        self.survival_bonus = survival_bonus
        self.exploration_bonus = exploration_bonus
        
        self.achievement_names = [
            'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
            'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
            'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
            'place_stone', 'place_table', 'wake_up'
        ]
        self.previous_achievements = {}
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if 'achievements' in info:
            self.previous_achievements = info['achievements'].copy()
        else:
            self.previous_achievements = {name: False for name in self.achievement_names}
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_achievements = info.get('achievements', {})
        
        # Count new achievements
        new_achievement_count = 0
        for name in self.achievement_names:
            if current_achievements.get(name, False) and not self.previous_achievements.get(name, False):
                new_achievement_count += 1
        
        # Achievement bonus
        achievement_reward = new_achievement_count * self.achievement_bonus
        
        # Survival bonus (not given if episode ended)
        survival_reward = self.survival_bonus if not terminated else 0.0
        
        # Exploration bonus
        exploration_reward = self.exploration_bonus
        
        # Combined reward
        shaped_reward = reward + achievement_reward + survival_reward + exploration_reward
        
        self.previous_achievements = current_achievements.copy()
        
        info['reward_breakdown'] = {
            'base': float(reward),
            'achievement': float(achievement_reward),
            'survival': float(survival_reward),
            'exploration': float(exploration_reward),
            'total': float(shaped_reward)
        }
        
        return obs, shaped_reward, terminated, truncated, info


# ===========================
#   METRICS CALLBACK
# ===========================

class CrafterMetricsCallback(BaseCallback):
    """
    Custom callback to track Crafter-specific metrics during PPO training.
    Now also tracks reward breakdown from reward shaping.
    """
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
        self.reward_breakdowns = []

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


# ===========================
#   PPO AGENT WITH REWARD SHAPING
# ===========================

class PPOAgent:
    """
    PPO Agent for Crafter Environment with Reward Shaping
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
                 seed=42,
                 use_reward_shaping=True,
                 reward_shaping_type='simple',  # 'simple' or 'enhanced'
                 reward_shaping_config=None):
        
        self.seed = seed
        self.use_reward_shaping = use_reward_shaping
        self.reward_shaping_type = reward_shaping_type
        
        # Create base environments
        base_train_env = make_crafter_env()
        base_eval_env = make_crafter_env()
        
        # Apply reward shaping if enabled
        if use_reward_shaping:
            if reward_shaping_type == 'enhanced':
                print("Using Enhanced Reward Shaping")
                self.train_env = EnhancedRewardShapingWrapper(
                    base_train_env, 
                    config=reward_shaping_config
                )
                self.eval_env = EnhancedRewardShapingWrapper(
                    base_eval_env, 
                    config=reward_shaping_config
                )
            else:  # simple
                print("Using Simple Reward Shaping")
                config = reward_shaping_config or {}
                self.train_env = SimpleRewardShapingWrapper(
                    base_train_env,
                    achievement_bonus=config.get('achievement_bonus', 5.0),
                    survival_bonus=config.get('survival_bonus', 0.01),
                    exploration_bonus=config.get('exploration_bonus', 0.005)
                )
                self.eval_env = SimpleRewardShapingWrapper(
                    base_eval_env,
                    achievement_bonus=config.get('achievement_bonus', 5.0),
                    survival_bonus=config.get('survival_bonus', 0.01),
                    exploration_bonus=config.get('exploration_bonus', 0.005)
                )
        else:
            print("No Reward Shaping - Using Base Rewards")
            self.train_env = base_train_env
            self.eval_env = base_eval_env
        
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

        self.metrics_callback = None

    def train(self, total_timesteps=500000, eval_freq=10000, save_dir='results/ppo_shaped'):
        os.makedirs(save_dir, exist_ok=True)
        
        self.metrics_callback = CrafterMetricsCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            verbose=1
        )
        
        print("="*60)
        print("Starting PPO Training with Reward Shaping")
        print("="*60)
        print(f"Reward Shaping: {'Enabled (' + self.reward_shaping_type + ')' if self.use_reward_shaping else 'Disabled'}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Evaluation frequency: {eval_freq}")
        print(f"Save directory: {save_dir}")
        print("\nHyperparameters:")
        for key, value in self.hyperparameters.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.metrics_callback,
            log_interval=100
        )
        
        model_path = os.path.join(save_dir, 'ppo_shaped_model.zip')
        self.model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        self.save_training_plots(save_dir)
        return self.model

    def save_training_plots(self, save_dir):
        """Save training metric plots"""
        if self.metrics_callback is None:
            return
        
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
            plt.title('Training Rewards Over Time (with Reward Shaping)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'training_rewards.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Plot evaluation rewards
        if len(self.metrics_callback.eval_rewards) > 0:
            plt.figure(figsize=(10, 6))
            eval_steps = np.arange(len(self.metrics_callback.eval_rewards)) * self.metrics_callback.eval_freq
            plt.plot(eval_steps, self.metrics_callback.eval_rewards, marker='o', linewidth=2)
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


# ===========================
#   MAIN ENTRY POINT
# ===========================

def main():
    """Main training script"""
    
    # OPTION 1: Simple Reward Shaping (Recommended to start)
    # agent = PPOAgent(
    #     learning_rate=3e-4,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    #     gae_lambda=0.95,
    #     clip_range=0.2,
    #     ent_coef=0.01,
    #     seed=42,
    #     use_reward_shaping=True,
    #     reward_shaping_type='simple',
    #     reward_shaping_config={
    #         'achievement_bonus': 5.0,
    #         'survival_bonus': 0.01,
    #         'exploration_bonus': 0.005
    #     }
    # )
    
    # OPTION 2: Enhanced Reward Shaping (More sophisticated)
    agent = PPOAgent(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42,
        use_reward_shaping=True,
        reward_shaping_type='enhanced',
        reward_shaping_config={
            'achievement_bonus': 10.0,
            'survival_bonus': 0.01,
            'exploration_bonus': 0.005,
            'health_penalty_scale': 0.02,
            'progression_multiplier': 1.5,
            'diversity_bonus': 2.0,
        }
    )
    
    # Train
    agent.train(
        total_timesteps=3_000_000,
        eval_freq=10000,
        save_dir='results/ppo_reward_shaped'
    )
    
    # Evaluate
    agent.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()