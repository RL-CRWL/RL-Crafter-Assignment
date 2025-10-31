"""
Compare all three PPO models with consistent evaluation

Usage:
    python Compare-PPO.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wrapper_ppo import make_crafter_env
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


class AchievementTrackingWrapper(gym.ObservationWrapper):
    """Wrapper for RecurrentPPO models that expect dict observations"""
    
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
        
        # Store original observation space
        original_obs_space = env.observation_space
        
        # Create dict observation space
        self.observation_space = spaces.Dict({
            'image': original_obs_space,
            'achievements': spaces.Box(low=0, high=1, shape=(self.num_achievements,), dtype=np.float32)
        })
        
        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.achievements = np.zeros(self.num_achievements, dtype=np.float32)
        return self.observation(obs), info
    
    def observation(self, obs):
        """Convert image observation to dict with achievements"""
        return {
            'image': obs,
            'achievements': self.achievements.copy()
        }
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update achievements from info
        if 'achievements' in info:
            for i, name in enumerate(self.achievement_names):
                if name in info['achievements'] and info['achievements'][name]:
                    self.achievements[i] = 1.0
        
        return self.observation(obs), reward, terminated, truncated, info


def calculate_geometric_mean(achievement_rates):
    """Calculate geometric mean of achievement unlock rates"""
    if not achievement_rates:
        return 0.0
    rates = [rate for rate in achievement_rates.values() if rate > 0]
    if not rates:
        return 0.0
    return np.exp(np.mean(np.log([rate + 1e-10 for rate in rates]))) * 100


def evaluate_model(model_path, env_config, n_episodes=50, model_name="Model", is_recurrent=False):
    """Evaluate a single model"""
    
    print(f"\n{'='*70}")
    print(f"Loading: {model_name}")
    print(f"{'='*70}")
    
    # Create base environment with correct preprocessing
    base_env = make_crafter_env(**env_config)
    
    # Load model based on type
    if is_recurrent:
        model = RecurrentPPO.load(model_path)
        # RecurrentPPO needs dict observations with achievements
        env = AchievementTrackingWrapper(base_env)
        print(f"  Loaded RecurrentPPO model with AchievementTrackingWrapper")
    else:
        model = PPO.load(model_path)
        # Regular PPO uses the base environment directly
        env = base_env
        print(f"  Loaded PPO model")
    
    print(f"  Model expects: {model.observation_space}")
    print(f"  Environment provides: {env.observation_space}")
    
    # Set the environment
    model.set_env(env)
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    episode_rewards = []
    episode_lengths = []
    all_achievements = defaultdict(int)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_achievements = set()
        
        # Initialize LSTM state for recurrent models
        if is_recurrent:
            state = None
            episode_start = True
        
        while not done:
            if is_recurrent:
                action, state = model.predict(obs, state=state, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
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
        
        # Progress update every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode+1}/{n_episodes} | "
                  f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                  f"Avg Length: {np.mean(episode_lengths):.1f}")
    
    # Calculate achievement rates
    achievement_rates = {ach: count / n_episodes for ach, count in all_achievements.items()}
    geometric_mean = calculate_geometric_mean(achievement_rates)
    
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_length': float(np.min(episode_lengths)),
        'max_length': float(np.max(episode_lengths)),
        'unique_achievements': len(all_achievements),
        'geometric_mean': float(geometric_mean),
        'achievement_rates': achievement_rates,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths
    }
    
    print(f"\n  ✅ {model_name} - Mean Reward: {metrics['mean_reward']:.2f}, "
          f"Geometric Mean: {geometric_mean:.2f}%")
    
    env.close()
    return metrics

def save_results(metrics, save_dir='../../results/PPO/results/ppo_baseline/', n_episodes=50):
    """Save evaluation results and plots"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, f'evaluation_metrics_{n_episodes}_episodes.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['all_rewards'], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(metrics['mean_reward'], color='red', linestyle='--', 
                label=f"Mean: {metrics['mean_reward']:.2f}")
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'reward_distribution_{n_episodes}_episodes.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot survival time distribution
    plt.figure(figsize=(10, 6))
    plt.hist(metrics['all_lengths'], bins=20, edgecolor='black', alpha=0.7, color='green')
    plt.axvline(metrics['mean_length'], color='red', linestyle='--',
                label=f"Mean: {metrics['mean_length']:.1f}")
    plt.xlabel('Survival Time (steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Survival Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, f'survival_distribution_{n_episodes}_episodes.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot achievement rates
    achievement_rates = metrics['achievement_rates']
    sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
    
    plt.figure(figsize=(12, 8))
    achievements_names = [a[0] for a in sorted_achievements]
    rates = [a[1] for a in sorted_achievements]
    
    colors = ['green' if r > 0 else 'gray' for r in rates]
    plt.barh(achievements_names, rates, color=colors, alpha=0.7)
    plt.xlabel('Unlock Rate (%)')
    plt.title('Achievement Unlock Rates')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'achievement_rates_{n_episodes}_episodes.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Results saved to {save_dir}")


def plot_comparison(all_metrics, save_path='../../results/PPO/results/comparison_plots.png', n_episodes=50):
    """Create comparison plots"""
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'PPO Models Comparison ({n_episodes} Episodes Each)', fontsize=16, fontweight='bold')
    
    models = list(all_metrics.keys())
    colors = ['green', 'orange', 'purple', 'blue', 'red'][:len(models)]
    
    # 1. Mean Rewards Comparison
    ax = axes[0, 0]
    rewards = [all_metrics[m]['mean_reward'] for m in models]
    stds = [all_metrics[m]['std_reward'] for m in models]
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, rewards, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Average Reward Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, reward, std) in enumerate(zip(bars, rewards, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Survival Time Comparison
    ax = axes[0, 1]
    lengths = [all_metrics[m]['mean_length'] for m in models]
    length_stds = [all_metrics[m]['std_length'] for m in models]
    bars = ax.bar(x_pos, lengths, yerr=length_stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Mean Survival Time (steps)')
    ax.set_title('Survival Time Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, length, std) in enumerate(zip(bars, lengths, length_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{length:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Geometric Mean Comparison
    ax = axes[0, 2]
    geom_means = [all_metrics[m]['geometric_mean'] for m in models]
    bars = ax.bar(x_pos, geom_means, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Geometric Mean (%)')
    ax.set_title('Achievement Performance (Geometric Mean)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, gm in zip(bars, geom_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{gm:.2f}%',
                ha='center', va='bottom', fontsize=9)
    
    # 4. Unique Achievements
    ax = axes[1, 0]
    unique_achs = [all_metrics[m]['unique_achievements'] for m in models]
    bars = ax.bar(x_pos, unique_achs, color=colors, alpha=0.7)
    ax.axhline(y=22, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Total (22)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylabel('Unique Achievements Unlocked')
    ax.set_title('Achievement Diversity')
    ax.set_ylim([0, 24])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, ach in zip(bars, unique_achs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ach}/22',
                ha='center', va='bottom', fontsize=9)
    
    # 5. Reward Distribution (Box Plot)
    ax = axes[1, 1]
    reward_data = [all_metrics[m]['all_rewards'] for m in models]
    bp = ax.boxplot(reward_data, tick_labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # 6. Episode Length Distribution (Box Plot)
    ax = axes[1, 2]
    length_data = [all_metrics[m]['all_lengths'] for m in models]
    bp = ax.boxplot(length_data, tick_labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Survival Time Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {save_path}")
    plt.close()


def print_detailed_comparison(all_metrics):
    """Print detailed comparison table"""
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Rank by mean reward
    models_by_reward = sorted(all_metrics.items(), 
                             key=lambda x: x[1]['mean_reward'], 
                             reverse=True)
    
    print("\n🏆 RANKING BY MEAN REWARD:")
    for i, (model, metrics) in enumerate(models_by_reward, 1):
        print(f"  {i}. {model}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    # Rank by geometric mean
    models_by_geom = sorted(all_metrics.items(), 
                           key=lambda x: x[1]['geometric_mean'], 
                           reverse=True)
    
    print("\n📊 ACHIEVEMENT PERFORMANCE:")
    for i, (model, metrics) in enumerate(models_by_geom, 1):
        print(f"  {i}. {model}: {metrics['geometric_mean']:.2f}% "
              f"({metrics['unique_achievements']}/22 unlocked)")
    
    # Detailed table
    print("\n" + "="*70)
    print("DETAILED METRICS")
    print("="*70)
    print(f"{'Model':<25} {'Reward':<15} {'Survival':<15} {'Achievements':<15} {'Geom Mean':<15}")
    print("-"*70)
    
    for model, metrics in all_metrics.items():
        print(f"{model:<25} "
              f"{metrics['mean_reward']:>6.2f}±{metrics['std_reward']:<5.2f} "
              f"{metrics['mean_length']:>7.1f}±{metrics['std_length']:<5.1f} "
              f"{metrics['unique_achievements']:>6}/22        "
              f"{metrics['geometric_mean']:>6.2f}%")
    
    print("="*70)


def main():
    """Main comparison function"""
    
    n_episodes = 10
    
    results_dir='../../results/PPO/'
    baseline_model_dir = 'ppo_baseline/ppo_baseline_model.zip'
    improv1_model_dir = 'ppo_improv_1/ppo_rnd_model.zip'
    improv2_model_dir = 'ppo_improv_2/recurrent_ppo_model.zip'
    
    print("\n" + "="*70)
    print(f"COMPARING 3 PPO MODELS OVER {n_episodes} EPISODES")
    print("="*70)
    
    # Define models to compare
    models = {
        'Baseline': {
            'path': f'{results_dir}models/{baseline_model_dir}',
            'results': f'{results_dir}/results/ppo_baseline/',
            'env_config': {
                'seed': 142,  # Different seed for fair evaluation
                'preprocess_type': 'none'
            },
            'is_recurrent': False
        },
        'Improvement 1': {
            'path': f'{results_dir}models/{improv1_model_dir}',
            'results': f'{results_dir}/results/ppo_improv_1/',
            'env_config': {
                'seed': 142,
                'preprocess_type': 'none'
            },
            'is_recurrent': False
        },
        'Improvement 2 (LSTM)': {
            'path': f'{results_dir}models/{improv2_model_dir}',
            'results': f'{results_dir}/results/ppo_improv_2/',
            'env_config': {
                'seed': 142,
                'preprocess_type': 'none'
            },
            'is_recurrent': True
        }
    }
    
    # Evaluate all models
    all_metrics = {}
    
    for model_name, config in models.items():
        if not os.path.exists(config['path']):
            print(f"\n⚠️  WARNING: Model not found at {config['path']}")
            print(f"   Skipping {model_name}")
            continue
        
        try:
            metrics = evaluate_model(
                config['path'],
                config['env_config'],
                n_episodes=n_episodes,
                model_name=model_name,
                is_recurrent=config.get('is_recurrent', False)
            )
            all_metrics[model_name] = metrics
            save_results(metrics, config['results'], n_episodes)
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_metrics:
        print("\n❌ No models successfully evaluated!")
        return
    
    # Print comparison
    print_detailed_comparison(all_metrics)
    
    jpg_results = f'{results_dir}results/final_comparison_{n_episodes}_episodes.png'
    
    # Create plots
    plot_comparison(all_metrics, jpg_results, n_episodes)
    
    # Save results to JSON
    json_results = f'{results_dir}results/final_comparison_results_{n_episodes}_episodes.json'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(json_results), exist_ok=True)
    
    # Make serializable
    serializable_metrics = {}
    for model, metrics in all_metrics.items():
        serializable_metrics[model] = {
            'mean_reward': float(metrics['mean_reward']),
            'std_reward': float(metrics['std_reward']),
            'mean_length': float(metrics['mean_length']),
            'std_length': float(metrics['std_length']),
            'unique_achievements': int(metrics['unique_achievements']),
            'geometric_mean': float(metrics['geometric_mean']),
            'achievement_rates': {k: float(v) for k, v in metrics['achievement_rates'].items()}
        }
    
    with open(json_results, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\n💾 Results saved to: {json_results}")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"  • Plots: {jpg_results}")
    print(f"  • Data:  {json_results}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()