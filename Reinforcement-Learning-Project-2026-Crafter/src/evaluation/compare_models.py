import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import argparse
from src.utils.wrappers import make_crafter_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class MultiModelComparator:
    """Compare multiple Crafter agents"""
    
    def __init__(self, n_episodes=50):
        self.n_episodes = n_episodes
        
        # Crafter achievements (22 total)
        self.achievements = [
            'collect_coal', 'collect_diamond', 'collect_drink',
            'collect_iron', 'collect_sapling', 'collect_stone',
            'collect_wood', 'defeat_skeleton', 'defeat_zombie',
            'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        # Store results for all models
        self.all_results = {}
        
    def detect_frame_stacking(self, model):
        """Detect if model was trained with frame stacking by checking observation space"""
        obs_space = model.observation_space
        if hasattr(obs_space, 'shape'):
            # Check if width dimension suggests frame stacking (256 = 64 * 4)
            if len(obs_space.shape) == 3 and obs_space.shape[2] == 256:
                return True
        return False
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*70}")
        
        # Detect if model uses frame stacking
        use_frame_stacking = self.detect_frame_stacking(model)
        
        if use_frame_stacking:
            print(f"  🎬 Detected frame stacking - using VecFrameStack environment")
            # Create vectorized environment with frame stacking (matching training setup)
            def make_env():
                return make_crafter_env(preprocess_type='normalize')
            
            env = DummyVecEnv([make_env])
            env = VecFrameStack(env, n_stack=4)
        else:
            print(f"  📺 Standard environment (no frame stacking)")
            env = make_crafter_env()
        
        episode_rewards = []
        episode_lengths = []
        achievement_unlocks = defaultdict(int)
        
        # Determine if this is a vectorized environment
        is_vec_env = use_frame_stacking
        
        for episode in range(self.n_episodes):
            if is_vec_env:
                # VecEnv API: obs = env.reset()
                obs = env.reset()
            else:
                # Gym API: obs, info = env.reset()
                obs, _ = env.reset()
            
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                
                if is_vec_env:
                    # VecEnv API: obs, reward, done, info = env.step(action)
                    obs, reward, done, info = env.step(action)
                    
                    # Extract from arrays
                    if isinstance(reward, np.ndarray):
                        reward = reward[0]
                    if isinstance(done, np.ndarray):
                        done = done[0]
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Handle vectorized info
                    actual_info = info[0] if isinstance(info, list) else info
                else:
                    # Gym API: obs, reward, terminated, truncated, info = env.step(action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    actual_info = info
                
                # Track achievements
                if 'achievements' in actual_info:
                    for achievement in self.achievements:
                        if actual_info['achievements'].get(achievement, False):
                            episode_achievements.add(achievement)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            for achievement in episode_achievements:
                achievement_unlocks[achievement] += 1
            
            # Progress update
            if (episode + 1) % 10 == 0:
                print(f"  Progress: {episode+1}/{self.n_episodes} | "
                      f"Avg Reward: {np.mean(episode_rewards):.2f} | "
                      f"Avg Length: {np.mean(episode_lengths):.1f}")
        
        # Clean up environment
        env.close()
        
        # Calculate metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_survival': float(np.mean(episode_lengths)),
            'std_survival': float(np.std(episode_lengths)),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'achievement_unlocks': dict(achievement_unlocks)
        }
        
        # Achievement rates
        achievement_rates = {}
        for achievement in self.achievements:
            rate = achievement_unlocks[achievement] / self.n_episodes * 100
            achievement_rates[achievement] = rate
        
        metrics['achievement_rates'] = achievement_rates
        
        # Geometric mean
        rates = [r/100 for r in achievement_rates.values() if r > 0]
        if rates:
            geometric_mean = np.exp(np.mean(np.log(rates))) * 100
        else:
            geometric_mean = 0.0
        metrics['geometric_mean_achievements'] = float(geometric_mean)
        metrics['num_achievements_unlocked'] = sum(1 for r in achievement_rates.values() if r > 0)
        
        print(f"\n  ✅ {model_name} - Mean Reward: {metrics['mean_reward']:.2f}, "
              f"Geometric Mean: {metrics['geometric_mean_achievements']:.2f}%")
        
        return metrics
    
    def compare_models(self, model_paths, labels):
        """Compare multiple models"""
        
        print(f"\n{'='*70}")
        print(f"COMPARING {len(model_paths)} MODELS OVER {self.n_episodes} EPISODES")
        print(f"{'='*70}")
        
        for model_path, label in zip(model_paths, labels):
            if not os.path.exists(model_path):
                print(f"❌ Error: Model not found at {model_path}")
                continue
            
            print(f"\nLoading: {label}")
            model = DQN.load(model_path)
            
            metrics = self.evaluate_model(model, label)
            self.all_results[label] = metrics
        
        print(f"\n{'='*70}")
        print("COMPARISON COMPLETE")
        print(f"{'='*70}")
        
        self.print_comparison()
        
    def print_comparison(self):
        """Print comparison summary"""
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        
        # Sort by mean reward
        sorted_models = sorted(self.all_results.items(), 
                              key=lambda x: x[1]['mean_reward'], 
                              reverse=True)
        
        print("\n🏆 RANKING BY MEAN REWARD:")
        for rank, (label, metrics) in enumerate(sorted_models, 1):
            print(f"  {rank}. {label}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        
        print("\n📊 ACHIEVEMENT PERFORMANCE:")
        sorted_by_geo = sorted(self.all_results.items(),
                               key=lambda x: x[1]['geometric_mean_achievements'],
                               reverse=True)
        for rank, (label, metrics) in enumerate(sorted_by_geo, 1):
            print(f"  {rank}. {label}: {metrics['geometric_mean_achievements']:.2f}% "
                  f"({metrics['num_achievements_unlocked']}/22 unlocked)")
        
        print("="*70)
    
    def save_comparison_plots(self, save_dir):
        """Generate comparison visualizations"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        labels = list(self.all_results.keys())
        
        # 1. Reward comparison boxplot
        plt.figure(figsize=(12, 6))
        reward_data = [self.all_results[label]['episode_rewards'] for label in labels]
        bp = plt.boxplot(reward_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        plt.ylabel('Episode Reward', fontsize=12)
        plt.title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'reward_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Mean reward bar chart
        plt.figure(figsize=(10, 6))
        means = [self.all_results[label]['mean_reward'] for label in labels]
        stds = [self.all_results[label]['std_reward'] for label in labels]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
        bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('Mean Reward Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mean_reward_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Achievement geometric mean comparison
        plt.figure(figsize=(10, 6))
        geo_means = [self.all_results[label]['geometric_mean_achievements'] for label in labels]
        bars = plt.bar(labels, geo_means, color=colors, alpha=0.8)
        plt.ylabel('Geometric Mean (%)', fontsize=12)
        plt.title('Achievement Performance (Geometric Mean)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        
        for bar, gm in zip(bars, geo_means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{gm:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'geometric_mean_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Survival time comparison
        plt.figure(figsize=(10, 6))
        survival_means = [self.all_results[label]['mean_survival'] for label in labels]
        survival_stds = [self.all_results[label]['std_survival'] for label in labels]
        bars = plt.bar(labels, survival_means, yerr=survival_stds, capsize=5, color=colors, alpha=0.8)
        plt.ylabel('Mean Survival Time (steps)', fontsize=12)
        plt.title('Survival Time Comparison', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=15, ha='right')
        
        for bar, mean in zip(bars, survival_means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'survival_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Achievement heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create matrix of achievement rates
        achievement_matrix = []
        for label in labels:
            rates = [self.all_results[label]['achievement_rates'].get(ach, 0) 
                    for ach in self.achievements]
            achievement_matrix.append(rates)
        
        achievement_matrix = np.array(achievement_matrix)
        
        sns.heatmap(achievement_matrix, annot=True, fmt='.0f', 
                   xticklabels=self.achievements, yticklabels=labels,
                   cmap='YlGnBu', cbar_kws={'label': 'Unlock Rate (%)'}, ax=ax)
        plt.title('Achievement Unlock Rates Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Achievements', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'achievement_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save metrics to JSON
        with open(os.path.join(save_dir, 'comparison_metrics.json'), 'w') as f:
            # Convert for JSON serialization
            json_results = {}
            for label, metrics in self.all_results.items():
                json_results[label] = {
                    'mean_reward': metrics['mean_reward'],
                    'std_reward': metrics['std_reward'],
                    'mean_survival': metrics['mean_survival'],
                    'std_survival': metrics['std_survival'],
                    'geometric_mean_achievements': metrics['geometric_mean_achievements'],
                    'num_achievements_unlocked': metrics['num_achievements_unlocked'],
                    'achievement_rates': metrics['achievement_rates']
                }
            json.dump(json_results, f, indent=4)
        
        print(f"\n✅ Comparison plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple DQN agents on Crafter')
    parser.add_argument('models', nargs='+', help='Paths to trained DQN models')
    parser.add_argument('--labels', nargs='+', required=True,
                       help='Labels for each model')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Number of evaluation episodes per model (default: 50)')
    parser.add_argument('--save_dir', type=str, default='results/comparison',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.models) != len(args.labels):
        print(f"❌ Error: Number of models ({len(args.models)}) must match number of labels ({len(args.labels)})")
        return
    
    if len(args.models) < 2:
        print("❌ Error: At least 2 models required for comparison")
        return
    
    # Create comparator
    comparator = MultiModelComparator(n_episodes=args.n_episodes)
    
    # Run comparison
    comparator.compare_models(args.models, args.labels)
    
    # Save results
    comparator.save_comparison_plots(args.save_dir)
    
    print(f"\n✅ Comparison complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()