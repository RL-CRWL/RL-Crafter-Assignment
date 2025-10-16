import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime

# Now import from src
from src.utils.wrappers import make_crafter_env
from stable_baselines3 import DQN

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. GIF creation will be skipped.")


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)


class CrafterEvaluator:
    """Comprehensive evaluator for Crafter agents"""
    
    def __init__(self, n_episodes=20, create_gifs=False, gif_dir=None):
        self.n_episodes = n_episodes
        self.env = make_crafter_env()
        self.create_gifs = create_gifs and PIL_AVAILABLE
        self.gif_dir = gif_dir
        
        # 22 Crafter achievements
        self.achievements = [
            'collect_coal', 'collect_diamond', 'collect_drink',
            'collect_iron', 'collect_sapling', 'collect_stone',
            'collect_wood', 'defeat_skeleton', 'defeat_zombie',
            'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_frames = []
        self.achievement_unlocks = defaultdict(int)
        self.episode_achievements = []
    
    def evaluate(self, model, agent_name="Agent"):
        """Run comprehensive evaluation"""
        
        print("\n" + "="*70)
        print(f"EVALUATING {agent_name.upper()} OVER {self.n_episodes} EPISODES")
        if self.create_gifs:
            print("GIF RECORDING: ENABLED")
        print("="*70)
        
        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            frames = []
            
            while not done:
                # Record frame if creating GIFs
                if self.create_gifs:
                    frame = self.env.render()
                    if frame is not None:
                        frames.append(frame)
                
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                # Track achievements
                if 'achievements' in info:
                    for achievement in self.achievements:
                        if info['achievements'].get(achievement, False):
                            episode_achievements.add(achievement)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_achievements.append(episode_achievements)
            self.episode_frames.append(frames)
            
            # Update achievement counts
            for achievement in episode_achievements:
                self.achievement_unlocks[achievement] += 1
            
            if (episode + 1) % 5 == 0:
                print(f"  Completed {episode+1}/{self.n_episodes} episodes | "
                      f"Avg Reward: {np.mean(self.episode_rewards):.2f} | "
                      f"Avg Length: {np.mean(self.episode_lengths):.1f}")
        
        metrics = self.calculate_metrics()
        self.print_results(metrics, agent_name)
        
        return metrics
    
    def calculate_metrics(self):
        """Calculate all required metrics"""
        
        metrics = {}
        
        # Reward metrics
        metrics['mean_reward'] = float(np.mean(self.episode_rewards))
        metrics['std_reward'] = float(np.std(self.episode_rewards))
        metrics['min_reward'] = float(np.min(self.episode_rewards))
        metrics['max_reward'] = float(np.max(self.episode_rewards))
        
        # Survival metrics
        metrics['mean_survival'] = float(np.mean(self.episode_lengths))
        metrics['std_survival'] = float(np.std(self.episode_lengths))
        
        # Achievement metrics
        achievement_rates = {}
        for achievement in self.achievements:
            rate = (self.achievement_unlocks[achievement] / self.n_episodes) * 100
            achievement_rates[achievement] = rate
        
        metrics['achievement_rates'] = achievement_rates
        
        # Geometric mean
        rates = [r/100 for r in achievement_rates.values() if r > 0]
        if rates:
            geometric_mean = np.exp(np.mean(np.log(rates))) * 100
        else:
            geometric_mean = 0.0
        metrics['geometric_mean'] = float(geometric_mean)
        
        # Count unique achievements
        metrics['achievements_unlocked'] = sum(1 for r in achievement_rates.values() if r > 0)
        metrics['total_achievements'] = len(self.achievements)
        
        return metrics
    
    def print_results(self, metrics, agent_name):
        """Print evaluation results"""
        
        print("\n" + "="*70)
        print(f"RESULTS FOR {agent_name}")
        print("="*70)
        
        print(f"\nREWARD METRICS:")
        print(f"  Mean: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        
        print(f"\nSURVIVAL METRICS:")
        print(f"  Mean Time: {metrics['mean_survival']:.1f} ± {metrics['std_survival']:.1f} steps")
        
        print(f"\nACHIEVEMENT METRICS:")
        print(f"  Unlocked: {metrics['achievements_unlocked']}/{metrics['total_achievements']}")
        print(f"  Geometric Mean: {metrics['geometric_mean']:.2f}%")
        
        # Top achievements
        sorted_ach = sorted(metrics['achievement_rates'].items(), 
                           key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 Achievements:")
        for i, (ach, rate) in enumerate(sorted_ach[:5], 1):
            if rate > 0:
                print(f"    {i}. {ach}: {rate:.1f}%")
        
        print("="*70)
    
    def save_gifs(self, agent_name):
        """Save episode GIFs"""
        
        if not self.create_gifs or not self.gif_dir:
            return
        
        os.makedirs(self.gif_dir, exist_ok=True)
        
        print(f"\nSaving GIFs...")
        gif_count = 0
        
        for episode, frames in enumerate(self.episode_frames):
            if len(frames) == 0:
                continue
            
            # Convert frames to PIL Images
            pil_frames = []
            for frame in frames:
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                
                pil_frames.append(Image.fromarray(frame))
            
            if pil_frames:
                gif_path = os.path.join(self.gif_dir, 
                                       f'{agent_name}_episode_{episode+1:03d}.gif')
                
                # Save as GIF with loop and duration
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,  # 100ms per frame
                    loop=0  # loop forever
                )
                gif_count += 1
                
                if (gif_count) % 5 == 0:
                    print(f"  Saved {gif_count} GIFs...")
        
        print(f"✓ Saved {gif_count} GIFs to {self.gif_dir}")
    
    def save_results(self, save_dir, agent_name):
        """Save evaluation results and plots"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        metrics = self.calculate_metrics()
        
        # Save JSON
        with open(os.path.join(save_dir, f'{agent_name}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Reward distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.episode_rewards, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(metrics['mean_reward'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {metrics['mean_reward']:.2f}")
        ax.set_xlabel('Episode Reward', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{agent_name} - Reward Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{agent_name}_reward_dist.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Survival time distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.episode_lengths, bins=15, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(metrics['mean_survival'], color='red', linestyle='--', linewidth=2,
                   label=f"Mean: {metrics['mean_survival']:.1f}")
        ax.set_xlabel('Survival Time (steps)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{agent_name} - Survival Time Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{agent_name}_survival_dist.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Achievement rates
        sorted_ach = sorted(metrics['achievement_rates'].items(), 
                           key=lambda x: x[1], reverse=True)
        names = [a[0] for a in sorted_ach]
        rates = [a[1] for a in sorted_ach]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if r > 0 else 'lightgray' for r in rates]
        ax.barh(names, rates, color=colors, alpha=0.7)
        ax.set_xlabel('Unlock Rate (%)', fontsize=12)
        ax.set_title(f'{agent_name} - Achievement Unlock Rates', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{agent_name}_achievements.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Results saved to {save_dir}")


def compare_agents(results_dirs, output_dir):
    """Compare multiple agent results from saved metrics"""
    
    print("\n" + "="*70)
    print("COMPARING AGENTS")
    print("="*70)
    
    all_metrics = {}
    agent_names = []
    
    for result_dir in results_dirs:
        # Try to find metrics.json in the directory
        metrics_file = None
        
        # Check for *_metrics.json files in the directory
        for file in os.listdir(result_dir):
            if file.endswith('_metrics.json'):
                metrics_file = os.path.join(result_dir, file)
                agent_name = file.replace('_metrics.json', '')
                break
        
        if metrics_file and os.path.exists(metrics_file):
            agent_names.append(agent_name)
            with open(metrics_file, 'r') as f:
                all_metrics[agent_name] = json.load(f)
            print(f"✓ Loaded metrics for {agent_name}")
        else:
            print(f"✗ Warning: Could not find metrics for {result_dir}")
    
    if not all_metrics:
        print("✗ No metrics found to compare")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    rows = []
    metrics_to_compare = ['mean_reward', 'std_reward', 'mean_survival', 
                         'geometric_mean', 'achievements_unlocked']
    
    for metric in metrics_to_compare:
        row = [metric.replace('_', ' ').title()]
        for agent_name in agent_names:
            if agent_name in all_metrics:
                value = all_metrics[agent_name].get(metric, 'N/A')
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            else:
                row.append("N/A")
        rows.append(row)
    
    table = ax.table(cellText=rows,
                    colLabels=['Metric'] + agent_names,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Agent Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'comparison_table.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Reward comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [all_metrics[name]['mean_reward'] for name in agent_names]
    stds = [all_metrics[name]['std_reward'] for name in agent_names]
    
    ax.bar(agent_names, means, yerr=stds, capsize=10, alpha=0.7, color='steelblue')
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Mean Reward Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_comparison.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # Achievement comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    geo_means = []
    for name in agent_names:
        geo_mean = all_metrics[name].get('geometric_mean', 
                  all_metrics[name].get('geometric_mean_achievements', 0))
        geo_means.append(geo_mean)
    
    ax.bar(agent_names, geo_means, alpha=0.7, color='green')
    ax.set_ylabel('Geometric Mean Achievement Rate (%)', fontsize=12)
    ax.set_title('Achievement Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'achievement_comparison.png'),
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and visualize DQN agents on Crafter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model and generate plots
  python Visualiser.py --model results/DQN_Baseline/dqn_baseline_model.zip --save_dir results/eval_plots

  # Evaluate with GIF recording
  python Visualiser.py --model results/DQN_Baseline/dqn_baseline_model.zip --create_gif --save_dir results/eval_plots --gif_dir results/gifs

  # Compare multiple evaluation results
  python Visualiser.py --compare results/eval_baseline results/eval_v1 --output results/comparison
        """
    )
    
    parser.add_argument('--model', type=str,
                       help='Path to trained model to evaluate')
    parser.add_argument('--n_episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--save_dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--create_gif', action='store_true',
                       help='Create GIFs of episodes')
    parser.add_argument('--gif_dir', type=str, default='results/gifs',
                       help='Directory to save GIFs (used with --create_gif)')
    parser.add_argument('--compare', nargs='+',
                       help='Multiple evaluation result directories to compare')
    parser.add_argument('--output', type=str, default='results/comparison',
                       help='Output directory for comparison plots')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare multiple agents
        print(f"Comparing {len(args.compare)} evaluation results...")
        compare_agents(args.compare, args.output)
    
    elif args.model:
        # Single agent evaluation and visualization
        print("\n" + "="*70)
        print("LOADING MODEL")
        print("="*70)
        
        if not os.path.exists(args.model):
            print(f"✗ Error: Model not found at {args.model}")
            return
        
        model = DQN.load(args.model)
        print(f"✓ Model loaded from {args.model}")
        
        # Extract agent name from model path
        agent_name = Path(args.model).parent.name
        
        evaluator = CrafterEvaluator(
            n_episodes=args.n_episodes,
            create_gifs=args.create_gif,
            gif_dir=args.gif_dir if args.create_gif else None
        )
        metrics = evaluator.evaluate(model, agent_name=agent_name)
        evaluator.save_results(args.save_dir, agent_name)
        
        if args.create_gif:
            evaluator.save_gifs(agent_name)
        
        print(f"\n✓ Evaluation and visualization complete!")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()