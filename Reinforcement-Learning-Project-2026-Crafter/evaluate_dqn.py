"""
Comprehensive evaluation script for DQN agent on Crafter
Tracks all metrics required by the assignment
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from utils.wrappers import make_crafter_env
from agents.DQN_baseline import DQNAgent


class CrafterEvaluator:
    """
    Comprehensive evaluator for Crafter agents
    Tracks: achievements, survival time, rewards
    """
    
    def __init__(self, agent, n_episodes=20):
        """
        Args:
            agent: Trained agent to evaluate
            n_episodes: Number of episodes for evaluation
        """
        self.agent = agent
        self.n_episodes = n_episodes
        self.env = make_crafter_env()
        
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
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        self.achievement_unlocks = defaultdict(int)
        
    def evaluate(self):
        """Run comprehensive evaluation"""
        
        print("="*70)
        print(f"EVALUATING AGENT OVER {self.n_episodes} EPISODES")
        print("="*70)
        
        for episode in range(self.n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()
            
            while not done:
                # Get action from agent
                action, _ = self.agent.model.predict(obs, deterministic=True)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Track achievements (if available in info)
                if 'achievements' in info:
                    for achievement in self.achievements:
                        if info['achievements'].get(achievement, False):
                            episode_achievements.add(achievement)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Update achievement counts
            for achievement in episode_achievements:
                self.achievement_unlocks[achievement] += 1
            
            # Progress update
            if (episode + 1) % 5 == 0:
                print(f"Completed {episode+1}/{self.n_episodes} episodes | "
                      f"Avg Reward: {np.mean(self.episode_rewards):.2f} | "
                      f"Avg Length: {np.mean(self.episode_lengths):.1f}")
        
        # Calculate final metrics
        metrics = self.calculate_metrics()
        self.print_results(metrics)
        
        return metrics
    
    def calculate_metrics(self):
        """Calculate all evaluation metrics"""
        
        metrics = {}
        
        # Reward metrics
        metrics['mean_reward'] = np.mean(self.episode_rewards)
        metrics['std_reward'] = np.std(self.episode_rewards)
        metrics['min_reward'] = np.min(self.episode_rewards)
        metrics['max_reward'] = np.max(self.episode_rewards)
        
        # Survival metrics
        metrics['mean_survival'] = np.mean(self.episode_lengths)
        metrics['std_survival'] = np.std(self.episode_lengths)
        metrics['min_survival'] = np.min(self.episode_lengths)
        metrics['max_survival'] = np.max(self.episode_lengths)
        
        # Achievement metrics
        achievement_rates = {}
        for achievement in self.achievements:
            rate = self.achievement_unlocks[achievement] / self.n_episodes * 100
            achievement_rates[achievement] = rate
        
        metrics['achievement_rates'] = achievement_rates
        
        # Geometric mean of achievement rates (Crafter's primary metric)
        rates = [r/100 for r in achievement_rates.values() if r > 0]
        if rates:
            geometric_mean = np.exp(np.mean(np.log(rates))) * 100
        else:
            geometric_mean = 0.0
        metrics['geometric_mean_achievements'] = geometric_mean
        
        # Count unlocked achievements
        metrics['num_achievements_unlocked'] = sum(1 for r in achievement_rates.values() if r > 0)
        metrics['total_achievements'] = len(self.achievements)
        
        return metrics
    
    def print_results(self, metrics):
        """Print evaluation results"""
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print("\n📊 REWARD METRICS:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        
        print("\n⏱️  SURVIVAL METRICS:")
        print(f"  Mean Survival Time: {metrics['mean_survival']:.1f} ± {metrics['std_survival']:.1f} steps")
        print(f"  Range: [{metrics['min_survival']:.0f}, {metrics['max_survival']:.0f}] steps")
        
        print("\n🏆 ACHIEVEMENT METRICS:")
        print(f"  Achievements Unlocked: {metrics['num_achievements_unlocked']}/{metrics['total_achievements']}")
        print(f"  Geometric Mean: {metrics['geometric_mean_achievements']:.2f}%")
        
        # Show top achievements
        sorted_achievements = sorted(
            metrics['achievement_rates'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n  Top 10 Most Unlocked Achievements:")
        for i, (achievement, rate) in enumerate(sorted_achievements[:10], 1):
            if rate > 0:
                print(f"    {i}. {achievement}: {rate:.1f}%")
        
        print("="*70)
    
    def save_results(self, save_dir):
        """Save evaluation results and plots"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save metrics to JSON
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot reward distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mean_reward'], color='red', linestyle='--', 
                   label=f"Mean: {metrics['mean_reward']:.2f}")
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'reward_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot survival time distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.episode_lengths, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.axvline(metrics['mean_survival'], color='red', linestyle='--',
                   label=f"Mean: {metrics['mean_survival']:.1f}")
        plt.xlabel('Survival Time (steps)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Survival Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'survival_distribution.png'),
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
        plt.savefig(os.path.join(save_dir, 'achievement_rates.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Results saved to {save_dir}")


def main():
    """Main evaluation script"""
    
    # Load trained agent
    print("Loading trained DQN agent...")
    agent = DQNAgent()
    agent.load('results/dqn_baseline/dqn_baseline_model.zip')
    
    # Create evaluator
    evaluator = CrafterEvaluator(agent, n_episodes=20)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Save results
    evaluator.save_results('results/dqn_baseline/evaluation')


if __name__ == "__main__":
    main()