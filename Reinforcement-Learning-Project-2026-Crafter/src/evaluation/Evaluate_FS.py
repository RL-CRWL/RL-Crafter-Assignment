import sys
import os
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import argparse

from src.utils.wrappers import make_crafter_env
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class CrafterEvaluator:
    """
    Evaluator for Crafter agents with automatic frame stacking
    and plotting of reward, survival, and achievement metrics.
    """

    def __init__(self, n_episodes=20):
        self.n_episodes = n_episodes

        self.achievements = [
            'collect_coal', 'collect_diamond', 'collect_drink',
            'collect_iron', 'collect_sapling', 'collect_stone',
            'collect_wood', 'defeat_skeleton', 'defeat_zombie',
            'eat_cow', 'eat_plant', 'make_iron_pickaxe',
            'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
            'make_wood_pickaxe', 'make_wood_sword', 'place_furnace',
            'place_plant', 'place_stone', 'place_table', 'wake_up'
        ]

        self.episode_rewards = []
        self.episode_lengths = []
        self.achievement_unlocks = defaultdict(int)

        self.env = None
        self.is_vec_env = False

    def _setup_environment(self, model):
        """Set up environment matching model's frame stacking."""
        obs_shape = model.observation_space.shape

        # Width-stacked frames (3, 64, 256)
        if len(obs_shape) == 3 and obs_shape[0] == 3 and obs_shape[2] > 64:
            n_stack = obs_shape[2] // 64
            print(f"🎬 Detected width-stacked frames ({n_stack} frames)")
            def make_env(): return make_crafter_env(preprocess_type='normalize')
            vec_env = DummyVecEnv([make_env])
            self.env = VecFrameStack(vec_env, n_stack=n_stack)
            self.is_vec_env = True

        # Channel-stacked frames (12, 64, 64)
        elif len(obs_shape) == 3 and obs_shape[0] > 3 and obs_shape[1:] == (64, 64):
            n_stack = obs_shape[0] // 3
            print(f"🎬 Detected channel-stacked frames ({n_stack} frames)")
            def make_env(): return make_crafter_env(preprocess_type='normalize')
            vec_env = DummyVecEnv([make_env])
            self.env = VecFrameStack(vec_env, n_stack=n_stack)
            self.is_vec_env = True

        else:
            print("📺 Standard environment (no frame stacking)")
            self.env = make_crafter_env(preprocess_type='normalize')
            self.is_vec_env = False

    def evaluate(self, model):
        """Evaluate the model over n_episodes."""
        print("="*70)
        print(f"EVALUATING AGENT OVER {self.n_episodes} EPISODES")
        print("="*70)

        self._setup_environment(model)
        print(f"Model expects observation shape: {model.observation_space.shape}")

        for episode in range(self.n_episodes):
            obs = self.env.reset() if self.is_vec_env else self.env.reset()[0]
            done = False
            episode_reward = 0
            episode_length = 0
            episode_achievements = set()

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if self.is_vec_env:
                    obs, reward, done, info = self.env.step(action)
                    reward = reward[0] if isinstance(reward, np.ndarray) else reward
                    done = done[0] if isinstance(done, np.ndarray) else done
                    actual_info = info[0] if isinstance(info, list) else info
                else:
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    actual_info = info

                episode_reward += reward
                episode_length += 1

                if 'achievements' in actual_info:
                    for achievement in self.achievements:
                        if actual_info['achievements'].get(achievement, False):
                            episode_achievements.add(achievement)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            for achievement in episode_achievements:
                self.achievement_unlocks[achievement] += 1

            if (episode + 1) % 5 == 0:
                print(f"Completed {episode+1}/{self.n_episodes} | "
                      f"Avg Reward: {np.mean(self.episode_rewards):.2f} | "
                      f"Avg Length: {np.mean(self.episode_lengths):.1f}")

        metrics = self.calculate_metrics()
        self.print_results(metrics)
        return metrics

    def calculate_metrics(self):
        """Compute evaluation metrics."""
        metrics = {
            'mean_reward': float(np.mean(self.episode_rewards)),
            'std_reward': float(np.std(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
            'max_reward': float(np.max(self.episode_rewards)),
            'mean_survival': float(np.mean(self.episode_lengths)),
            'std_survival': float(np.std(self.episode_lengths)),
            'min_survival': float(np.min(self.episode_lengths)),
            'max_survival': float(np.max(self.episode_lengths))
        }

        achievement_rates = {a: self.achievement_unlocks[a] / self.n_episodes * 100
                             for a in self.achievements}
        metrics['achievement_rates'] = achievement_rates
        metrics['num_achievements_unlocked'] = sum(1 for r in achievement_rates.values() if r>0)
        metrics['total_achievements'] = len(self.achievements)

        rates = [r/100 for r in achievement_rates.values() if r>0]
        metrics['geometric_mean_achievements'] = np.exp(np.mean(np.log(rates))) * 100 if rates else 0.0
        return metrics

    def print_results(self, metrics):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"📊 Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"⏱️  Mean Survival: {metrics['mean_survival']:.1f} ± {metrics['std_survival']:.1f}")
        print(f"🏆 Achievements: {metrics['num_achievements_unlocked']}/{metrics['total_achievements']}")
        print(f"📈 Geometric Mean Achievements: {metrics['geometric_mean_achievements']:.2f}%")
        print("="*70)

    def save_results(self, save_dir):
        """Save metrics and generate plots."""
        os.makedirs(save_dir, exist_ok=True)
        metrics = self.calculate_metrics()

        # Save metrics
        with open(os.path.join(save_dir,'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        # Reward distribution
        plt.figure(figsize=(10,6))
        plt.hist(self.episode_rewards, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(metrics['mean_reward'], color='red', linestyle='--', label=f"Mean: {metrics['mean_reward']:.2f}")
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Episode Reward Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir,'reward_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Survival distribution
        plt.figure(figsize=(10,6))
        plt.hist(self.episode_lengths, bins=20, edgecolor='black', alpha=0.7, color='green')
        plt.axvline(metrics['mean_survival'], color='red', linestyle='--', label=f"Mean: {metrics['mean_survival']:.1f}")
        plt.xlabel('Survival Time (steps)')
        plt.ylabel('Frequency')
        plt.title('Survival Time Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir,'survival_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Achievement rates
        achievement_rates = metrics['achievement_rates']
        sorted_achievements = sorted(achievement_rates.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(12,8))
        names = [a[0] for a in sorted_achievements]
        rates = [a[1] for a in sorted_achievements]
        colors = ['green' if r>0 else 'gray' for r in rates]
        plt.barh(names, rates, color=colors, alpha=0.7)
        plt.xlabel('Unlock Rate (%)')
        plt.title('Achievement Unlock Rates')
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,'achievement_rates.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ Results and plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate a DQN agent on Crafter')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='results/evaluation')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return

    model = DQN.load(args.model)
    evaluator = CrafterEvaluator(n_episodes=args.n_episodes)
    evaluator.evaluate(model)
    evaluator.save_results(args.save_dir)


if __name__ == "__main__":
    main()
