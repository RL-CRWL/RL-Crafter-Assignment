# evaluation.py
import numpy as np
from collections import defaultdict
from scipy.stats.mstats import gmean

def evaluate_model(model, num_episodes=100):
    """
    Evaluate the trained model on Crafter environment
    Returns: dict with all required metrics
    """
    # Recreate the environment (same as training but without recorder)
    import crafter
    from shimmy import GymV21CompatibilityV0
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
    
    env = crafter.Env(reward=True)
    env = GymV21CompatibilityV0(env=env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    # Track metrics
    survival_times = []
    cumulative_rewards = []
    achievement_counts = defaultdict(int)
    
    print(f"Evaluating model over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        timestep = 0
        episode_achievements = set()
        
        # Store initial achievement counts
        if episode == 0 and timestep == 0:
            # Take one step to get initial state
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            initial_achievements = info[0]['achievements'].copy()
            # Reset properly for the episode
            obs = env.reset()
            done = False
            episode_reward = 0.0
            timestep = 0
        
        # Track achievements at the start of episode
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        previous_achievements = info[0]['achievements'].copy()
        episode_reward += float(reward[0])
        timestep += 1
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += float(reward[0])
            timestep += 1
            
            # Track NEW achievements by comparing with previous step
            if info and len(info) > 0:
                current_achievements = info[0]['achievements']
                
                # Check for new achievements
                for achievement, count in current_achievements.items():
                    if count > previous_achievements.get(achievement, 0):
                        episode_achievements.add(achievement)
                
                previous_achievements = current_achievements.copy()
        
        # Store episode results
        survival_times.append(timestep)
        cumulative_rewards.append(episode_reward)
        
        # Count achievements for this episode
        for achievement in episode_achievements:
            achievement_counts[achievement] += 1
        
        print(f"Episode {episode + 1}: Survival={timestep}, Reward={episode_reward:.1f}, Achievements={len(episode_achievements)}")
        if episode_achievements:
            print(f"  Achievements unlocked: {list(episode_achievements)}")
    
    env.close()
    
    # Calculate final metrics
    metrics = {
        'survival_time': {
            'mean': np.mean(survival_times),
            'std': np.std(survival_times)
        },
        'cumulative_reward': {
            'mean': np.mean(cumulative_rewards),
            'std': np.std(cumulative_rewards)
        },
        'achievement_unlock_rates': {},
        'geometric_mean_achievements': 0.0
    }
    
    # Calculate achievement unlock rates
    for achievement, count in achievement_counts.items():
        rate = count / num_episodes
        metrics['achievement_unlock_rates'][achievement] = rate
    
    # Calculate geometric mean (only if we have achievements)
    if metrics['achievement_unlock_rates']:
        rates_list = list(metrics['achievement_unlock_rates'].values())
        metrics['geometric_mean_achievements'] = float(gmean(rates_list))
    
    return metrics

def print_evaluation_results(metrics, model_name):
    """Print formatted evaluation results"""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {model_name}")
    print(f"{'='*60}")
    
    print(f"\nSURVIVAL TIME:")
    print(f"  Average: {metrics['survival_time']['mean']:.1f} ± {metrics['survival_time']['std']:.1f} steps")
    
    print(f"\nCUMULATIVE REWARD:")
    print(f"  Average: {metrics['cumulative_reward']['mean']:.1f} ± {metrics['cumulative_reward']['std']:.1f}")
    
    print(f"\nACHIEVEMENTS:")
    if metrics['achievement_unlock_rates']:
        print(f"  Geometric Mean of Unlock Rates: {metrics['geometric_mean_achievements']:.4f}")
        print(f"  Unlock Rates ({len(metrics['achievement_unlock_rates'])} unique achievements):")
        for achievement, rate in sorted(metrics['achievement_unlock_rates'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"    {achievement}: {rate:.1%}")
    else:
        print("  No achievements unlocked during evaluation")
        print(f"  Geometric Mean of Unlock Rates: 0.0000")
    
    print(f"{'='*60}\n")