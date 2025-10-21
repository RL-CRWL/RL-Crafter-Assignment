"""
Optimized training configuration for NVIDIA GTX 750 Ti (2GB VRAM)

This configuration is specifically tuned for the GTX 750 Ti's capabilities:
- 2GB VRAM (conservative batch sizes)
- 640 CUDA cores (moderate model complexity)
- Efficient Maxwell architecture (good for RL training)
"""

# ============================================================================
# DQN BASELINE CONFIGURATION (Algorithm from course)
# ============================================================================
DQN_CONFIG = {
    'name': 'DQN_Baseline_GTX750Ti',
    
    # Training parameters
    'total_timesteps': 500000,        # 500K steps (reasonable for initial baseline)
    'eval_freq': 10000,               # Evaluate every 10K steps
    'save_freq': 50000,               # Save checkpoint every 50K steps
    
    # DQN-specific hyperparameters
    'learning_rate': 1e-4,            # Standard learning rate
    'buffer_size': 50000,             # Reduced from 100K to save memory
    'learning_starts': 5000,          # Start learning after 5K steps
    'batch_size': 32,                 # Conservative batch size for 2GB VRAM
    'gamma': 0.99,                    # Discount factor
    'target_update_interval': 5000,   # Update target network every 5K steps
    'train_freq': 4,                  # Train every 4 steps
    'gradient_steps': 1,              # 1 gradient step per update
    
    # Exploration parameters
    'exploration_fraction': 0.15,     # Explore for 15% of training
    'exploration_initial_eps': 1.0,   # Start with full exploration
    'exploration_final_eps': 0.05,    # End with 5% exploration
    
    # Environment configuration
    'preprocess_type': 'none',        # Start with no preprocessing
    'reward_shaping': 'none',         # No reward shaping for baseline
    
    # Hardware
    'device': 'cuda',                 # Use GPU
    'seed': 42,
}

# ============================================================================
# PPO CONFIGURATION (Algorithm NOT from course - example)
# ============================================================================
PPO_CONFIG = {
    'name': 'PPO_Baseline_GTX750Ti',
    
    # Training parameters
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    
    # PPO-specific hyperparameters
    'learning_rate': 3e-4,
    'n_steps': 512,                   # Reduced from 2048 for memory
    'batch_size': 64,                 # Mini-batch size
    'n_epochs': 4,                    # Reduced from 10 for faster training
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,                 # Entropy coefficient for exploration
    'vf_coef': 0.5,                   # Value function coefficient
    'max_grad_norm': 0.5,
    
    # Environment configuration
    'preprocess_type': 'none',
    'reward_shaping': 'none',
    
    # Hardware
    'device': 'cuda',
    'seed': 42,
}

# ============================================================================
# IMPROVEMENT CONFIGURATIONS
# ============================================================================

# Improvement 1: Observation Preprocessing + Better CNN

# DQN_IMPROVEMENT_1 = {
#     **DQN_CONFIG,
#     'name': 'DQN_Improvement1_Normalize_CNN',
#     'preprocess_type': 'normalize',   # Normalize observations to [0, 1]
#     'learning_rate': 2.5e-4,          # Slightly higher LR with normalization
#     'batch_size': 64,                 # Larger batches for stability
#     'buffer_size': 100000,            # Larger buffer
#     'learning_starts': 10000,         # More exploration before training
#     'exploration_fraction': 0.2,      # Longer exploration period
#     'exploration_final_eps': 0.02,    # Lower final epsilon
#     'use_custom_cnn': True,           # Use improved CNN architecture
#     'use_frame_stacking': False,       # Stack 3 frames for temporal info
# }

DQN_IMPROVEMENT_1 = {
    'name': 'DQN_Improvement1_Normalize_CNN_Safe',
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    
    # Reduced for memory safety
    'batch_size': 32,                 # Reduced from 64
    'buffer_size': 75000,             # Reduced from 100K
    'learning_starts': 10000,
    
    'learning_rate': 2.5e-4,
    'gamma': 0.99,
    'target_update_interval': 5000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    'exploration_fraction': 0.2,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.02,
    
    # Improvements
    'preprocess_type': 'normalize',
    'reward_shaping': 'none',
    'use_custom_cnn': True,
    'use_frame_stacking': False,      # Disabled for memory
    
    'device': 'cuda',
    'seed': 42,
}

# Improvement 2: Reward Shaping
DQN_IMPROVEMENT_2 = {
    **DQN_IMPROVEMENT_1,
    'name': 'DQN_Improvement2_RewardShaping',
    'reward_shaping': 'achievement_bonus',  # Bonus for achievements
    'learning_rate': 1e-4,            # Back to standard LR
}

# ============================================================================
# MEMORY MONITORING SETTINGS
# ============================================================================
MEMORY_SETTINGS = {
    'log_gpu_memory': True,           # Log GPU memory usage
    'clear_cache_freq': 10000,        # Clear CUDA cache every 10K steps
    'memory_warning_threshold': 0.9,  # Warn if GPU memory > 90%
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================
EVAL_SETTINGS = {
    'n_eval_episodes': 10,            # Number of episodes for evaluation
    'deterministic': True,            # Use deterministic policy
    'render': False,                  # Don't render (not supported anyway)
    'track_achievements': True,       # Track achievement unlocking
}

# ============================================================================
# QUICK TEST CONFIGURATION (for debugging)
# ============================================================================
QUICK_TEST_CONFIG = {
    **DQN_CONFIG,
    'name': 'DQN_QuickTest',
    'total_timesteps': 10000,         # Just 10K steps for testing
    'eval_freq': 2000,                # Evaluate more frequently
    'buffer_size': 10000,             # Smaller buffer
    'learning_starts': 1000,          # Start learning quickly
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config(config_name='DQN_CONFIG'):
    """
    Get a configuration by name
    
    Args:
        config_name: Name of the config (e.g., 'DQN_CONFIG', 'PPO_CONFIG')
    
    Returns:
        Configuration dictionary
    """
    configs = {
        'DQN_CONFIG': DQN_CONFIG,
        'PPO_CONFIG': PPO_CONFIG,
        'DQN_IMPROVEMENT_1': DQN_IMPROVEMENT_1,
        'DQN_IMPROVEMENT_2': DQN_IMPROVEMENT_2,
        'QUICK_TEST_CONFIG': QUICK_TEST_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name].copy()


def print_config(config):
    """Print configuration in a readable format"""
    print("\n" + "="*70)
    print(f"CONFIGURATION: {config.get('name', 'Unnamed')}")
    print("="*70)
    
    # Group settings
    groups = {
        'Training': ['total_timesteps', 'eval_freq', 'save_freq', 'learning_rate'],
        'Memory': ['buffer_size', 'batch_size', 'learning_starts'],
        'Exploration': ['exploration_fraction', 'exploration_initial_eps', 'exploration_final_eps'],
        'Environment': ['preprocess_type', 'reward_shaping'],
        'Hardware': ['device', 'seed'],
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in config:
                print(f"  {key:30s}: {config[key]}")
    
    # Print remaining keys
    printed_keys = set()
    for keys in groups.values():
        printed_keys.update(keys)
    printed_keys.add('name')
    
    remaining = {k: v for k, v in config.items() if k not in printed_keys}
    if remaining:
        print("\nOther Parameters:")
        for key, value in remaining.items():
            print(f"  {key:30s}: {value}")
    
    print("="*70 + "\n")


def estimate_memory_usage(config):
    """
    Estimate GPU memory usage for a given configuration
    
    Returns:
        Estimated memory in GB
    """
    # Rough estimates for Crafter (64x64x3 images)
    obs_size_mb = 64 * 64 * 3 / 1024 / 1024  # Size of one observation in MB
    
    # Replay buffer
    buffer_memory = config['buffer_size'] * obs_size_mb * 2  # Current + next obs
    
    # Batch processing
    batch_memory = config['batch_size'] * obs_size_mb * 4  # Overhead for processing
    
    # Model parameters (rough estimate for CNN)
    model_memory = 50  # MB for a typical CNN
    
    # Total in GB
    total_mb = buffer_memory + batch_memory + model_memory
    total_gb = total_mb / 1024
    
    return total_gb


if __name__ == "__main__":
    # Print all configurations
    configs_to_show = ['DQN_CONFIG', 'DQN_IMPROVEMENT_1', 'DQN_IMPROVEMENT_2']
    
    for config_name in configs_to_show:
        config = get_config(config_name)
        print_config(config)
        
        # Estimate memory
        est_memory = estimate_memory_usage(config)
        print(f"Estimated GPU Memory Usage: {est_memory:.2f} GB")
        
        if est_memory > 1.8:  # GTX 750 Ti has 2GB
            print("⚠ WARNING: This may be tight on GTX 750 Ti memory!")
        else:
            print("✓ Should fit comfortably on GTX 750 Ti")
        print()