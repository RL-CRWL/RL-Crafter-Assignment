"""
Updated configuration file with Improvement 2 settings
"""

# ============================================================================
# DQN BASELINE CONFIGURATION
# ============================================================================
DQN_CONFIG = {
    'name': 'DQN_Baseline_GTX750Ti',
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    
    'learning_rate': 1e-4,
    'buffer_size': 50000,
    'learning_starts': 5000,
    'batch_size': 32,
    'gamma': 0.99,
    'target_update_interval': 5000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    'exploration_fraction': 0.15,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    
    'preprocess_type': 'none',
    'reward_shaping': 'none',
    
    'device': 'cuda',
    'seed': 42,
}

# ============================================================================
# IMPROVEMENT 1: Normalized Observations + Custom CNN
# ============================================================================
DQN_IMPROVEMENT_1 = {
    'name': 'DQN_Improvement1_Normalize_CNN',
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    
    'batch_size': 32,
    'buffer_size': 75000,
    'learning_starts': 10000,
    
    'learning_rate': 2.5e-4,  # Higher LR with normalization
    'gamma': 0.99,
    'target_update_interval': 5000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    'exploration_fraction': 0.2,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.02,
    
    'preprocess_type': 'normalize',
    'reward_shaping': 'none',
    'use_custom_cnn': True,
    'use_frame_stacking': False,
    
    'device': 'cuda',
    'seed': 42,
}

# ============================================================================
# IMPROVEMENT 2: Stable CNN + Strategic Reward Shaping
# ============================================================================
DQN_IMPROVEMENT_2 = {
    'name': 'DQN_Improvement2_Stable_RewardShaping',
    'total_timesteps': 500000,
    'eval_freq': 10000,
    'save_freq': 50000,
    
    # Memory-safe settings
    'batch_size': 32,
    'buffer_size': 75000,
    'learning_starts': 10000,
    
    # Back to stable learning rate
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'target_update_interval': 5000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    # Extended exploration
    'exploration_fraction': 0.25,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.01,
    
    # Key improvements
    'preprocess_type': 'normalize',
    'reward_shaping': 'strategic',  # New strategic shaping
    'use_custom_cnn': True,
    'use_stable_cnn': True,  # Stable CNN (no batch norm)
    'use_frame_stacking': False,
    
    'device': 'cuda',
    'seed': 42,
}

# ============================================================================
# QUICK TEST CONFIGURATION
# ============================================================================
QUICK_TEST_CONFIG = {
    'name': 'DQN_QuickTest',
    'total_timesteps': 10000,
    'eval_freq': 2000,
    'buffer_size': 10000,
    'learning_starts': 1000,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'gamma': 0.99,
    'target_update_interval': 1000,
    'train_freq': 4,
    'gradient_steps': 1,
    'exploration_fraction': 0.15,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    'preprocess_type': 'none',
    'reward_shaping': 'none',
    'device': 'cuda',
    'seed': 42,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config(config_name='DQN_CONFIG'):
    """Get configuration by name"""
    configs = {
        'DQN_CONFIG': DQN_CONFIG,
        'DQN_IMPROVEMENT_1': DQN_IMPROVEMENT_1,
        'DQN_IMPROVEMENT_2': DQN_IMPROVEMENT_2,
        'QUICK_TEST_CONFIG': QUICK_TEST_CONFIG,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name].copy()


def print_config(config):
    """Print configuration"""
    print("\n" + "="*70)
    print(f"CONFIGURATION: {config.get('name', 'Unnamed')}")
    print("="*70)
    
    groups = {
        'Training': ['total_timesteps', 'eval_freq', 'learning_rate'],
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
    """Estimate GPU memory usage"""
    obs_size_mb = 64 * 64 * 3 / 1024 / 1024
    buffer_memory = config['buffer_size'] * obs_size_mb * 2
    batch_memory = config['batch_size'] * obs_size_mb * 4
    model_memory = 50
    total_mb = buffer_memory + batch_memory + model_memory
    total_gb = total_mb / 1024
    return total_gb