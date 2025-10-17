'''
PPO Training for Crafter using Stable-Baselines3
Fixed: Convert gym.spaces to gymnasium.spaces
'''

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SB3VecEnvWrapper(gym.Wrapper):
    """
    Wrapper that ensures gym -> gymnasium compatibility for SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # CRITICAL FIX: Convert old gym.spaces to gymnasium.spaces
        # Check if action_space is from old gym module
        if hasattr(env.action_space, 'n') and type(env.action_space).__module__.startswith('gym.'):
            # It's a gym.spaces.Discrete, convert to gymnasium.spaces.Discrete
            self.action_space = spaces.Discrete(env.action_space.n)
        else:
            self.action_space = env.action_space
            
        # Also convert observation space if needed
        if type(env.observation_space).__module__.startswith('gym.'):
            if hasattr(env.observation_space, 'shape'):
                # It's a Box space
                self.observation_space = spaces.Box(
                    low=env.observation_space.low,
                    high=env.observation_space.high,
                    shape=env.observation_space.shape,
                    dtype=env.observation_space.dtype
                )
        else:
            self.observation_space = env.observation_space
    
    def reset(self, **kwargs):
        """Return (observation, info) tuple for gymnasium compatibility."""
        # Remove 'seed' from kwargs if present and handle separately
        # Some older envs don't support the seed parameter
        seed = kwargs.pop('seed', None)
        
        # Try to reset with remaining kwargs
        try:
            result = self.env.reset(**kwargs)
        except TypeError:
            # If kwargs not supported, try without them
            result = self.env.reset()
        
        # Handle seed separately if provided
        if seed is not None and hasattr(self.env, 'seed'):
            self.env.seed(seed)
        
        # Ensure we return (obs, info) tuple
        if not isinstance(result, tuple):
            # Old format: just observation
            return result, {}
        elif len(result) == 2:
            # Correct format: (obs, info)
            return result
        else:
            # Unexpected format - take first element as obs
            print(f"Warning: reset() returned {len(result)} values, expected 2")
            return result[0], {}
    
    def step(self, action):
        """Handle action conversion and ensure gymnasium format (5-tuple)."""
        if isinstance(action, np.ndarray):
            action = int(action.item())
        
        result = self.env.step(action)
        
        # Gymnasium expects: (obs, reward, terminated, truncated, info)
        # Old gym returns: (obs, reward, done, info)
        if len(result) == 4:
            # Old gym format - convert to gymnasium
            obs, reward, done, info = result
            return obs, reward, done, False, info  # terminated=done, truncated=False
        else:
            # Already gymnasium format (5-tuple)
            return result


def make_crafter_env_sb3():
    """Create SB3-compatible Crafter environment."""
    from src.utils.wrappers import make_crafter_env
    
    env = make_crafter_env()
    env = SB3VecEnvWrapper(env)
    
    return env


class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN feature extractor."""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
        )
        
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class TrainingCallback(BaseCallback):
    """Callback for monitoring training."""
    def __init__(self, check_freq: int = 10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                print(f"\n{'='*60}")
                print(f"Timestep: {self.num_timesteps:,}")
                print(f"Mean Reward (last 100 eps): {mean_reward:.2f}")
                print(f"Mean Episode Length: {mean_length:.0f}")
                print(f"{'='*60}")
        return True


if __name__ == "__main__":
    print("="*60)
    print("PPO Training on Crafter with Stable-Baselines3")
    print("="*60)
    
    # Test environment
    print("\nTesting single environment...")
    test_env = make_crafter_env_sb3()
    
    print(f"Single env action space: {test_env.action_space}")
    print(f"  Type: {type(test_env.action_space)}")
    print(f"  Module: {type(test_env.action_space).__module__}")
    print(f"  Is gymnasium.spaces.Discrete: {isinstance(test_env.action_space, spaces.Discrete)}")
    
    obs, info = test_env.reset()
    print(f"✓ Reset works: obs shape={obs.shape}, info={info}")
    
    action = test_env.action_space.sample()
    obs, reward, terminated, truncated, info = test_env.step(action)
    print(f"✓ Step works: obs shape={obs.shape}, reward={reward}")
    
    test_env.close()
    
    print("\n" + "="*60)
    print("Creating vectorized environment...")
    print("="*60)
    
    # Create vectorized environment
    env = DummyVecEnv([make_crafter_env_sb3])
    
    print(f"\nAfter DummyVecEnv:")
    print(f"  Action space: {env.action_space}")
    print(f"  Type: {type(env.action_space)}")
    print(f"  Module: {type(env.action_space).__module__}")
    print(f"  Is gymnasium.spaces.Discrete: {isinstance(env.action_space, spaces.Discrete)}")
    
    env = VecMonitor(env, "./logs/")
    
    print(f"\nAfter VecMonitor:")
    print(f"  Action space: {env.action_space}")
    print(f"  Type: {type(env.action_space)}")
    print(f"  Module: {type(env.action_space).__module__}")
    print(f"  Is gymnasium.spaces.Discrete: {isinstance(env.action_space, spaces.Discrete)}")
    
    print("\n✓ Vectorized environment created")
    
    # Test vectorized environment reset
    print("\nTesting vectorized environment reset...")
    vec_obs = env.reset()
    print(f"  Vec reset returned: {type(vec_obs)}, shape: {vec_obs.shape if hasattr(vec_obs, 'shape') else 'N/A'}")
    
    # Custom policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[128], vf=[128])
    )
    
    # Create PPO model
    print("\nCreating PPO model...")
    print(f"Env action_space for PPO: {env.action_space}")
    print(f"Env action_space type: {type(env.action_space)}")
    
    try:
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="./ppo_crafter_tensorboard/"
        )
        
        print("✓ Model created successfully!")
        print("="*60)
        
        # Create callback
        training_callback = TrainingCallback(check_freq=10000)
        
        # Train
        print("\nStarting training...")
        print("="*60)
        
        model.learn(
            total_timesteps=1_000_000,
            callback=training_callback,
            progress_bar=True
        )
        
        print("\n" + "="*60)
        print("Training complete! Saving model...")
        model.save("ppo_crafter_final")
        print("✓ Model saved as 'ppo_crafter_final.zip'")
        
    except Exception as e:
        print(f"\n✗ Error creating or training model: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
    
    env.close()
    print("\n" + "="*60)
    print("Done!")
    print("="*60)