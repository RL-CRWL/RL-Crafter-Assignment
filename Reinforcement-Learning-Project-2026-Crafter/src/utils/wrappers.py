"""
Environment wrapper for Crafter - FIXED dimension handling

CRITICAL FIX: Properly transpose observations for PyTorch CNNs
PyTorch expects: (channels, height, width) = (3, 64, 64)
Crafter returns: (height, width, channels) = (64, 64, 3)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import warnings
import crafter


class CrafterGymnasiumWrapper(gym.Env):
    """
    Wrapper to make Crafter compatible with Gymnasium API.
    """
    
    def __init__(self, size=(64, 64)):
        super().__init__()
        
        # Create native Crafter environment
        self._env = crafter.Env(size=size)
        
        # Define spaces based on Crafter's native spaces
        # Crafter returns (H, W, C) format
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )
        
        self.action_space = spaces.Discrete(17)
        
        self._last_obs = None
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        
        obs = self._env.reset()
        self._last_obs = obs
        
        info = {}
        
        return obs, info
    
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        
        self._last_obs = obs
        
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self._last_obs
    
    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()


class ObservationPreprocessingWrapper(gym.ObservationWrapper):
    """
    Preprocesses observations from Crafter.
    
    CRITICAL: Handles channel-first conversion for PyTorch!
    """
    
    def __init__(self, env, preprocess_type='none'):
        super().__init__(env)
        self.preprocess_type = preprocess_type
        
        # Get original shape (H, W, C)
        orig_shape = env.observation_space.shape
        
        # Modify observation space based on preprocessing
        if preprocess_type == 'grayscale':
            # Grayscale: 1 channel, channel-first for PyTorch
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(1, orig_shape[0], orig_shape[1]),  # (C, H, W)
                dtype=np.float32
            )
        elif preprocess_type == 'downsample':
            # Downsampled 32x32 RGB, channel-first
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(3, 32, 32),  # (C, H, W)
                dtype=np.uint8
            )
        elif preprocess_type == 'normalize':
            # Normalized RGB, channel-first for PyTorch
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(3, orig_shape[0], orig_shape[1]),  # (C, H, W) = (3, 64, 64)
                dtype=np.float32
            )
        # else: 'none' - keep original but still convert to channel-first
        else:
            # Even for 'none', convert to channel-first for consistency
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(3, orig_shape[0], orig_shape[1]),  # (C, H, W)
                dtype=np.uint8
            )
    
    def observation(self, obs):
        """
        Apply preprocessing to observation
        
        CRITICAL: Always return (C, H, W) format for PyTorch!
        
        Args:
            obs: Raw observation from environment (H, W, C) format
        
        Returns:
            Preprocessed observation in (C, H, W) format
        """
        if self.preprocess_type == 'none':
            # Just transpose to channel-first
            return obs.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        elif self.preprocess_type == 'grayscale':
            # Convert RGB to grayscale
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
            # Normalize to [0, 1]
            gray = gray.astype(np.float32) / 255.0
            # Add channel dimension and return as (C, H, W)
            return gray[np.newaxis, :, :]  # (1, H, W)
        
        elif self.preprocess_type == 'downsample':
            # Simple downsample using striding (64x64 -> 32x32)
            downsampled = obs[::2, ::2, :]  # (32, 32, 3)
            # Transpose to channel-first
            return downsampled.transpose(2, 0, 1)  # (3, 32, 32)
        
        elif self.preprocess_type == 'normalize':
            # Normalize pixel values to [0, 1]
            normalized = obs.astype(np.float32) / 255.0  # (H, W, C)
            # Transpose to channel-first
            return normalized.transpose(2, 0, 1)  # (3, H, W) = (3, 64, 64)
        
        else:
            warnings.warn(f"Unknown preprocessing type: {self.preprocess_type}")
            return obs.transpose(2, 0, 1)


class RewardShapingWrapper(gym.Wrapper):
    """
    Applies reward shaping to the environment.
    """
    
    def __init__(self, env, shaping_type='none'):
        super().__init__(env)
        self.shaping_type = shaping_type
        self.last_info = {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        
        if self.shaping_type == 'none':
            shaped_reward = reward
        
        elif self.shaping_type == 'scale':
            shaped_reward = reward * 2.0
        
        elif self.shaping_type == 'achievement_bonus':
            shaped_reward = reward
            if 'achievements' in info and 'achievements' in self.last_info:
                current_achievements = info.get('achievements', {})
                last_achievements = self.last_info.get('achievements', {})
                
                for achievement, unlocked in current_achievements.items():
                    if unlocked and not last_achievements.get(achievement, False):
                        shaped_reward += 10.0
        
        self.last_info = info
        return obs, shaped_reward, terminated, truncated, info


class ActionPreprocessingWrapper(gym.ActionWrapper):
    """
    Preprocesses or modifies the action space.
    """
    
    def __init__(self, env, action_type='none'):
        super().__init__(env)
        self.action_type = action_type
    
    def action(self, act):
        return act


def make_crafter_env(seed=None,
                     size=(64, 64),
                     preprocess_type='none',
                     reward_shaping='none',
                     action_type='none'):
    """
    Factory function to create a properly wrapped Crafter environment.
    
    Args:
        seed: Random seed for reproducibility
        size: Observation size (default: (64, 64))
        preprocess_type: Observation preprocessing type
            - 'none': No preprocessing (still converts to channel-first)
            - 'grayscale': Convert to grayscale
            - 'downsample': Downsample to 32x32
            - 'normalize': Normalize to [0, 1]
        reward_shaping: Reward shaping type
        action_type: Action preprocessing type
    
    Returns:
        Wrapped Crafter environment ready for training
    """
    
    # Create base environment
    try:
        env = CrafterGymnasiumWrapper(size=size)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create Crafter environment. "
            f"Make sure crafter==1.8.0 is installed.\n"
            f"Error: {e}"
        )
    
    # Apply observation preprocessing (ALWAYS apply to get channel-first format)
    env = ObservationPreprocessingWrapper(env, preprocess_type=preprocess_type)
    
    # Apply reward shaping
    if reward_shaping != 'none':
        env = RewardShapingWrapper(env, shaping_type=reward_shaping)
    
    # Apply action preprocessing
    if action_type != 'none':
        env = ActionPreprocessingWrapper(env, action_type=action_type)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    return env


def make_crafter_vec_env(n_envs=4, seed=None, size=(64, 64),
                         preprocess_type='none', reward_shaping='none'):
    """
    Create multiple parallel Crafter environments for vectorized training.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    def make_env(idx):
        def _init():
            env = make_crafter_env(
                seed=seed + idx if seed is not None else None,
                size=size,
                preprocess_type=preprocess_type,
                reward_shaping=reward_shaping
            )
            return env
        return _init
    
    envs = DummyVecEnv([make_env(i) for i in range(n_envs)])
    return envs