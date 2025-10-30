"""
<<<<<<< HEAD
Crafter Environment Wrapper for Gymnasium Compatibility.
Provides observation preprocessing and reward shaping utilities.
=======
Environment wrapper for Crafter - FIXED dimension handling

CRITICAL FIX: Properly transpose observations for PyTorch CNNs
PyTorch expects: (channels, height, width) = (3, 64, 64)
Crafter returns: (height, width, channels) = (64, 64, 3)
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
"""

import crafter
import gymnasium as gym
import numpy as np
import warnings
from gymnasium import spaces

class CrafterGymnasiumWrapper(gym.Env):
    """
    Wrapper to make Crafter compatible with Gymnasium API.
    """

    def __init__(self, size=(64, 64)):
        super().__init__()
        self._env = crafter.Env(size=size)
<<<<<<< HEAD
        self.observation_space = spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
=======
        
        # Define spaces based on Crafter's native spaces
        # Crafter returns (H, W, C) format
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )
        
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
        self.action_space = spaces.Discrete(17)
        self._last_obs = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        obs = self._env.reset()
        self._last_obs = obs
<<<<<<< HEAD
=======
        
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
        info = {}
        return obs, info

    def step(self, action):
<<<<<<< HEAD
        obs, reward, done, info = self._env.step(action) # Old Gym
        self._last_obs = obs
=======
        obs, reward, done, info = self._env.step(action)
        
        self._last_obs = obs
        
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info
<<<<<<< HEAD

    def render(self, mode='rgb_array'):
        # Crafter's render() doesn't take mode argument
        # It always returns RGB array
        if mode != 'rgb_array':
            raise ValueError(f"Unsupported render mode: {mode}. Only 'rgb_array' is supported.")
        return self._env.render()

=======
    
    def render(self):
        return self._last_obs
    
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()

class ObservationPreprocessingWrapper(gym.ObservationWrapper):
    """
    Preprocesses observations from Crafter.
<<<<<<< HEAD
    Options: none, grayscale, downsample, normalize
=======
    
    CRITICAL: Handles channel-first conversion for PyTorch!
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
    """
    def __init__(self, env, preprocess_type='none'):
        super().__init__(env)
        self.preprocess_type = preprocess_type
<<<<<<< HEAD
        if preprocess_type == 'grayscale':
            self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 1), dtype=np.float32)
        elif preprocess_type == 'downsample':
            self.observation_space = spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
        elif preprocess_type == 'normalize':
            self.observation_space = spaces.Box(low=0, high=1, shape=(64, 64, 3), dtype=np.float32)
        # else: no change

    def observation(self, obs):
        if self.preprocess_type == 'none':
            return obs
        elif self.preprocess_type == 'grayscale':
=======
        
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
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
            gray = gray.astype(np.float32) / 255.0
<<<<<<< HEAD
            return gray[..., np.newaxis]
        elif self.preprocess_type == 'downsample':
            return obs[::2, ::2, :]
        elif self.preprocess_type == 'normalize':
            return obs.astype(np.float32) / 255.0
=======
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
        
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
        else:
            warnings.warn(f"Unknown preprocessing type: {self.preprocess_type}")
            return obs.transpose(2, 0, 1)

class RewardShapingWrapper(gym.Wrapper):
    """
<<<<<<< HEAD
    Wrapper for reward shaping in Crafter.
    Options: none, scale, achievement_bonus
=======
    Applies reward shaping to the environment.
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
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
<<<<<<< HEAD
                for achievement, unlocked in current_achievements.items():
                    if unlocked and not last_achievements.get(achievement, False):
                        shaped_reward += 10.0 # Bonus for new achievement
            self.last_info = info
=======
                
                for achievement, unlocked in current_achievements.items():
                    if unlocked and not last_achievements.get(achievement, False):
                        shaped_reward += 10.0
        
        self.last_info = info
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
        return obs, shaped_reward, terminated, truncated, info

class ActionPreprocessingWrapper(gym.ActionWrapper):
    """
<<<<<<< HEAD
    Optionally modify actions before sending to environment.
    Currently: no preprocessing.
=======
    Preprocesses or modifies the action space.
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
    """
    def __init__(self, env, action_type='none'):
        super().__init__(env)
        self.action_type = action_type

    def action(self, act):
        return act

def make_crafter_env(seed=None, size=(64, 64), preprocess_type='none', reward_shaping='none', action_type='none'):
    """
<<<<<<< HEAD
    Creates the fully wrapped Crafter environment.
    Set wrappers using arguments.
    Returns environment compatible with Gymnasium and Stable Baselines3.
    """
    env = CrafterGymnasiumWrapper(size=size)
    if preprocess_type != 'none':
        env = ObservationPreprocessingWrapper(env, preprocess_type=preprocess_type)
=======
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
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
    if reward_shaping != 'none':
        env = RewardShapingWrapper(env, shaping_type=reward_shaping)
    if action_type != 'none':
        env = ActionPreprocessingWrapper(env, action_type=action_type)
    if seed is not None:
        env.reset(seed=seed)
    return env
<<<<<<< HEAD
=======


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
>>>>>>> d809efa4efe4d57c547cead309f3aab075872923
