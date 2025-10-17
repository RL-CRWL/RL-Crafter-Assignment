"""
Environment wrapper for Crafter to ensure Gymnasium compatibility
and handle API differences between Gym and Gymnasium versions.

This wrapper converts between different Gym/Gymnasium API versions
and provides utilities for environment creation and preprocessing.

DO NOT MODIFY: API compatibility logic (reset/step signatures)
YOU CAN MODIFY: Observation preprocessing, action space modifications, reward shaping
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import warnings
import crafter


class CrafterGymnasiumWrapper(gym.Env):
    """
    Wrapper to make Crafter compatible with Gymnasium API.
    This wraps the native crafter.Env() directly.
    """
    
    def __init__(self, size=(64, 64)):
        """
        Initialize wrapper around native Crafter environment
        
        Args:
            size: Observation size (default: (64, 64))
        """
        super().__init__()
        
        # Create native Crafter environment
        self._env = crafter.Env(size=size)
        
        # Define spaces based on Crafter's native spaces
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], 3),
            dtype=np.uint8
        )
        
        self.action_space = spaces.Discrete(17)
        
        self._last_obs = None
    
    def reset(self, seed=None, options=None):
        """
        Reset environment with Gymnasium API
        Returns: obs, info
        """
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        
        obs = self._env.reset()
        self._last_obs = obs
        
        # Crafter doesn't return info on reset, so we create empty dict
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        Step environment with Gymnasium API
        Returns: obs, reward, terminated, truncated, info
        """
        # Crafter's step returns (obs, reward, done, info) - old Gym API
        obs, reward, done, info = self._env.step(action)
        
        self._last_obs = obs
        
        # Convert old Gym API to Gymnasium API
        # Split 'done' into 'terminated' and 'truncated'
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        return self._last_obs
    
    def close(self):
        """Close the environment"""
        if hasattr(self._env, 'close'):
            self._env.close()


class ObservationPreprocessingWrapper(gym.ObservationWrapper):
    """
    Preprocesses observations from Crafter.
    
    YOU CAN MODIFY THIS for preprocessing techniques such as:
    - Grayscaling
    - Downsampling
    - Feature extraction
    - Normalization
    - Stacking frames
    
    Currently: No preprocessing (identity operation)
    """
    
    def __init__(self, env, preprocess_type='none'):
        """
        Initialize preprocessing wrapper
        
        Args:
            env: Environment to wrap
            preprocess_type: Type of preprocessing to apply
                - 'none': No preprocessing (default)
                - 'grayscale': Convert to grayscale and normalize
                - 'downsample': Downsample to 32x32
                - 'normalize': Normalize pixel values to [0, 1]
        """
        super().__init__(env)
        self.preprocess_type = preprocess_type
        
        # Modify observation space if needed
        if preprocess_type == 'grayscale':
            # Grayscale: 1 channel instead of 3
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(64, 64, 1),
                dtype=np.float32
            )
        elif preprocess_type == 'downsample':
            # Downsampled 32x32 RGB
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(32, 32, 3),
                dtype=np.uint8
            )
        elif preprocess_type == 'normalize':
            # Normalized RGB
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(64, 64, 3),
                dtype=np.float32
            )
        # else: 'none' - keep original observation space
    
    def observation(self, obs):
        """
        Apply preprocessing to observation
        
        Args:
            obs: Raw observation from environment (64x64 RGB)
        
        Returns:
            Preprocessed observation
        """
        if self.preprocess_type == 'none':
            return obs
        
        elif self.preprocess_type == 'grayscale':
            # Convert RGB to grayscale using standard weights
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
            # Normalize to [0, 1]
            gray = gray.astype(np.float32) / 255.0
            return gray[..., np.newaxis]
        
        elif self.preprocess_type == 'downsample':
            # Simple downsample using striding (64x64 -> 32x32)
            return obs[::2, ::2, :]
        
        elif self.preprocess_type == 'normalize':
            # Normalize pixel values to [0, 1]
            return obs.astype(np.float32) / 255.0
        
        else:
            warnings.warn(f"Unknown preprocessing type: {self.preprocess_type}")
            return obs


class RewardShapingWrapper(gym.Wrapper):
    """
    Applies reward shaping to the environment.
    
    YOU CAN MODIFY THIS for different reward shaping strategies such as:
    - Scaling survival rewards
    - Bonus rewards for achievements
    - Penalty for dying
    - Exploration bonuses
    
    Currently: No shaping (identity operation)
    """
    
    def __init__(self, env, shaping_type='none'):
        """
        Initialize reward shaping wrapper
        
        Args:
            env: Environment to wrap
            shaping_type: Type of reward shaping
                - 'none': No shaping (default)
                - 'scale': Scale all rewards by factor
                - 'achievement_bonus': Bonus for achievements
        """
        super().__init__(env)
        self.shaping_type = shaping_type
        self.last_info = {}
    
    def step(self, action):
        """Apply reward shaping and return modified reward"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        shaped_reward = reward
        
        if self.shaping_type == 'none':
            shaped_reward = reward
        
        elif self.shaping_type == 'scale':
            # Example: scale survival reward by factor of 2
            shaped_reward = reward * 2.0
        
        elif self.shaping_type == 'achievement_bonus':
            # Example: add bonus for achievement unlocks
            shaped_reward = reward
            if 'achievements' in info and 'achievements' in self.last_info:
                current_achievements = info.get('achievements', {})
                last_achievements = self.last_info.get('achievements', {})
                
                # Check for new achievements
                for achievement, unlocked in current_achievements.items():
                    if unlocked and not last_achievements.get(achievement, False):
                        shaped_reward += 10.0  # Bonus for new achievement
        
        self.last_info = info
        return obs, shaped_reward, terminated, truncated, info


class ActionPreprocessingWrapper(gym.ActionWrapper):
    """
    Preprocesses or modifies the action space.
    
    YOU CAN MODIFY THIS for action space modifications such as:
    - Discrete to continuous action mapping
    - Action filtering
    - Action repetition
    
    Currently: No action preprocessing (identity operation)
    """
    
    def __init__(self, env, action_type='none'):
        """
        Initialize action preprocessing wrapper
        
        Args:
            env: Environment to wrap
            action_type: Type of action processing
                - 'none': No processing (default)
        """
        super().__init__(env)
        self.action_type = action_type
    
    def action(self, act):
        """Process action before sending to environment"""
        return act


def make_crafter_env(seed=None,
                     size=(64, 64),
                     preprocess_type='none',
                     reward_shaping='none',
                     action_type='none'):
    """
    Factory function to create a properly wrapped Crafter environment.
    
    This function creates a Crafter environment using the native API
    and applies all necessary wrappers for Gymnasium compatibility.
    
    Args:
        seed: Random seed for reproducibility
        size: Observation size (default: (64, 64))
        preprocess_type: Observation preprocessing type
            - 'none': No preprocessing (default)
            - 'grayscale': Convert to grayscale
            - 'downsample': Downsample to 32x32
            - 'normalize': Normalize to [0, 1]
        reward_shaping: Reward shaping type
            - 'none': No shaping (default)
            - 'scale': Scale rewards
            - 'achievement_bonus': Bonus for achievements
        action_type: Action preprocessing type
            - 'none': No preprocessing (default)
    
    Returns:
        Wrapped Crafter environment ready for training with Stable Baselines3
    
    Example:
        env = make_crafter_env(preprocess_type='normalize')
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """
    
    # Create base environment using native Crafter API
    try:
        env = CrafterGymnasiumWrapper(size=size)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create Crafter environment. "
            f"Make sure crafter==1.8.0 is installed.\n"
            f"Error: {e}"
        )
    
    # Apply observation preprocessing
    if preprocess_type != 'none':
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


# Utility function for creating vectorized environments
def make_crafter_vec_env(n_envs=4, seed=None, size=(64, 64),
                         preprocess_type='none', reward_shaping='none'):
    """
    Create multiple parallel Crafter environments for vectorized training.
    
    Args:
        n_envs: Number of parallel environments
        seed: Base seed for reproducibility
        size: Observation size
        preprocess_type: Observation preprocessing type
        reward_shaping: Reward shaping type
    
    Returns:
        DummyVecEnv with n_envs parallel environments
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
    return envsimport crafter
import gymnasium as gym
from typing import Optional, Any, Tuple


class CrafterGymWrapper(gym.Env):
    """
    Wraps crafter.Env to be compatible with Gymnasium wrappers.

    Crafter uses the old Gym API (4-tuple step returns), so this wrapper
    converts it to the new Gymnasium API (5-tuple step returns).
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, seed: int = 0, **kwargs):
        """
        Initialize the Crafter environment wrapper.

        Args:
            seed: Random seed for environment initialization
            **kwargs: Additional arguments to pass to crafter.Env
        """
        super().__init__()
        self.env = crafter.Env(seed=seed, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self._seed = seed

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        """
        Reset the environment.

        Args:
            seed: Optional seed for resetting the environment
            options: Optional options dictionary (for Gymnasium compatibility)

        Returns:
            observation: The initial observation
            info: Additional information dictionary
        """
        # If a new seed is provided, recreate the environment
        if seed is not None and seed != self._seed:
            self._seed = seed
            self.env = crafter.Env(seed=seed)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space

        obs = self.env.reset()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: The action to take

        Returns:
            observation: The observation after taking the action
            reward: The reward received
            terminated: Whether the episode has ended (task completion)
            truncated: Whether the episode was truncated (time limit)
            info: Additional information dictionary
        """
        # DEBUG: Print the action being taken
        print(f"Action received: {action}")
        
        # Get the current agent position before taking the action (if available)
        if hasattr(self.env, 'player_pos'):
            old_position = getattr(self.env, 'player_pos', 'Unknown')
            #print(f"Agent position BEFORE action: {old_position}")
        elif hasattr(self.env, '_player'):
            old_position = getattr(self.env._player, 'pos', 'Unknown')
            #print(f"Agent position BEFORE action: {old_position}")
        else:
            print("Cannot access agent position - checking observation")
        
        # Take the actual step in the underlying Crafter environment
        obs, reward, done, info = self.env.step(action)
        
        # DEBUG: Check position after the action
        if hasattr(self.env, 'player_pos'):
            new_position = getattr(self.env, 'player_pos', 'Unknown')
            #print(f"Agent position AFTER action: {new_position}")
        elif hasattr(self.env, '_player'):
            new_position = getattr(self.env._player, 'pos', 'Unknown')
            #print(f"Agent position AFTER action: {new_position}")
        
        # DEBUG: Print reward and other info
        ''' print(f"Reward received: {reward}")
        print(f"Episode done: {done}")
        print("---")  # Separator for readability'''
        
        # Crafter returns done=True when the agent dies
        # Gymnasium separates this into 'terminated' and 'truncated'
        terminated = done
        truncated = False  # Crafter doesn't have explicit time limits

        return obs, reward, terminated, truncated, info


    def render(self, mode: str = 'rgb_array'):
        """
        Render the environment.

        Args:
            mode: Render mode ('rgb_array' or 'human')

        Returns:
            RGB array if mode='rgb_array', None otherwise
        """
        return self.env.render(mode=mode)

    def close(self):
        """Close the environment and clean up resources."""
        if hasattr(self.env, "close"):
            self.env.close()


def make_crafter_env(seed: int = 0, **kwargs) -> gym.Env:
    """
    Create a wrapped Crafter environment with episode statistics tracking.

    This function creates a CrafterGymWrapper and adds the RecordEpisodeStatistics
    wrapper to track episode returns and lengths.

    Args:
        seed: Random seed for environment initialization
        **kwargs: Additional arguments to pass to crafter.Env

    Returns:
        Wrapped Crafter environment compatible with Gymnasium

    Example:
        >>> env = make_crafter_env(seed=42)
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(0)
    """
    env = CrafterGymWrapper(seed=seed, **kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


# For backwards compatibility, you can also create the environment directly
def create_crafter_env(seed: int = 0, size: Tuple[int, int] = (64, 64), **kwargs) -> gym.Env:
    """
    Alternative function to create a Crafter environment with custom settings.

    Args:
        seed: Random seed
        size: Size of the observation window (width, height)
        **kwargs: Additional arguments

    Returns:
        Wrapped Crafter environment
    """
    return make_crafter_env(seed=seed, size=size, **kwargs)
