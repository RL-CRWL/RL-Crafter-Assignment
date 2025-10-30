"""
Crafter Environment Wrapper for Gymnasium Compatibility.
Provides observation preprocessing and reward shaping utilities.
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
        self.observation_space = spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
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
        obs, reward, done, info = self._env.step(action) # Old Gym
        self._last_obs = obs
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        # Crafter's render() doesn't take mode argument
        # It always returns RGB array
        if mode != 'rgb_array':
            raise ValueError(f"Unsupported render mode: {mode}. Only 'rgb_array' is supported.")
        return self._env.render()

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()

class ObservationPreprocessingWrapper(gym.ObservationWrapper):
    """
    Preprocesses observations from Crafter.
    Options: none, grayscale, downsample, normalize
    """
    def __init__(self, env, preprocess_type='none'):
        super().__init__(env)
        self.preprocess_type = preprocess_type
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
            gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
            gray = gray.astype(np.float32) / 255.0
            return gray[..., np.newaxis]
        elif self.preprocess_type == 'downsample':
            return obs[::2, ::2, :]
        elif self.preprocess_type == 'normalize':
            return obs.astype(np.float32) / 255.0
        else:
            warnings.warn(f"Unknown preprocessing type: {self.preprocess_type}")
            return obs.transpose(2, 0, 1)

class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper for reward shaping in Crafter.
    Options: none, scale, achievement_bonus
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
                        shaped_reward += 10.0 # Bonus for new achievement
            self.last_info = info
        return obs, shaped_reward, terminated, truncated, info

class ActionPreprocessingWrapper(gym.ActionWrapper):
    """
    Optionally modify actions before sending to environment.
    Currently: no preprocessing.
    """
    def __init__(self, env, action_type='none'):
        super().__init__(env)
        self.action_type = action_type

    def action(self, act):
        return act

def make_crafter_env(seed=None, size=(64, 64), preprocess_type='none', reward_shaping='none', action_type='none'):
    """
    Creates the fully wrapped Crafter environment.
    Set wrappers using arguments.
    Returns environment compatible with Gymnasium and Stable Baselines3.
    """
    env = CrafterGymnasiumWrapper(size=size)
    if preprocess_type != 'none':
        env = ObservationPreprocessingWrapper(env, preprocess_type=preprocess_type)
    if reward_shaping != 'none':
        env = RewardShapingWrapper(env, shaping_type=reward_shaping)
    if action_type != 'none':
        env = ActionPreprocessingWrapper(env, action_type=action_type)
    if seed is not None:
        env.reset(seed=seed)
    return env
