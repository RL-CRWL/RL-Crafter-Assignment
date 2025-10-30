"""
Crafter Egocentric Wrapper
Adapted from MiniGrid egocentric implementation for pixel-based observations
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CrafterEgocentricWrapper(gym.Wrapper):
    """
    Egocentric observation wrapper for Crafter environment.
    
    Transforms the 64x64 top-down RGB view into an agent-centered perspective:
    1. Rotates view so agent always faces "up" 
    2. Centers agent in the observation (translation invariance)
    3. Optionally crops to smaller view size around agent
    """
    
    def __init__(self, env, crop_size=None, debug=False):
        super().__init__(env)
        
        self.crop_size = crop_size
        self.debug = debug
        
        # Agent state tracking
        self.agent_pos = np.array([32, 32])
        self.agent_dir = 3  # 0=right, 1=down, 2=left, 3=up
        
        # Crafter action mapping
        self.action_to_direction = {
            1: 2,  # move_left -> facing left
            2: 0,  # move_right -> facing right
            3: 3,  # move_up -> facing up
            4: 1,  # move_down -> facing down
        }
        
        # Update observation space
        if crop_size:
            new_shape = (3, crop_size, crop_size)
        else:
            new_shape = env.observation_space.shape
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=new_shape,
            dtype=np.uint8
        )
        
        if self.debug:
            print(f"\n{'='*70}")
            print("CRAFTER EGOCENTRIC WRAPPER INITIALIZED")
            print(f"{'='*70}")
            print(f"Crop size: {crop_size if crop_size else 'None (full 64x64)'}")
            print(f"Output shape: {new_shape}")
            print(f"{'='*70}\n")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent_pos = np.array([32, 32])
        self.agent_dir = 3
        return self._transform_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if action in self.action_to_direction:
            self.agent_dir = self.action_to_direction[action]
        
        if 'player_pos' in info:
            self.agent_pos = np.array(info['player_pos'])
        
        if self.debug:
            print(f"Action: {action}, Dir: {self.agent_dir}, Pos: {self.agent_pos}")
        
        ego_obs = self._transform_observation(obs)
        return ego_obs, reward, terminated, truncated, info
    
    def _transform_observation(self, obs):
        obs_hwc = obs.transpose(1, 2, 0)
        k = (3 - self.agent_dir) % 4
        rotated = np.rot90(obs_hwc, k=k)
        
        agent_y, agent_x = self.agent_pos
        rotated_y, rotated_x = self._rotate_position(agent_y, agent_x, k, obs_hwc.shape[0])
        
        centered = self._recenter_image(rotated, rotated_y, rotated_x)
        
        if self.crop_size:
            centered = self._crop_around_center(centered, self.crop_size)
        
        return centered.transpose(2, 0, 1)
    
    def _rotate_position(self, y, x, k, size):
        if k == 0:
            return y, x
        elif k == 1:
            return size - 1 - x, y
        elif k == 2:
            return size - 1 - y, size - 1 - x
        elif k == 3:
            return x, size - 1 - y
    
    def _recenter_image(self, img, agent_y, agent_x):
        h, w = img.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        shift_y = center_y - agent_y
        shift_x = center_x - agent_x
        
        recentered = np.roll(img, shift_y, axis=0)
        recentered = np.roll(recentered, shift_x, axis=1)
        
        return recentered
    
    def _crop_around_center(self, img, crop_size):
        h, w, c = img.shape
        center_y, center_x = h // 2, w // 2
        
        half = crop_size // 2
        start_y = center_y - half
        end_y = center_y + half + (crop_size % 2)
        start_x = center_x - half
        end_x = center_x + half + (crop_size % 2)
        
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - h)
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - w)
        
        start_y = max(0, start_y)
        end_y = min(h, end_y)
        start_x = max(0, start_x)
        end_x = min(w, end_x)
        
        cropped = img[start_y:end_y, start_x:end_x]
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = np.pad(
                cropped,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        return cropped