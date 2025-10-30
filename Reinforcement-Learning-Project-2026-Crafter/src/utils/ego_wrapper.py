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
    
    This provides rotation and translation invariance, helping the agent
    learn spatial relationships independent of absolute position/orientation.
    
    Key differences from MiniGrid:
    - Works with RGB pixel observations instead of symbolic grid
    - Tracks agent direction from movement actions (no explicit direction)
    - Uses player_pos from info dict instead of grid encoding
    """
    
    def __init__(self, env, crop_size=None, debug=False):
        """
        Args:
            env: Crafter environment (must provide player_pos in info)
            crop_size: If provided, crop to (crop_size, crop_size) around agent.
                      If None, keeps full 64x64 but recentered and rotated.
                      Recommended: 32, 48, or None
            debug: If True, prints diagnostic information
        """
        super().__init__(env)
        
        self.crop_size = crop_size
        self.debug = debug
        
        # Agent state tracking
        self.agent_pos = np.array([32, 32])  # Default center
        self.agent_dir = 3  # 0=right, 1=down, 2=left, 3=up (start facing up)
        
        # Crafter action mapping
        self.ACTIONS = {
            'noop': 0,
            'move_left': 1,
            'move_right': 2, 
            'move_up': 3,
            'move_down': 4,
            # Actions 5+ are non-movement (do, sleep, place, make)
        }
        
        # Direction mapping: action -> direction change
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
            new_shape = env.observation_space.shape  # Keep (3, 64, 64)
        
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
        
        # Initialize agent state
        self.agent_pos = np.array([32, 32])  # Crafter starts at center
        self.agent_dir = 3  # Start facing up
        
        return self._transform_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update agent direction based on movement action
        if action in self.action_to_direction:
            self.agent_dir = self.action_to_direction[action]
        
        # Get agent position from info (Crafter provides this!)
        if 'player_pos' in info:
            self.agent_pos = np.array(info['player_pos'])
        
        if self.debug:
            print(f"Action: {action}, Dir: {self.agent_dir}, Pos: {self.agent_pos}")
        
        # Transform to egocentric view
        ego_obs = self._transform_observation(obs)
        
        return ego_obs, reward, terminated, truncated, info
    
    def _transform_observation(self, obs):
        """
        Apply egocentric transformation:
        1. Rotate so agent faces up (direction normalization)
        2. Center agent in view (position normalization)
        3. Optionally crop around agent (focus on local neighborhood)
        
        Args:
            obs: (3, 64, 64) CHW RGB observation
            
        Returns:
            Transformed observation in egocentric frame
        """
        # Convert CHW -> HWC for easier manipulation
        obs_hwc = obs.transpose(1, 2, 0)  # (64, 64, 3)
        
        # Step 1: Rotate based on agent direction
        # We want agent to always face "up" (direction 3)
        k = (3 - self.agent_dir) % 4  # Rotation amount (90° increments)
        rotated = np.rot90(obs_hwc, k=k)
        
        # Step 2: Rotate agent position to match
        agent_y, agent_x = self.agent_pos
        rotated_y, rotated_x = self._rotate_position(
            agent_y, agent_x, k, obs_hwc.shape[0]
        )
        
        # Step 3: Recenter so agent is at middle of view
        centered = self._recenter_image(rotated, rotated_y, rotated_x)
        
        # Step 4: Crop if requested
        if self.crop_size:
            centered = self._crop_around_center(centered, self.crop_size)
        
        # Convert back to CHW
        return centered.transpose(2, 0, 1)
    
    def _rotate_position(self, y, x, k, size):
        """
        Rotate a position (y, x) by k*90 degrees around image center.
        
        Args:
            y, x: Original position
            k: Number of 90° CCW rotations (0, 1, 2, or 3)
            size: Image size (assumed square)
            
        Returns:
            (new_y, new_x) after rotation
        """
        if k == 0:
            return y, x
        elif k == 1:  # 90° CCW
            return size - 1 - x, y
        elif k == 2:  # 180°
            return size - 1 - y, size - 1 - x
        elif k == 3:  # 270° CCW (90° CW)
            return x, size - 1 - y
        else:
            raise ValueError(f"Invalid rotation k={k}")
    
    def _recenter_image(self, img, agent_y, agent_x):
        """
        Recenter image so agent is at center position.
        Uses numpy roll for efficient wraparound shifting.
        
        Args:
            img: (H, W, C) image
            agent_y, agent_x: Current agent position
            
        Returns:
            Recentered image with agent at center
        """
        h, w = img.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Calculate required shifts
        shift_y = center_y - agent_y
        shift_x = center_x - agent_x
        
        # Roll along both axes (wraparound at boundaries)
        recentered = np.roll(img, shift_y, axis=0)
        recentered = np.roll(recentered, shift_x, axis=1)
        
        return recentered
    
    def _crop_around_center(self, img, crop_size):
        """
        Crop image around center point (where agent is located).
        Pads with zeros if crop extends beyond image boundaries.
        
        Args:
            img: (H, W, C) centered image
            crop_size: Target crop size
            
        Returns:
            (crop_size, crop_size, C) cropped image
        """
        h, w, c = img.shape
        center_y, center_x = h // 2, w // 2
        
        half = crop_size // 2
        
        # Calculate crop boundaries
        start_y = center_y - half
        end_y = center_y + half + (crop_size % 2)
        start_x = center_x - half
        end_x = center_x + half + (crop_size % 2)
        
        # Handle boundaries with zero padding
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - h)
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - w)
        
        # Clamp to valid range
        start_y = max(0, start_y)
        end_y = min(h, end_y)
        start_x = max(0, start_x)
        end_x = min(w, end_x)
        
        # Crop
        cropped = img[start_y:end_y, start_x:end_x]
        
        # Pad if necessary
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = np.pad(
                cropped,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        return cropped


# Convenience function for your wrappers.py
def make_crafter_egocentric_env(seed=42, preprocess_type='normalize', 
                                 crop_size=None, debug=False):
    """
    Create Crafter environment with egocentric observations.
    
    Args:
        seed: Random seed
        preprocess_type: 'normalize' or None (from your existing wrapper)
        crop_size: Size to crop around agent (None = full 64x64)
        debug: Print diagnostic info
        
    Returns:
        Wrapped environment with egocentric observations
    """
    from utils.wrappers import make_crafter_env
    
    # Create base environment with your existing preprocessing
    env = make_crafter_env(seed=seed, preprocess_type=preprocess_type)
    
    # Add egocentric wrapper
    env = CrafterEgocentricWrapper(env, crop_size=crop_size, debug=debug)
    
    return env