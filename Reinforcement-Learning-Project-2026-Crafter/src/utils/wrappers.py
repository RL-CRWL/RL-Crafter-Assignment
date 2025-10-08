import crafter
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
            print(f"Agent position BEFORE action: {old_position}")
        elif hasattr(self.env, '_player'):
            old_position = getattr(self.env._player, 'pos', 'Unknown')
            print(f"Agent position BEFORE action: {old_position}")
        else:
            print("Cannot access agent position - checking observation")
        
        # Take the actual step in the underlying Crafter environment
        obs, reward, done, info = self.env.step(action)
        
        # DEBUG: Check position after the action
        if hasattr(self.env, 'player_pos'):
            new_position = getattr(self.env, 'player_pos', 'Unknown')
            print(f"Agent position AFTER action: {new_position}")
        elif hasattr(self.env, '_player'):
            new_position = getattr(self.env._player, 'pos', 'Unknown')
            print(f"Agent position AFTER action: {new_position}")
        
        # DEBUG: Print reward and other info
        print(f"Reward received: {reward}")
        print(f"Episode done: {done}")
        print("---")  # Separator for readability
        
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
