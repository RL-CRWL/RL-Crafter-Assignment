import gymnasium as gym
from gymnasium.envs.registration import register
import crafter  # make sure the crafter package is imported

# Register the CrafterPartial-v1 environment
try:
    register(
        id="CrafterPartial-v1",
        entry_point="crafter.env:CrafterPartialEnv",  # adjust if the class is in a different module
    )
except Exception as e:
    # Ignore if already registered
    pass

# Now you can make the environment
env = gym.make("CrafterPartial-v1", render_mode="human")
