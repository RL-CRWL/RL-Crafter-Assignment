import gymnasium as gym
import stable_baselines3
import argparse
import crafter
from shimmy import GymV21CompatibilityV0

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
args = parser.parse_args()


raw_env = crafter.Env()

env = GymV21CompatibilityV0(env=raw_env)
env = crafter.Recorder(
    env, './path/to/logdir',
    save_stats=True,
    save_video=False,
    save_episode=False,
)
