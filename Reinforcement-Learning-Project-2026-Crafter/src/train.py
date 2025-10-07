import gym as old_gym
import stable_baselines3
import argparse
import crafter
from shimmy import GymV21CompatibilityV0
from gym.envs.registration import register

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=float, default=5e5)
args = parser.parse_args()

register(id='CrafterNoReward-v1',entry_point=crafter.Env)

env = old_gym.make('CrafterNoReward-v1')  # Or CrafterNoReward-v1
env = crafter.Recorder(
  env, './path/to/logdir',
  save_stats=True,
  save_video=False,
  save_episode=False,
)
env = GymV21CompatibilityV0(env=env)  

