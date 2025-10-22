import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from networks import ModifiedCNN
import argparse
import crafter
from shimmy import GymV21CompatibilityV0

parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='logdir/crafter_reward-ppo/0')
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--algorithm', default='ppo', choices=['ppo', 'a2c', 'dqn'])
parser.add_argument('--record-video', action='store_true', help='Record videos during training')
parser.add_argument('--video-interval', type=int, default=100, help='Record every N episodes')
args = parser.parse_args()

def make_env():
    # create the crafter environment (old gym)
    crafter_env = crafter.Env(reward=True)
    
    # wrap with recorder
    crafter_env = crafter.Recorder(
        crafter_env, 
        args.outdir,
        save_stats=True,
        save_video=True,
        save_episode=False,
    )
    
    # converts old gym to gymnasium using Shimmy
    env = GymV21CompatibilityV0(env=crafter_env)
    env = Monitor(env, filename=f"{args.outdir}/monitor")
    
    return env

# wrap in vectorised environment for stable baselines
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

policy_kwargs = dict(
    features_extractor_class=ModifiedCNN,
    features_extractor_kwargs=dict(features_dim=512)
)


# possible models
if args.algorithm == 'ppo':
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"{args.outdir}/tensorboard/"
    )
elif args.algorithm == 'a2c':
    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
        tensorboard_log=f"{args.outdir}/tensorboard/",
    )
elif args.algorithm == 'dqn':
        model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard/dqn/",
    )



# training
print(f"\n{'='*60}")
print(f"Training {args.algorithm.upper()} for {args.steps:,} steps...")
print(f"Output directory: {args.outdir}")
print(f"{'='*60}\n")

# how frequently we see we see the results
log_interval = 10
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=args.outdir, name_prefix='crafter_model')
model.learn(
    total_timesteps=args.steps,
    progress_bar=True,
    log_interval=log_interval,
    callback=checkpoint_callback
)

# prints bars for nicer visualisation
print(f"\n{'='*60}")
print(f"Training Complete")
print(f"{'='*60}")
model.save(f"{args.outdir}/{args.algorithm}_crafter")
print(f"Model saved to {args.outdir}/{args.algorithm}_crafter")
print(f"\nView results in TensorBoard:")
print(f"  tensorboard --logdir {args.outdir}/tensorboard/")
print(f"{'='*60}\n")