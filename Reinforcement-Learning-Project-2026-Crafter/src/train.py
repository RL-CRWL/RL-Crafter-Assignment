import gymnasium as gym
import stable_baselines3
import argparse
import crafter
import os
import json

from shimmy import GymV21CompatibilityV0
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from evaluate import evaluate_model, print_evaluation_results
from networks import LSTMCNN

# ========== ARGUMENTS ==========

steps=1_000_000
MODEL_NUM = 2

logdir=f'Reinforcement-Learning-Project-2026-Crafter/logdir/crafter_dqn/'
recorder_dir = f"{logdir}/DQN_{MODEL_NUM}/recording/"
checkpoint_dir = f"{logdir}/DQN_{MODEL_NUM}/checkpoints/"
eval_dir = f"{logdir}/DQN_{MODEL_NUM}/callbacks/"
save_dir = f"DQN_{MODEL_NUM}/final_model/"

# ensure all directories exist
os.makedirs(logdir, exist_ok=True)
os.makedirs(recorder_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
os.makedirs(os.path.join(logdir, save_dir), exist_ok=True)

# ========== ENV ==========
def wrap_env():
    env = crafter.Env(reward=True)
    env = crafter.Recorder(env, recorder_dir, save_stats=True, save_video=False, save_episode=False)
    env = GymV21CompatibilityV0(env=env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    # improvement 2
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env

env = wrap_env()
eval_env = wrap_env()

# ========== CALLBACKS ==========
checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=checkpoint_dir, name_prefix="dqn_checkpoint"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=eval_dir,
    log_path=logdir,
    eval_freq=10_000,
    deterministic=True,
    render=False,
)

# ========== MODEL ==========
policy_kwargs = dict(
    features_extractor_class=LSTMCNN,
    features_extractor_kwargs=dict(features_dim=512),
)
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=10_000,
    batch_size=64,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=2_500,
    exploration_fraction=0.2,
    exploration_final_eps=0.02,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=os.path.join(logdir, "tensorboard/"),
)

# ========== TRAIN ==========
log_interval = 100
model.learn(
    total_timesteps=steps,
    progress_bar=True,
    log_interval=log_interval,
    callback=[checkpoint_callback, eval_callback],
)

# ========== SAVE ==========
model.save(os.path.join(logdir, save_dir))
print("Training complete! Model saved to:", logdir)

# ========== LOAD ==========
# model = DQN.load(os.path.join(logdir, save_dir))

# ========== EVALUATE ==========
print("\nRunning evaluation...")
metrics = evaluate_model(model, num_episodes=10)
print_evaluation_results(metrics, f"DQN_{MODEL_NUM}")
metrics_file = f"{logdir}/DQN_{MODEL_NUM}/metrics_DQN_{MODEL_NUM}.json"
with open(metrics_file, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_file}")

