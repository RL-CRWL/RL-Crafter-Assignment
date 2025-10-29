import gymnasium as gym
import stable_baselines3
import argparse
import crafter
import os

from shimmy import GymV21CompatibilityV0
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from reward_shaping import CrafterRewardShaping
from networks import LSTMCNN

# ========== ARGUMENTS ==========
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='logdir/crafter_dqn/')
parser.add_argument('--steps', type=int, default=1_000_000)
args = parser.parse_args()

os.makedirs(args.logdir, exist_ok=True)

# ========== ENV ==========
def wrap_env():
    env = crafter.Env(reward=True)
    env = crafter.Recorder(env, args.logdir, save_stats=True, save_video=False, save_episode=False)
    env = GymV21CompatibilityV0(env=env)
    # improvement 1
    env = CrafterRewardShaping(env)

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
    save_freq=50_000, save_path=args.logdir, name_prefix="dqn_checkpoint"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=args.logdir,
    log_path=args.logdir,
    eval_freq=20_000,
    deterministic=True,
    render=False,
)

# ========== MODEL ==========
model_num = 2
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
    # policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=os.path.join(args.logdir, "tensorboard/"),
)

# ========== TRAIN ==========
log_interval = 100
model.learn(
    total_timesteps=args.steps,
    progress_bar=True,
    log_interval=log_interval,
    callback=[checkpoint_callback, eval_callback],
)

# ========== SAVE ==========
model.save(os.path.join(args.logdir, f"saved/DQN_{model_num}/dqn_final_model"))
print("Training complete! Model saved to:", args.logdir)

# ========== LOAD ==========
# model = DQN.load(os.path.join(args.logdir, f"saved/DQN_{model_num}/dqn_final_model"))

# ========== TEST OBSERVATIONS ==========
def test_observations(model, env, num_episodes=10, max_steps=20):
    all_episode_achievements = []
    
    for episode in range(num_episodes):
        print(f"\n=== Testing Observations - Episode {episode + 1} ===")
        obs = env.reset()
        done = [False]
        step_count = 0
        achievement_totals = {}
        
        while not done[0] and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # print(f"Step {step_count}:")
            # print(f"  Observation shape: {obs.shape}")
            # print(f"  Observation dtype: {obs.dtype}")
            # print(f"  Min/Max: [{obs.min():.3f}, {obs.max():.3f}]")
            # print(f"  Mean/Std: [{obs.mean():.3f}, {obs.std():.3f}]")
            # print(f"  Reward: {reward}")
            # print(f"  Done: {done}")
            
            # Track and print achievement info if available
            if info and 'achievements' in info[0]:
                achievements = info[0]['achievements']
                # print(f"  Achievements: {achievements}")
                for achievement, count in achievements.items():
                    achievement_totals[achievement] = achievement_totals.get(achievement, 0) + count
            
            step_count += 1
        
        all_episode_achievements.append(achievement_totals)
        print(f"\n=== Episode {episode + 1} Achievement Totals ===")
        for achievement, total in sorted(achievement_totals.items()):
            print(f"  {achievement}: {total}")
        print(f"=== Episode {episode + 1} ended after {step_count} steps ===\n")
    
    # Calculate and print averages
    print(f"\n=== Average Achievements over {num_episodes} Episodes ===")
    all_achievements = set()
    for ep_achievements in all_episode_achievements:
        all_achievements.update(ep_achievements.keys())
    
    for achievement in sorted(all_achievements):
        total = sum(ep.get(achievement, 0) for ep in all_episode_achievements)
        average = total / num_episodes
        print(f"  {achievement}: {average:.2f}")

# Test observations after training
print("\nTesting observations with trained model...")
test_observations(model, eval_env)