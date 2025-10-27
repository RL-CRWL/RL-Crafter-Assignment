# Setup

# Baseline
Location: src/PPO_baseline.py
Running trains over 500 000 steps, and creates model and saves to results.

# Visualize model
To see GIFs, run src/evaluation/ppo_baseline_viz.py.

# Custom implementations
ActorCriticPPO is custom and ActorCritic is just normal with one step look ahead policy.

# Utilities
## Wrappers
Wrappers allow us to work with crafter environment
Serves as our own Gym

# Improvements
Avoid changing hyper parameters
Experiment with them in addition to changes elsewhere
Changing NN architecture (hyperparameter)
Increase previous state-actions considered
Try LSTM 
Reshape rewards
