conda env create -f env1.yml 


conda activate crafter_env

for windows:
python -m pip list | findstr crafter
python -m pip list | findstr gymnasium
python -m pip list | findstr stable-baselines3

for linux:
python -m pip list | grep crafter
python -m pip list | grep gymnasium
python -m pip list | grep stable_baselines3


expected output:

crafter              1.7.1
gymnasium            0.29.1
stable-baselines3    2.3.2


run test_crafter.py to check if everything got installed



### Training
python scripts/train.py --config DQN_CONFIG 


###  Evaluate Each Agent (REQUIRED FOR REPORT)
powershell# Evaluate baseline
python src/evaluation/evaluate_agent.py --model results/dqn_baseline_*/dqn_baseline_model.zip --n_episodes 50 --save_dir results/evaluation/baseline

# Evaluate improvement 1
python src/evaluation/evaluate_agent.py --model results/dqn_improvement1_*/dqn_baseline_model.zip --n_episodes 50 --save_dir results/evaluation/improvement1

# Evaluate improvement 2
python src/evaluation/evaluate_agent.py --model results/dqn


### Compare All Agents (FINAL COMPARISON)
powershellpython src/evaluation/evaluate_agent.py \
  --compare results/dqn_baseline_*/dqn_baseline_model.zip results/dqn_improvement1_*/dqn_baseline_model.zip results/dqn_improvement2_*/dqn_baseline_model.zip \
  --labels "Baseline" "Improvement 1" "Improvement 2" \
  --n_episodes 50 \
  --save_dir results/final_comparison



### Visualising

# Visualize training progress for each agent
python src/evaluation/visualize_training.py --results_dir results/dqn_baseline_* --save_dir results/viz/baseline

python src/evaluation/visualize_training.py --results_dir results/dqn_improvement1_* --save_dir results/viz/improvement1

python src/evaluation/visualize_training.py --results_dir results/dqn_improvement2_* --save_dir results/viz/improvement2

# Create comprehensive comparison report
python src/evaluation/visualize_training.py \
  --compare results/dqn_baseline_* results/dqn_improvement1_* results/dqn_improvement2_* \
  --labels "Baseline" "Improvement 1" "Improvement 2" \
  --save_dir results/comparison_report

#### GIFS

# Create GIF of baseline agent playing
python src/evaluation/visualize_training.py --model results/dqn_baseline_*/dqn_baseline_model.zip --create_gif --save_dir results/gifs/baseline

# Create GIF of best agent
python src/evaluation/visualize_training.py --model results/dqn_improvement2_*/dqn_baseline_model.zip --create_gif --save_dir results/gifs/improvement2


### Visualise and Evaluate
python src/evaluation/Visualiser.py --model results/DQN_Baseline/dqn_baseline_model.zip --n_episodes 50 --save_dir results/viz/baseline

### Visualise,Evaluate,Gif
python src/evaluation/Visualiser.py --model results/DQN_Baseline/dqn_baseline_model.zip --n_episodes 50 --save_dir results/eval_plots