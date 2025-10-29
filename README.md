# RL-Crafter-Assignment
RL Group Assignment 2025

environment setup
python3.12 -m venv ~/venvs/rlenv (if you have a specific folder to save venvs to)
python3.12 -m venv rlenv (otherwise, this should be in the directory)
source ~/venvs/rlenv/bin/activate
else source rlenv/bin/activate

pip install crafter - Install Crafter
pip install stable-baselines3[extra] - provides algorithms such as the DQN
pip install shimmy - creates a wrappers so that crafter can work with gymnasium instead of gym
pip install gym - required by shimmy if we would like to use gymnasium
(gymnasium required with gym to use shimmy)

if you want to see the game in action and participate:
pip install pygame # Needed for human interface
python3 -m crafter.run_gui # Start the game

will implement a text file and a shell script for auto-install

=======================================
run using python train.py --algorithm dqn --steps 500000 --outdir logdir/dqn
when video works: python train.py --algorithm dqn --steps 100000 --record-video --outdir logdir/dqn_with_video