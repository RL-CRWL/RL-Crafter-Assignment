conda create -n crafter_env python=3.10


conda activate crafter_env


pip install torch==2.3.1 stable-baselines3==2.3.2 gymnasium==0.29.1 crafter==1.7.1 matplotlib tqdm

pip install ruamel.yaml==0.17.30


python -m pip list | findstr crafter
python -m pip list | findstr gymnasium
python -m pip list | findstr stable-baselines3


expected output:

crafter              1.7.1
gymnasium            0.29.1
stable-baselines3    2.3.2


run test.py to check if everything got installed
