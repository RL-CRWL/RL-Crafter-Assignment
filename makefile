eval_src=Reinforcement-Learning-Project-2026-Crafter/src/evaluation/Evaluate-PPO.py
ppo_results=Reinforcement-Learning-Project-2026-Crafter/results/PPO
ppo_base=Reinforcement-Learning-Project-2026-Crafter/results/PPO/models/ppo_baseline/ppo_baseline_model.zip
ppo_improv_1=Reinforcement-Learning-Project-2026-Crafter/results/PPO/models/ppo_improv_1/ppo_rnd_model.zip
ppo_improv_2=Reinforcement-Learning-Project-2026-Crafter/results/PPO/models/ppo_improv_2/recurrent_ppo_model.zip
episodes=50
ppo_base_save=Reinforcement-Learning-Project-2026-Crafter/results/PPO/results/ppo_baseline
ppo_1_save=Reinforcement-Learning-Project-2026-Crafter/results/PPO/results/ppo_improv_1
ppo_2_save=Reinforcement-Learning-Project-2026-Crafter/results/PPO/results/ppo_improv_2

PPO_Baseline: Reinforcement-Learning-Project-2026-Crafter/src/agents/PPO_baseline.py
	python $^

PPO_Improv1: Reinforcement-Learning-Project-2026-Crafter/src/agents/PPO_improv1.py
	python $^

Results: $(eval_src)
	python $(eval_src) --model $(ppo_base) --n_episodes $(episodes) --save_dir $(ppo_base_save)
	python $(eval_src) --model $(ppo_improv_1) --n_episodes $(episodes) --save_dir $(ppo_1_save)
	python $(eval_src) --model $(ppo_improv_2) --n_episodes $(episodes) --save_dir $(ppo_2_save)
