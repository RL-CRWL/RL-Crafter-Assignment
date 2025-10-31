# 🎮 Reinforcement Learning - Crafter Environment

Deep Q-Network (DQN) implementations for the Crafter survival game environment. This project implements and iteratively improves RL agents through systematic experimentation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)

## 📊 Results Summary

| Model | Mean Reward | Survival Time | Achievements | Geometric Mean |
|-------|-------------|---------------|--------------|----------------|
| Baseline | 3.26 ± 1.50 | 177.7 steps | 10/22 | 21.79% |
| **Improvement 1** | **4.40 ± 1.27** ✅ | 193.5 steps | 10/22 | 34.61% |
| Improvement 2 | 3.58 ± 1.37 | **198.1 steps** ✅ | 8/22 | **48.24%** ✅ |

**Key Achievements:**
- 🏆 **Improvement 1:** +35% reward improvement over baseline
- 🏆 **Improvement 2:** Best survival time and achievement consistency
- ✅ Both improvements successfully beat baseline

## 🗂️ Project Structure

```
Reinforcement-Learning-Project-2026-Crafter/
├── configs/
│   └── configs.py              # Training configurations
├── src/
│   ├── agents/
│   │   ├── DQN_baseline.py     # Baseline DQN implementation
│   │   ├── DQN_improv1.py      # Improvement 1: Enhanced CNN
│   │   └── DQN_improv2.py      # Improvement 2: Curriculum Learning
│   ├── utils/
│   │   └── wrappers.py         # Environment wrappers
│   └── evaluation/
│       └── compare_models.py   # Model comparison script
├── scripts/
│   ├── train.py                # Main training script
│   └── evaluate_saved_model.py # Evaluation script
├── results/                    # Training results and saved models
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Reinforcement-Learning-Project-2026-Crafter.git
cd Reinforcement-Learning-Project-2026-Crafter

# Create virtual environment
python -m venv crafter_env
source crafter_env/bin/activate  # On Windows: crafter_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train baseline
python scripts/train.py --config DQN_CONFIG

# Train Improvement 1 (Enhanced CNN)
python scripts/train.py --config DQN_IMPROVEMENT_1

# Train Improvement 2 (Curriculum Learning)
python scripts/train.py --config DQN_IMPROVEMENT_2

# Quick test (10k steps)
python scripts/train.py --config QUICK_TEST_CONFIG
```

### 3. Evaluate Models

```bash
# Evaluate a saved model
python scripts/evaluate_saved_model.py results/model_directory/model.zip --n_episodes 50

# Compare all three models
python src/evaluation/compare_models.py \
    results/baseline/model.zip \
    results/improv1/model.zip \
    results/improv2/model.zip \
    --labels "Baseline" "Improvement 1" "Improvement 2" \
    --n_episodes 50
```

## 🔬 Methodology

### Baseline: DQN with Standard CNN

Standard Deep Q-Network implementation using Stable Baselines3:
- 3-layer CNN for feature extraction
- Experience replay buffer (50k)
- Target network with periodic updates
- ε-greedy exploration (15% decay)

**Results:** 3.26 mean reward, 177.7 survival time

### Improvement 1: Enhanced CNN Architecture

**Motivation:** Improve feature extraction and training stability

**Key Changes:**
- ✅ Observation normalization (0-255 → 0-1)
- ✅ Wider CNN (48-96-96 filters vs 32-64-64)
- ✅ Orthogonal weight initialization
- ✅ Increased replay buffer (75k)

**Failed Attempt:** Initially tried 4-layer deep CNN with high learning rate (2.5e-4)
- Result: 0.90 reward (catastrophic failure)
- Lesson: Incremental changes beat radical ones

**Successful Approach:** Conservative hyperparameters with proven architecture
- Result: **4.40 reward (+35% vs baseline)** ✅

### Improvement 2: Curriculum Learning

**Motivation:** Improve achievement consistency and survival through better exploration

**Key Changes:**
- ✅ Extended exploration (30% vs 10%)
- ✅ Lower final epsilon (0.01 vs 0.05)
- ✅ Same proven CNN as Improvement 1
- ❌ No reward shaping (maintains train/eval consistency)

**Failed Attempt:** Reward shaping for achievements
- Result: 1.66 reward (training/eval mismatch)
- Lesson: Train on what you evaluate on

**Successful Approach:** Curriculum learning without reward modification
- Result: **198.1 survival (best), 48.24% geometric mean (best)** ✅

## 📈 Training Details

### Hardware Requirements

- **GPU:** NVIDIA GTX 750 Ti (2GB VRAM) or better
- **CPU:** Multi-core recommended
- **RAM:** 8GB minimum
- **Storage:** ~500MB per trained model

### Training Time

- **Baseline:** ~6-8 hours (500k steps)
- **Improvement 1:** ~7-9 hours (larger buffer)
- **Improvement 2:** ~7-9 hours

### Memory Optimization

For 2GB VRAM GPUs:
- Buffer size limited to 75k (target was 100k)
- Batch size 32 (optimal for training speed)
- Gradient accumulation not needed

## 📊 Evaluation Metrics

We evaluate agents on the following metrics:

1. **Mean Reward:** Average cumulative reward per episode
2. **Survival Time:** Average number of steps before death
3. **Achievement Unlocking:** Number of unique achievements (out of 22)
4. **Geometric Mean:** Consistency of achievement unlocking across episodes
5. **Standard Deviation:** Variance in performance

## 🔍 Key Findings

### 1. Observation Preprocessing Matters
Normalizing pixel values from [0,255] to [0,1] improved reward by 35%

### 2. Hyperparameter Stability is Critical
- Learning rate 2.5e-4: 0.90 reward ❌
- Learning rate 1e-4: 4.40 reward ✅

### 3. Incremental Beats Radical
Simplified improvements outperformed aggressive architectural changes

### 4. Exploration Schedule Affects Behavior
Extended exploration (30% vs 10%) improved survival and consistency

### 5. Train What You Evaluate
Reward shaping created train/eval mismatch, hurting performance

## 🛠️ Configuration Options

Edit `configs/configs.py` to modify hyperparameters:

```python
# Example: Custom configuration
CUSTOM_CONFIG = {
    'learning_rate': 1e-4,
    'buffer_size': 75000,
    'batch_size': 32,
    'exploration_fraction': 0.2,
    'preprocess_type': 'normalize',
    'device': 'cuda',
    'total_timesteps': 500000,
}
```

Available preprocessing types:
- `'none'` - No preprocessing
- `'normalize'` - Scale to [0, 1]
- `'grayscale'` - Convert to grayscale (not implemented)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{crafter_rl_2025,
  author = {[Your Name]},
  title = {Iterative Improvements to DQN for Crafter Environment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/Reinforcement-Learning-Project-2026-Crafter}
}
```

## 📚 References

1. **Crafter Environment:** Hafner, D. (2021). "Benchmarking the Spectrum of Agent Capabilities." [arXiv:2109.06780](https://arxiv.org/abs/2109.06780)
2. **DQN:** Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature.
3. **Stable Baselines3:** Raffin, A., et al. (2021). [Documentation](https://stable-baselines3.readthedocs.io/)

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📄 License

This project is for academic use only. See the course guidelines for details.

## 🙏 Acknowledgments

- **Crafter Environment:** Thanks to Danijar Hafner for the excellent benchmark
- **Stable Baselines3:** For the robust DQN implementation
- **Course Instructors:** For guidance and support

## 📞 Contact

For questions about this project, contact:
- [Your Name] - [Your Email]
- Course: COMS4061A/COMS7071A
- University of the Witwatersrand

---

**Status:** ✅ Assignment Complete - Both improvements successfully beat baseline!

**Last Updated:** October 30, 2025
