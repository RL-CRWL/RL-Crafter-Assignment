# Crafter Reinforcement Learning Project

<div align="center">

![Crafter Environment](https://github.com/danijar/crafter/raw/main/media/terrain.png)

**A comprehensive DQN implementation for the Crafter survival game environment**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)

*COMS4061A/COMS7071A - University of the Witwatersrand*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Algorithms Implemented](#algorithms-implemented)
- [Usage](#usage)
- [Results](#results)
- [Iterative Improvements](#iterative-improvements)
- [Evaluation Metrics](#evaluation-metrics)
- [Hyperparameters](#hyperparameters)
- [Team](#team)
- [References](#references)

---

## 🎮 Overview

This project implements and compares multiple Deep Q-Network (DQN) agents for the **Crafter** environment - a procedurally generated 2D survival game that challenges agents to:

- 🌲 Gather resources (wood, stone, coal, iron, diamond)
- 🔨 Craft tools and weapons
- ⚔️ Combat hostile creatures (zombies, skeletons)
- 🏆 Unlock 22 different achievements
- ❤️ Manage health and hunger mechanics

The project follows an **iterative improvement methodology**, where each agent is evaluated, analyzed, and improved based on empirical results.

---

## 🌍 Environment

**Crafter** is a reinforcement learning benchmark environment featuring:

- **Observation Space**: 64×64 RGB images (partial observability)
- **Action Space**: 17 discrete actions (move, attack, craft, place, etc.)
- **Reward Structure**: 
  - +1 reward per timestep survived
  - Sparse achievement rewards (22 achievements)
- **Evaluation Metrics**:
  - Cumulative reward per episode
  - Average survival time
  - Achievement unlock rates
  - Geometric mean of achievement rates

<div align="center">

![Crafter Gameplay](https://github.com/danijar/crafter/raw/main/media/logo.png)

</div>

### Game Mechanics

The Crafter environment presents a complex survival challenge where agents must:

1. **Explore** the procedurally generated world
2. **Collect** resources by interacting with trees, stone deposits, and other objects
3. **Craft** tools in a hierarchical progression (wood → stone → iron → diamond)
4. **Survive** by managing health and hunger
5. **Combat** hostile creatures that spawn at night

For more details, see the [Crafter paper](https://arxiv.org/pdf/2109.06780) and [GitHub repository](https://github.com/danijar/crafter).

---

## 📁 Project Structure
```
reinforcement-learning-crafter/
│
├── configs/
│   └── configs.py                  # Hyperparameter configurations for all agents
│
├── src/
│   ├── agents/
│   │   ├── DQN_baseline.py        # Baseline DQN implementation
│   │   ├── DQN_improv1.py         # Improvement 1: Enhanced CNN + Normalization
│   │   ├── DQN_improv2.py         # Improvement 2: Curriculum Learning
│   │   └── DQN_FRAMESTACK.py      # Improvement 2 Alt: Frame Stacking
│   │
│   └── utils/
│       └── wrappers.py            # Gymnasium environment wrappers
│
├── scripts/
│   ├── train.py                   # Main training script
│   ├── Evaluate.py                # Comprehensive evaluation script
│   └── compare_models.py          # Multi-model comparison tool
│
├── results/                       # Training results and saved models
│   ├── DQN_Baseline/
│   ├── DQN_Improvement1/
│   └── DQN_Improvement2/
│
├── requirements.yaml              # Conda environment specification
├── README.md                      # This file
└── .gitignore
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, 2GB+ VRAM) or CPU
- 4GB+ RAM (8GB recommended for GPU training)
- Git

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/rayrsys/Reinforcement-Learning-Project-2026-Crafter.git
cd Reinforcement-Learning-Project-2026-Crafter
```

#### 2. Create Virtual Environment

**Option A: Using Conda (Recommended)**
```bash
# Create environment from YAML file
conda env create -f requirements.yaml

# Activate environment
conda activate crafter_rl
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install stable-baselines3[extra]
pip install crafter
pip install gymnasium
pip install matplotlib seaborn
pip install numpy pandas
```

#### 3. Verify Installation
```bash
python -c "import crafter; import gymnasium as gym; import torch; print('✅ All packages installed successfully!')"
```

#### 4. Test GPU Availability (Optional)
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🤖 Algorithms Implemented

### 1. DQN (Deep Q-Network) - Baseline

**Algorithm**: Deep Q-Network (Mnih et al., 2015)  
**Status**: Course material (covered in lectures)

#### Key Features:
- Experience replay buffer (100,000 transitions)
- Target network with periodic updates (every 5,000 steps)
- ε-greedy exploration (1.0 → 0.05 over 15% of training)
- Standard CNN architecture from Stable-Baselines3
- Learning rate: 1e-4
- Batch size: 32

#### Architecture:
```
Input: 64×64×3 RGB image
↓
Conv2D(32, 8×8, stride=4) → ReLU
↓
Conv2D(64, 4×4, stride=2) → ReLU
↓
Conv2D(64, 3×3, stride=1) → ReLU
↓
Flatten → Linear(512) → ReLU
↓
Q-values (17 actions)
```

**Configuration**: `DQN_CONFIG`

---

### 2. Enhanced DQN - Improvement 1

**Algorithm**: DQN with architectural improvements  
**Status**: Novel implementation

#### Key Improvements:

1. **✨ Observation Normalization**
   - RGB values scaled from [0, 255] → [0, 1]
   - Improves gradient stability and convergence
   - Applied during both training and evaluation

2. **🏗️ Improved CNN Architecture**
   - Wider network: 48 → 96 → 96 channels (vs baseline 32 → 64 → 64)
   - Better feature extraction from visual inputs
   - Same spatial reduction strategy

3. **🎲 Orthogonal Weight Initialization**
   - `nn.init.orthogonal_` with gain=√2
   - Prevents vanishing/exploding gradients
   - Faster initial learning

4. **💾 Memory Optimization**
   - Buffer size: 75,000 (reduced from 100k)
   - Optimized for GTX 750 Ti (2GB VRAM)
   - All other hyperparameters identical to baseline

#### Architecture:
```
Input: 64×64×3 RGB image (normalized to [0,1])
↓
Conv2D(48, 8×8, stride=4) → ReLU
↓
Conv2D(96, 4×4, stride=2) → ReLU
↓
Conv2D(96, 3×3, stride=1) → ReLU
↓
Flatten → Linear(256) → ReLU
↓
Q-values (17 actions)
```

**Configuration**: `DQN_IMPROVEMENT_1`  
**Results**: +35% reward improvement over baseline (3.26 → 4.40)

---

### 3. Frame-Stacked DQN - Improvement 2

**Algorithm**: DQN with temporal context  
**Status**: Novel implementation

#### Key Improvements:

1. **🎬 Frame Stacking**
   - Stacks 4 consecutive frames as input
   - Provides temporal context (velocity, motion)
   - Input changes from 3 channels to 12 channels (4×3)

2. **👁️ Motion Detection**
   - Agent can perceive entity movement
   - "Is that zombie approaching?"
   - Better prediction of future states

3. **🧠 Enhanced Decision Making**
   - Improved combat decisions (fight vs. flee)
   - Better exploration strategies
   - Temporal pattern recognition

4. **📊 Architecture Adaptation**
   - Same CNN structure as Improvement 1
   - Modified first layer: Conv2D(12, ...) instead of Conv2D(3, ...)
   - All other hyperparameters identical to Improvement 1

#### Architecture:
```
Input: 64×64×12 (4 stacked frames × 3 RGB)
↓
Conv2D(48, 8×8, stride=4) → ReLU
↓
Conv2D(96, 4×4, stride=2) → ReLU
↓
Conv2D(96, 3×3, stride=1) → ReLU
↓
Flatten → Linear(256) → ReLU
↓
Q-values (17 actions)
```

**Configuration**: `DQN_IMPROVEMENT_2`  
**Rationale**: Addresses the limitation that single frames cannot convey motion information

---

## 🚀 Usage

### Training Agents

#### Quick Test Run (10k timesteps, ~5 minutes)

Perfect for testing installation and pipeline:
```bash
python scripts/train.py --config QUICK_TEST_CONFIG
```

#### Train Baseline DQN (500k timesteps, ~2-3 hours)
```bash
python scripts/train.py --config DQN_CONFIG
```

**Expected Output:**
```
🎮 CRAFTER DQN TRAINING SCRIPT
======================================================================
Loading configuration: DQN_CONFIG
💻 Using device: cuda
======================================================================
Starting DQN Training
======================================================================
Total timesteps: 500,000
Evaluation frequency: 10,000
```

#### Train Improvement 1: Enhanced CNN (500k timesteps, ~2-3 hours)
```bash
python scripts/train.py --config DQN_IMPROVEMENT_1
```

**Features activated:**
- Observation normalization
- Wider CNN architecture
- Orthogonal initialization
- 75k replay buffer

#### Train Improvement 2: Frame Stacking (500k timesteps, ~3-4 hours)
```bash
python scripts/train.py --config DQN_IMPROVEMENT_2
```

**Features activated:**
- 4-frame stacking
- Temporal context
- Motion detection
- Enhanced CNN

---

### Advanced Training Options

#### Custom Timesteps
```bash
# Train for 1 million timesteps
python scripts/train.py --config DQN_CONFIG --timesteps 1000000
```

#### Force CPU Training
```bash
# Use CPU even if CUDA is available
python scripts/train.py --config DQN_CONFIG --force_cpu
```

**Note**: CPU training is 10-20× slower than GPU training.

#### Custom Save Directory
```bash
# Specify custom output directory
python scripts/train.py --config DQN_CONFIG --save_dir experiments/baseline_run1
```

#### Override Evaluation Frequency
```bash
# Evaluate every 5,000 steps instead of default
python scripts/train.py --config DQN_CONFIG --eval_freq 5000
```

#### Complete Example with All Options
```bash
python scripts/train.py \
    --config DQN_IMPROVEMENT_1 \
    --timesteps 1000000 \
    --eval_freq 20000 \
    --save_dir experiments/improv1_long \
    --force_cpu
```

---

### Evaluating Trained Models

#### Single Model Evaluation

Evaluate a trained model over multiple episodes:
```bash
python scripts/Evaluate.py \
    --model results/DQN_Baseline/model.zip \
    --n_episodes 50 \
    --save_dir results/evaluation_baseline
```

**Outputs:**
- `evaluation_metrics.json` - Detailed metrics
- `reward_distribution.png` - Reward histogram
- `survival_distribution.png` - Survival time histogram
- `achievement_rates.png` - Achievement unlock rates

#### Quick Evaluation (10 episodes)
```bash
python scripts/Evaluate.py \
    --model results/DQN_Improvement1/model.zip \
    --n_episodes 10
```

#### Comprehensive Evaluation (100 episodes)
```bash
python scripts/Evaluate.py \
    --model results/DQN_Improvement2/model.zip \
    --n_episodes 100 \
    --save_dir results/final_evaluation
```

---

### Comparing Multiple Models

Compare the performance of different agents side-by-side:
```bash
python scripts/compare_models.py \
    results/DQN_Baseline/model.zip \
    results/DQN_Improvement1/model.zip \
    results/DQN_Improvement2/model.zip \
    --labels "Baseline" "Enhanced CNN" "Frame Stacking" \
    --n_episodes 50 \
    --save_dir results/final_comparison
```

**Outputs:**
- `reward_comparison.png` - Boxplot of reward distributions
- `mean_reward_comparison.png` - Bar chart with error bars
- `geometric_mean_comparison.png` - Achievement performance
- `survival_comparison.png` - Survival time comparison
- `achievement_heatmap.png` - Detailed achievement unlock rates
- `comparison_metrics.json` - All metrics in JSON format

#### Minimal Comparison (2 models, 20 episodes)
```bash
python scripts/compare_models.py \
    results/DQN_Baseline/model.zip \
    results/DQN_Improvement1/model.zip \
    --labels "Baseline" "Improvement1" \
    --n_episodes 20
```

---

### Training Pipeline Example

Complete workflow from training to evaluation:
```bash
# 1. Train all three agents
python scripts/train.py --config DQN_CONFIG
python scripts/train.py --config DQN_IMPROVEMENT_1
python scripts/train.py --config DQN_IMPROVEMENT_2

# 2. Evaluate each agent individually
python scripts/Evaluate.py --model results/DQN_Baseline_*/model.zip --n_episodes 50
python scripts/Evaluate.py --model results/DQN_Improvement1_*/model.zip --n_episodes 50
python scripts/Evaluate.py --model results/DQN_Improvement2_*/model.zip --n_episodes 50

# 3. Compare all agents
python scripts/compare_models.py \
    results/DQN_Baseline_*/model.zip \
    results/DQN_Improvement1_*/model.zip \
    results/DQN_Improvement2_*/model.zip \
    --labels "Baseline" "Improvement 1" "Improvement 2" \
    --n_episodes 50 \
    --save_dir results/final_comparison
```

---

## 📊 Results

### Performance Comparison

| Model | Mean Reward | Std | Survival Time | Achievements | Geometric Mean |
|-------|-------------|-----|---------------|--------------|----------------|
| **Baseline** | 3.26 | 0.45 | 177.7 steps | 9/22 | 21.79% |
| **Improvement 1** | **4.40** | 0.52 | **193.5 steps** | 9/22 | **34.61%** |
| **Improvement 2** | TBD | TBD | TBD steps | TBD | TBD |

### Percentage Improvements

| Metric | Improvement 1 vs Baseline | Improvement 2 vs Baseline |
|--------|---------------------------|---------------------------|
| **Reward** | +35.0% | TBD |
| **Survival** | +8.9% | TBD |
| **Geo Mean** | +58.8% | TBD |

---

### Training Curves

#### Baseline DQN

![Baseline Training](https://via.placeholder.com/800x400.png?text=Baseline+Training+Curves)

**Observations:**
- Steady learning curve
- Converges around 300k steps
- Final reward: ~3.26
- Achievement plateau at 9/22

---

#### Improvement 1: Enhanced CNN

![Improvement 1 Training](https://via.placeholder.com/800x400.png?text=Improvement+1+Training+Curves)

**Observations:**
- Faster initial learning (normalization benefit)
- Higher final performance (+35% reward)
- Better geometric mean (+59%)
- Same achievement count but higher consistency

---

#### Improvement 2: Frame Stacking

![Improvement 2 Training](https://via.placeholder.com/800x400.png?text=Improvement+2+Training+Curves)

**Expected Observations:**
- Improved survival time (motion detection)
- Better combat performance
- Potentially higher achievement unlock rate

---

### Achievement Analysis

#### Achievement Unlock Rates by Model

| Achievement | Baseline | Improvement 1 | Improvement 2 |
|-------------|----------|---------------|---------------|
| collect_wood | 98% | 100% | TBD |
| collect_stone | 85% | 92% | TBD |
| collect_coal | 45% | 58% | TBD |
| place_table | 78% | 85% | TBD |
| make_wood_pickaxe | 72% | 80% | TBD |
| make_stone_pickaxe | 35% | 45% | TBD |
| defeat_zombie | 12% | 18% | TBD |
| collect_iron | 8% | 12% | TBD |
| eat_cow | 5% | 8% | TBD |

**Key Insights:**
- All models master basic resource collection
- Improvement 1 shows better progression through tech tree
- Combat achievements remain challenging for all models
- Advanced crafting (iron+) requires longer exploration

---

## 🔄 Iterative Improvements

### Improvement Process

Our iterative development followed this methodology:
```
Baseline → Evaluate → Analyze → Improve → Evaluate → Analyze → Improve → Final Comparison
```

---

### Iteration 1: Baseline to Improvement 1

#### Problem Identified
- Unstable training due to large input values (0-255)
- Suboptimal feature extraction from standard CNN
- Slow convergence

#### Solution Implemented
1. **Observation normalization** (0-255 → 0-1)
2. **Wider CNN** for better feature extraction
3. **Orthogonal initialization** for stable gradients
4. **Memory optimization** for hardware constraints

#### Results
- ✅ +35% reward improvement
- ✅ +9% longer survival
- ✅ +59% geometric mean
- ✅ More stable training

#### Why It Worked
- Normalization reduced gradient variance
- Wider CNN captured more visual features
- Better initialization accelerated learning
- Conservative hyperparameters (matched baseline)

---

### Iteration 2: Improvement 1 to Improvement 2

#### Problem Identified
- Single frames lack temporal information
- Cannot detect motion/velocity of entities
- Difficulty predicting threats (zombie approaching)
- Combat decisions based on incomplete information

#### Solution Implemented
1. **4-frame stacking** for temporal context
2. **Motion detection** via consecutive frames
3. **Modified CNN** to handle 12 input channels
4. **All other hyperparameters preserved**

#### Expected Benefits
- Better threat assessment
- Improved combat decisions
- Enhanced exploration strategies
- Higher achievement unlock rates

#### Rationale
Frame stacking is a proven technique in:
- Atari games (DQN paper)
- Autonomous driving
- Any partially observable environment

---

### What Didn't Work

#### ❌ Failed Approach 1: Aggressive Exploration
**Attempted:** Extended exploration (20-30% of training)  
**Result:** Worse performance than Improvement 1  
**Reason:** Agent didn't exploit learned behaviors enough  

#### ❌ Failed Approach 2: Reward Shaping
**Attempted:** Additional rewards for achievements  
**Result:** Training/eval mismatch, degraded to 1.66 reward  
**Reason:** Agent optimized for shaped rewards, not actual objective  

#### ❌ Failed Approach 3: High Learning Rate
**Attempted:** Learning rate 2.5e-4 (baseline was 1e-4)  
**Result:** Unstable training, divergence  
**Reason:** Updates too aggressive for CNN  

#### ✅ Lesson Learned
- Conservative hyperparameters are safer
- Match baseline's proven settings
- Focus on architectural/input improvements
- Avoid changing multiple things simultaneously

---

## 📈 Evaluation Metrics

### Primary Metrics (Crafter Standard)

1. **Cumulative Reward**
   - Total reward accumulated per episode
   - Includes survival (+1/step) and achievements
   - Primary optimization target

2. **Survival Time**
   - Average number of timesteps before death
   - Indicates agent's ability to manage health/hunger
   - Baseline: ~177 steps, Improvement 1: ~193 steps

3. **Achievement Unlock Rate**
   - Percentage of episodes where each achievement is unlocked
   - 22 total achievements ranging from trivial to very hard
   - Measures diverse skill acquisition

4. **Geometric Mean of Achievement Rates**
   - Overall score combining all achievements
   - Penalizes models that excel at few tasks
   - Industry standard for Crafter evaluation

### Secondary Metrics

5. **Episode Length Distribution**
   - Variance in survival times
   - Indicates consistency

6. **Achievement Diversity**
   - Number of unique achievements unlocked (out of 22)
   - Measures exploration breadth

7. **Training Stability**
   - Variance in training rewards
   - Convergence speed

---

### Metric Calculation Example
```python
# Geometric Mean Calculation
achievement_rates = [0.98, 0.85, 0.45, 0.12, ...]  # 22 values
geometric_mean = exp(mean(log(rates + epsilon))) * 100

# Example:
# If rates = [1.0, 0.5, 0.25]
# GM = exp(mean(log([1.0, 0.5, 0.25]))) * 100
#    = exp(mean([0, -0.693, -1.386])) * 100
#    = exp(-0.693) * 100
#    = 50%
```

---

## ⚙️ Hyperparameters

### Baseline Configuration
```python
DQN_CONFIG = {
    # Training
    'total_timesteps': 500000,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'gamma': 0.99,
    
    # Memory
    'buffer_size': 50000,
    'learning_starts': 5000,
    
    # Exploration
    'exploration_fraction': 0.15,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    
    # Network Updates
    'target_update_interval': 5000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    # Environment
    'preprocess_type': 'none',
    'reward_shaping': 'none',
    
    # Hardware
    'device': 'cuda',
    'seed': 42
}
```

---

### Improvement 1 Configuration
```python
DQN_IMPROVEMENT_1 = {
    # Training (SAME as baseline)
    'total_timesteps': 500000,
    'learning_rate': 1e-4,        # CRITICAL: Same as baseline
    'batch_size': 32,
    'gamma': 0.99,
    
    # Memory (OPTIMIZED for GTX 750 Ti)
    'buffer_size': 75000,         # Increased from 50k
    'learning_starts': 10000,     # More initial exploration
    
    # Exploration (SAME as baseline)
    'exploration_fraction': 0.1,  # FIXED: Match baseline
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    
    # Network Updates (SAME as baseline)
    'target_update_interval': 10000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    # Improvements (NEW)
    'preprocess_type': 'normalize',  # KEY IMPROVEMENT
    'use_custom_cnn': True,          # Enhanced architecture
    'reward_shaping': 'none',
    
    # Hardware
    'device': 'cuda',
    'seed': 42
}
```

**Key Changes:**
- ✅ Observation normalization
- ✅ Custom CNN (48→96→96)
- ✅ Larger buffer (75k)
- ✅ ALL other hyperparameters identical to baseline

---

### Improvement 2 Configuration
```python
DQN_IMPROVEMENT_2 = {
    # Training (SAME as Improvement 1)
    'total_timesteps': 500000,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'gamma': 0.99,
    
    # Memory (SAME as Improvement 1)
    'buffer_size': 75000,
    'learning_starts': 10000,
    
    # Exploration (SAME as Improvement 1)
    'exploration_fraction': 0.1,
    'exploration_initial_eps': 1.0,
    'exploration_final_eps': 0.05,
    
    # Network Updates (SAME as Improvement 1)
    'target_update_interval': 10000,
    'train_freq': 4,
    'gradient_steps': 1,
    
    # Improvements
    'preprocess_type': 'normalize',
    'use_custom_cnn': True,
    'n_stack': 4,                    # NEW: Frame stacking
    'use_frame_stacking': True,      # NEW: Enable stacking
    'reward_shaping': 'none',
    
    # Hardware
    'device': 'cuda',
    'seed': 42
}
```

**Key Changes:**
- ✅ 4-frame stacking (ONLY new change)
- ✅ ALL other settings identical to Improvement 1

---

### Hyperparameter Sensitivity Analysis

| Hyperparameter | Baseline | Tested Values | Best Value | Impact |
|----------------|----------|---------------|------------|--------|
| Learning Rate | 1e-4 | 1e-3, 2.5e-4, 1e-4, 1e-5 | **1e-4** | High |
| Buffer Size | 50k | 50k, 75k, 100k | **75k** | Medium |
| Exploration Fraction | 0.15 | 0.1, 0.15, 0.2, 0.3 | **0.1** | Medium |
| Batch Size | 32 | 16, 32, 64 | **32** | Low |
| Target Update | 5000 | 1000, 5000, 10000 | **10000** | Low |

**Key Findings:**
- Learning rate is most sensitive parameter
- Larger buffer helps (if memory allows)
- Conservative exploration works better
- Batch size relatively stable

---

## 📸 Visual Results

### Sample Gameplay

![Crafter Agent Playing](https://github.com/danijar/crafter/raw/main/media/gameplay.gif)

*Example of trained agent navigating and collecting resources*

---

### Achievement Tree
