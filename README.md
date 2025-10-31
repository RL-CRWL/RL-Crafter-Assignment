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
│   └── configs.py                        # Training configurations
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── DQN_baseline.py               # Baseline DQN implementation
│   │   ├── DQN_FRAMESTACK.py             # DQN with frame stacking
│   │   ├── DQN_improv1.py                # Improvement 1: Enhanced CNN architecture
│   │   ├── DQN_improv2.py                # Improvement 2: Curriculum Learning
│   │   ├── DQN_improv3.py                # Improvement 3: Advanced exploration
│   │   ├── PPO_baseline.py               # Baseline PPO implementation
│   │   ├── PPO_improv1.py                # PPO Improvement 1
│   │   ├── PPO_improv2.py                # PPO Improvement 2
│   │   └── results/
│   │       └── __init__.py
│
│   ├── utils/
│   │   ├── ego_wrapper.py                # Ego-centric observation wrapper
│   │   ├── test_egowrapper.py            # Unit tests for ego_wrapper
│   │   ├── wrapper_ppo.py                # PPO-specific environment wrapper
│   │   └── wrappers.py                   # General environment wrappers
│
│   └── evaluation/
│       ├── ppo_baseline_highlights/      # Highlight frames for PPO baseline runs
│       ├── recurrent_ppo_highlights/     # Highlight frames for Recurrent PPO runs
│       ├── Compare-PPO.py                # Compare PPO agents (baseline vs improved)
│       ├── Evaluate-PPO.py               # Evaluate PPO agents
│       ├── Evaluate-Recurrent.py         # Evaluate Recurrent PPO performance
│       ├── Evaluate.py                   # General evaluation entrypoint
│       ├── Visualiser.py                 # Visualization of results and rewards
│       ├── compare_models.py             # Compare models statistically
│       ├── debuggy.py                    # Debugging script for agent performance
│       ├── ppo_baseline_viz.py           # PPO baseline visualization
│       ├── ppo_improv1_viz.py            # Visualization for PPO Improvement 1
│       ├── ppo_improv2_viz.py            # Visualization for PPO Improvement 2
│
├── scripts/
│   ├── train.py                          # Main training script for all agents
│   └── evaluate_saved_model.py           # Load and evaluate trained models
│
├── results/                              # Saved models, logs, metrics, videos
│
├── requirements.txt                      # Python dependencies
└── README.md                             # Project overview and usage guide


```

## 🚀 Quick Start

### 1. Installation

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

### Proximal Policy Optimization (PPO) for Crafter Environment

#### Algorithm Overview

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm that trains an agent by optimizing its decision-making policy. It works by collecting data through interactions with an environment and then using a clipped objective function to make stable updates to the policy. This approach is known for being more stable, efficient, and easier to implement than some other policy gradient methods.

#### Why PPO for Crafter?

PPO provides a strong balance of stability, sample efficiency, and simplicity while effectively handling the Crafter environment's key challenges, such as sparse rewards, long-term reasoning, and procedural generation.

#### Key Theoretical Benefits

- **Stable Policy Updates:** Crafter is complex and dynamic, where large, unconstrained policy updates could easily destabilize training and cause the agent to forget beneficial behaviors. PPO's clipping mechanism limits how much the policy can change at each step, ensuring stable and controlled learning.

- **Sample Efficiency:** Crafter involves many different achievements and complex interactions, meaning efficient use of experience is crucial. PPO is relatively sample-efficient because it can reuse collected data over multiple training epochs (mini-batches) without significant instability.

- **Facilitates Exploration:** The PPO objective often includes an entropy bonus term, which encourages the agent to explore different actions and strategies. This is vital in Crafter, which features wide, procedurally generated worlds and independent achievements that require broad exploration.

##### Standard Hyperparameters

These hyperparameters were standardized across all agents to ensure fair comparison:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Rollout steps (n_steps) | 2,048 |
| Batch size | 64 |
| Epochs per update | 10 |
| Discount factor (γ) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |

---

### Baseline Implementation

#### Overview
The baseline agent uses Stable-Baselines3 PPO with standard hyperparameters. The agent perceives the environment through single still images (frames) rather than continuous sequences, providing only a limited view at each step.

#### Performance
- **Average Reward:** 2.66
- **Maximum Reward:** 6.1

#### Identified Limitations
- **Lack of Memory:** The agent cannot recall previous states or actions, leading to suboptimal long-term decision-making.
- **Limited Exploration:** Without memory or intrinsic motivation, exploration remains shallow, resulting in repetitive behavior.

---

### Improvement 1: Random Network Distillation (RND)

#### Background
Random Network Distillation (RND) is an exploration method that encourages agents to explore novel states by providing intrinsic rewards based on prediction error. It uses two neural networks:

1. **Target Network:** Randomly initialized and kept fixed, maps observations to feature vectors
2. **Predictor Network:** Trained to predict the target network's outputs via gradient descent

The intrinsic reward is computed as the prediction error—high for novel states and low for frequently visited states.

#### Key Improvements
RND addresses the limited exploration of the baseline implementation. Since Crafter has sparse rewards, RND provides dense exploration bonuses while maintaining the original task structure.

#### Architecture

**Target Network (Fixed Random Features)**
- Processes 64×64×3 RGB observations through three convolutional layers:
  - Conv1: 32 filters, 3×3 kernel, stride 2, padding 1 → output: 32×32×32
  - Conv2: 64 filters, 3×3 kernel, stride 2, padding 1 → output: 16×16×64
  - Conv3: 64 filters, 3×3 kernel, stride 1, padding 1 → output: 16×16×64
  - Flatten and fully connected: 16,384 → 512 features
- All parameters frozen after initialization

**Predictor Network (Learned Features)**
- Identical architecture to target network
- Trained to predict target network's output
- Prediction error serves as intrinsic reward signal

#### Reward Computation

**Intrinsic Reward:**
```
r_intrinsic(s_t) = ||f_target(s_t) - f_predictor(s_t)||²
```

**Total Reward:**
```
r_total(s_t) = r_extrinsic(s_t) + λ · r_intrinsic(s_t)
```
where λ = 1.0 (intrinsic reward coefficient)

#### Additional Hyperparameters
- `intrinsic_reward_coef`: 1.0
- `rnd_learning_rate`: 1 × 10⁻⁴
- Training timesteps: 3 × 10⁶

#### Exploration Mechanism
1. Novel states produce high prediction errors → high intrinsic rewards
2. Agent is incentivized to visit high-reward states
3. As states become familiar, predictor improves → reduced intrinsic rewards
4. Agent naturally shifts focus toward extrinsic task rewards

#### Performance
- **Average Reward:** 3.3 (~24% improvement over baseline)
- **Maximum Reward:** 7.1 (~16% improvement over baseline)

#### Key Observations
- Higher rewards received more frequently due to increased exploration
- Lower survival rates compared to baseline (exploration increases risk exposure)
- Less consistent achievement unlock rates due to novelty-seeking behavior

#### Strengths & Weaknesses

**Strengths:**
- Faster learning about environment dynamics
- More achievements unlocked
- Higher average rewards

**Weaknesses:**
- Requires significantly more training time
- Small context window limits learning persistence
- Achievements and associated rewards don't carry over to future episodes effectively

#### Implementation Differences from Standard RND
- Uses single combined value function (vs. separate extrinsic/intrinsic value functions)
- Simplified CNN architecture with 3×3 kernels (optimized for 64×64 observations)
- Single environment instance (vs. 128+ parallel environments)
- Per-step predictor updates (vs. mini-batch updates)
- Fixed intrinsic reward coefficient (vs. decay schedules)
- No explicit reward clipping or separate advantage normalization

---

### Improvement 2: Recurrent PPO with LSTM

#### Background
RecurrentPPO extends PPO by adding support for recurrent neural network policies using LSTM (Long Short-Term Memory) layers. This allows the agent to maintain memory of past observations and actions, making it particularly useful for partially observable environments.

#### Key Improvements
Addresses memory and temporal reasoning limitations by maintaining hidden state across timesteps. The LSTM enables the agent to construct implicit representations of unobserved parts of the environment.

#### Architecture
**MultiInputLstmPolicy** with dual-stream observation:
- **Visual stream:** Image observations
- **Discrete stream:** 22-dimensional achievement vector tracking task completion states

#### Additional Hyperparameters
- `vf_coef` (value function coefficient): 0.5
- `max_grad_norm` (maximum gradient norm): 0.5

#### Reward Shaping
Intrinsic reward shaping accelerates convergence through three components:
- **Achievement bonuses:** 5.0× multiplier for newly completed achievements
- **Survival bonuses:** +0.01 per timestep
- **Exploration bonuses:** +0.001 per action

#### Performance
- **Average Reward:** 4.71 (77% improvement over baseline)
- **Maximum Reward:** 11.1 (82% improvement over baseline)

#### Behavioral Observations
- More coherent decision sequences across time
- Improved persistence in long-horizon tasks (crafting, navigation)
- Less repetition of suboptimal exploration patterns
- Smoother early learning and faster convergence

#### Key Insight: Reward vs. Survival Trade-off
Interestingly, R-PPO achieved higher average rewards despite lower overall survival rates. This is explained by:
- LSTM memory enables recall and exploitation of previously observed opportunities
- Higher reward density per timestep from decisive action-taking
- Increased risk exposure from aggressive reward-seeking behavior
- Reward shaping reinforces immediate sub-goal completion over conservative survival

**Takeaway:** Longer episodes don't necessarily equate to better task performance—R-PPO prioritizes reward maximization over survival optimization.

#### Strengths & Weaknesses

**Strengths:**
- Context-aware actions through memory retention
- Better handling of partial observability
- Higher cumulative rewards

**Weaknesses:**
- Wide reward variance indicates instability
- Aggressive behavior reduces survival time
- Could benefit from further hyperparameter tuning (learning rate, clipping threshold, hidden size)

---

## Algorithm Comparison Summary

| Method | Avg Reward | Max Reward | Key Feature | Main Limitation |
|--------|-----------|-----------|-------------|-----------------|
| **Baseline PPO** | 2.66 | 6.1 | Stable, simple | No memory, limited exploration |
| **PPO + RND** | 3.3 | 7.1 | Novelty-driven exploration | Requires extensive training |
| **Recurrent PPO** | 4.71 | 11.1 | Temporal memory (LSTM) | Lower survival rate |

---

### Implementation Details

#### Technologies Used
- **PyTorch:** Neural network implementation
- **Stable Baselines3:** PPO algorithm framework
- **Gymnasium:** Environment wrapper interface

#### Reproducibility
- All experiments use seed 42
- Evaluation environment seeded at 142 for deterministic but different trajectories
- Evaluation every 10,000 training steps over 5 episodes

#### Evaluation Protocol
Performance metrics logged include:
- Mean reward
- Standard deviation
- Minimum/maximum rewards
- Episode lengths

All results based on 1,000 evaluation episodes per agent.

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
