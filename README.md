# Vehicular Edge Computing Task Offloading System

## Description

This project implements a Vehicular Edge Computing (VEC) task offloading optimization system based on deep reinforcement learning. It implements and evaluates three deep reinforcement learning algorithms (A2C, DQN, DDQN) to optimize task offloading decisions in vehicular edge computing environments, improving task processing success rates and system performance.

**Key Features:**
- Complete VEC environment simulation
- Three reinforcement learning algorithms with pre-trained models
- Multiple baseline strategies for comparison (local processing, full offloading, random strategy)
- Comprehensive performance evaluation and visualization

---

## Dataset Information

**This project does not use any public datasets.** All simulations are generated programmatically based on the VEC environment model implemented in the `environment/` directory.

---

## Code Information

### Directory Structure

```
code/
├── environment/              # VEC environment simulation
│   ├── __init__.py
│   ├── task.py              # Task generation and management
│   ├── vehicle.py           # Vehicle simulation
│   ├── rsu.py               # RSU (edge server) simulation
│   ├── communication.py     # V2X communication
│   └── environment.py       # Main environment class
│
├── models/                   # Reinforcement learning algorithms
│   ├── __init__.py
│   ├── a2c.py               # Actor-Critic algorithm
│   ├── dqn.py               # Deep Q-Network
│   ├── ddqn.py              # Double DQN
│   ├── a2c_model.pth        # Pre-trained A2C model (10.96 MB)
│   ├── dqn_model.pth        # Pre-trained DQN model (35.43 MB)
│   └── ddqn_model.pth       # Pre-trained DDQN model (9.30 MB)
│
├── baselines/                # Baseline strategies
│   ├── __init__.py
│   ├── local.py             # Local processing only
│   ├── offload.py           # Full offloading strategy
│   └── random_strategy.py   # Random decision strategy
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── env_wrapper.py       # RL environment wrapper
│
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── test_all.py               # Quick test script
├── run.py                    # Main execution script
├── requirements.txt          # Dependency list
└── README.md                 # Project documentation
```

---

## Usage Instructions

### Installation

1. **Clone or download the project**
```bash
cd code
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python test_all.py
```

### Quick Start

#### 1. Run Complete Workflow
```bash
python run.py
```
This command will:
- Test all baseline strategies
- Evaluate all pre-trained models
- Generate performance comparison charts

#### 2. Train Models

**Train A2C model:**
```bash
python train.py --model a2c --episodes 100
```

**Train DQN model:**
```bash
python train.py --model dqn --episodes 100
```

**Train DDQN model:**
```bash
python train.py --model ddqn --episodes 100
```

**Training arguments:**
- `--model`: Model type (a2c/dqn/ddqn)
- `--episodes`: Number of training episodes (default: 100)
- `--hidden-dim`: Hidden layer dimension (default: 512)
- `--lr`: Learning rate (default: 0.0005)
- `--gamma`: Discount factor (default: 0.99)
- `--batch-size`: Batch size (default: 1024)
- `--update-freq`: Update frequency (default: 10)
- `--epsilon-start`: Initial exploration rate (default: 0.3)
- `--epsilon-end`: Final exploration rate (default: 0.01)
- `--epsilon-decay`: Exploration rate decay (default: 0.995)
- `--seed`: Random seed (default: 42)

#### 3. Evaluate Models

**Evaluate single model:**
```bash
python evaluate.py --model a2c --runs 5
```

**Compare all methods:**
```bash
python evaluate.py --compare --runs 3
```

**Evaluation arguments:**
- `--model`: Model to evaluate (a2c/dqn/ddqn/all)
- `--runs`: Number of evaluation runs (default: 5)
- `--seed`: Random seed (default: 42)
- `--compare`: Compare all methods including baselines

### Output

Training and evaluation results are saved to:
- `results/models/`: Trained model checkpoints
- `results/figures/`: Performance visualization charts
  - `strategy_comparison.png`: Strategy comparison bar chart

---

## Requirements

### Environment
- **OS**: Windows / Linux / macOS
- **Python**: 3.8 or higher
- **GPU**: Optional (CUDA acceleration supported, but CPU works fine)

### Python Dependencies

```
torch>=1.9.0          # Deep learning framework
numpy>=1.19.0         # Numerical computing
matplotlib>=3.3.0     # Data visualization
scipy>=1.5.0          # Scientific computing (optional)
```

See `requirements.txt` for the complete list.

### Hardware Recommendations
- **Memory**: 8GB or more recommended
- **Storage**: At least 500MB available space
- **CPU**: Multi-core processor recommended for training

---

## Methodology

### 1. Problem Formulation

This project models the VEC task offloading problem as a Markov Decision Process (MDP):
- **State**: Vehicle state, RSU state, task state, communication state
- **Action**: Task offloading decision (local/offload to which RSU/skip)
- **Reward**: Based on task success rate and latency

### 2. Reinforcement Learning Algorithms

#### A2C (Advantage Actor-Critic)
- Policy gradient method with value function baseline
- Stable training with actor-critic architecture

#### DQN (Deep Q-Network)
- Q-learning with experience replay and target network
- High sample efficiency with offline learning

#### DDQN (Double DQN)
- Addresses Q-value overestimation in DQN
- Uses double estimation for more accurate value learning

### 3. Baseline Strategies

- **本地**: All tasks processed locally
- **Offload**: All tasks offloaded to the nearest RSU
- **Random**: Random selection of offloading target

### 4. Performance Metrics

- **Task Success Rate**: Proportion of tasks completed within deadline
- **Average Latency**: Average task processing time
- **Load Balance**: Distribution of workload across computing nodes

---

## Citation

**This project does not use any public datasets.** All experiments are conducted using the custom VEC simulation environment. No citation is required for dataset usage.

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions, bug reports, and feature requests are welcome!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Reporting Issues

If you find a bug or have a feature request, please submit an issue on the Issues page.

### Code Standards

- Follow PEP 8 Python code style
- Add appropriate documentation for new features
- Ensure code passes existing tests

---

## Contact

For questions or suggestions, please contact:
- **Email**: charlie_ge2023@163.com
- **议题**: Submit an issue on the GitHub repository

---

## Changelog

### v1.0.0 (2025-11-29)
- Initial release
- Implemented A2C, DQN, DDQN algorithms
- Provided pre-trained models and baseline comparisons
- Complete documentation and usage instructions

---

**Thank you for using this project!**

