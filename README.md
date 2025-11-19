# Vehicular Edge Computing Task Offloading System

## Overview

This project implements and evaluates three deep reinforcement learning algorithms (A2C, DQN, DDQN) for optimizing task offloading decisions in Vehicular Edge Computing (VEC) environments.

## Features

- **Complete VEC Environment Simulation**: Following 3GPP TR 37.885 standards
- **Three RL Algorithms**: A2C, DQN, and DDQN with pre-trained models
- **Baseline Strategies**: Local processing, full offloading, and random offloading
- **Comprehensive Evaluation**: Performance metrics and visualizations

## Quick Start

### Run Everything
```bash
python run
```

### Train Specific Model
```bash
python train.py --model a2c --episodes 100
python train.py --model dqn --episodes 100
python train.py --model ddqn --episodes 100
```

### Evaluate Models
```bash
# Evaluate single model
python evaluate.py --model a2c --runs 5

# Compare all methods
python evaluate.py --compare --runs 3
```

### Test Installation
```bash
python test_all.py
```

## Directory Structure

```
code/
├── environment/          # VEC environment simulation
│   ├── task.py          # Task management
│   ├── vehicle.py       # Vehicle simulation
│   ├── rsu.py          # RSU (edge server) simulation
│   ├── communication.py # V2X communication
│   └── environment.py   # Main environment
├── models/              # RL algorithms
│   ├── a2c.py          # Actor-Critic algorithm
│   ├── dqn.py          # Deep Q-Network
│   ├── ddqn.py         # Double DQN
│   ├── a2c_model.pth   # Pre-trained A2C model
│   ├── dqn_model.pth   # Pre-trained DQN model
│   └── ddqn_model.pth  # Pre-trained DDQN model
├── baselines/           # Baseline strategies
│   ├── local.py        # Local processing only
│   ├── offload.py      # Full offloading
│   └── random_strategy.py # Random decisions
├── utils/               # Utility functions
│   └── env_wrapper.py  # RL environment wrapper
├── train.py            # Training script
├── evaluate.py         # Evaluation script
├── test_all.py         # Quick test script
└── run                 # Main execution script
```

## Pre-trained Models

Three pre-trained models are included:
- `models/a2c_model.pth` (10.96 MB)
- `models/dqn_model.pth` (35.43 MB)
- `models/ddqn_model.pth` (9.30 MB)

## Output

Results are saved to:
- `results/models/` - Trained model checkpoints
- `results/figures/` - Performance visualization plots

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.19+
- Matplotlib 3.3+

See `requirements.txt` for complete list.

## Citation

If you use this code, please cite our work.

