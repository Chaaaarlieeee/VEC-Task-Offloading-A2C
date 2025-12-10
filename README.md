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

### Dataset Reproduction

To reproduce the simulation dataset used in our experiments, use the provided reproduction script:

```bash
python reproduce_simulation.py --episodes 100 --seed 42 --output ./data
```

**Script Parameters:**
- `--episodes`: Number of simulation episodes to generate (default: 100)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output`: Output directory for generated dataset (default: ./data)
- `--format`: Output format - 'pickle', 'json', or 'both' (default: pickle)

**Dataset Structure:**
The generated dataset includes:
- **Episode Data**: Complete state-action-reward trajectories for each episode
- **Task Outcomes**: Detailed information about each task (size, complexity, deadline, success status)
- **Statistics**: Success rates, rewards, and performance metrics
- **Metadata**: Environment configuration and reproducibility information

**Example Usage:**
```python
import pickle

# Load the generated dataset
with open('./data/simulation_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Access episode data
episode = dataset['episodes'][0]
print(f"Episode success rate: {episode['statistics']['success_rate']}")
```

For detailed documentation on the dataset format and structure, see the comments in `reproduce_simulation.py`.

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
├── reproduce_simulation.py   # Dataset reproduction script
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

#### 1. Reproduce Simulation Dataset (Recommended First Step)
```bash
python reproduce_simulation.py --episodes 100 --seed 42 --output ./data
```
This generates the simulation dataset used in experiments. This step is essential for reproducibility.

#### 2. Run Complete Workflow
```bash
python run.py
```
This command will:
- Test all baseline strategies
- Evaluate all pre-trained models
- Generate performance comparison charts

#### 3. Train Models

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

#### 4. Evaluate Models

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

**State Space (88 dimensions):**
- **Vehicle State (6 dims)**: Position, speed, CPU capacity, cache availability, queue length, current processing tasks
- **RSU State (70 dims)**: For each of 10 RSUs - position, CPU capacity, cache availability, queue length, failure status, distance to vehicle, coverage indicator
- **Transmission State (6 dims)**: Number of transmitting tasks, average transmission rate, channel quality metrics
- **New Task State (5 dims)**: Task size, complexity, priority, deadline, urgency indicator
- **Time State (1 dim)**: Current simulation time

**Action Space (12 actions):**
- Action 0: Process task locally on vehicle
- Actions 1-10: Offload task to RSU 1-10
- Action 11: Skip task assignment (reject task)

**Reward Function:**
The reward is designed to maximize task success rate while minimizing latency:
```
reward = α * (success_indicator) - β * (normalized_latency) - γ * (penalty_for_failure)
```
Where:
- α = 10.0 (reward for successful task completion)
- β = 1.0 (penalty coefficient for latency)
- γ = 5.0 (penalty for task failure)

### 2. Reinforcement Learning Algorithms

#### A2C (Advantage Actor-Critic)
- **Architecture**: Policy network (actor) and value network (critic) with shared feature extraction
- **Network Structure**: 
  - Input: 88-dimensional state vector
  - Hidden layers: 512-1024-512 neurons with ReLU activation and layer normalization
  - Output: 12-dimensional action probability distribution (actor) + value estimate (critic)
- **Training**: On-policy learning with advantage estimation
- **Hyperparameters**: Learning rate = 0.0005, γ = 0.99, batch size = 1024

#### DQN (Deep Q-Network)
- **Architecture**: Deep Q-network with experience replay buffer
- **Network Structure**:
  - Input: 88-dimensional state vector
  - Hidden layers: 512-1024-512 neurons with ReLU activation
  - Output: 12-dimensional Q-value estimates
- **Key Features**: Target network, ε-greedy exploration, experience replay
- **Hyperparameters**: Learning rate = 0.0005, γ = 0.99, ε decay = 0.995, replay buffer = 10000

#### DDQN (Double DQN)
- **Improvement over DQN**: Decouples action selection from action evaluation to reduce overestimation
- **Network Structure**: Same as DQN with dual Q-network architecture
- **Key Features**: Double Q-learning, target network, experience replay
- **Hyperparameters**: Same as DQN with additional target network update frequency

### 3. Baseline Strategies

- **Local Processing**: All tasks are processed on the vehicle's local computing resources
  - Advantage: No transmission delay
  - Limitation: Limited by vehicle's CPU capacity and cache
  
- **Full Offloading**: All tasks are offloaded to the nearest available RSU
  - Advantage: Leverages powerful edge servers
  - Limitation: Subject to transmission delays and RSU availability
  
- **Random Strategy**: Tasks are randomly assigned to local processing or nearest RSU (50/50 split)
  - Purpose: Provides a baseline for comparing intelligent decision-making

### 4. Performance Metrics

- **Task Success Rate**: Proportion of tasks completed within their deadline constraints
- **Average Latency**: Mean time from task generation to completion
- **Queue Length**: Average number of tasks waiting in queues
- **Resource Utilization**: Cache usage and CPU utilization across nodes
- **Cumulative Reward**: Total reward accumulated during simulation

### 5. Reproducibility

All experiments can be reproduced using the provided scripts with fixed random seeds:
- Default seed: 42
- Task generation: Deterministic given seed
- Environment dynamics: Fully reproducible
- Training: Use `--seed` parameter to ensure consistent results

**Example:**
```bash
# Reproduce training with seed 42
python train.py --model a2c --episodes 100 --seed 42

# Reproduce evaluation
python evaluate.py --model a2c --runs 5 --seed 42

# Reproduce full dataset
python reproduce_simulation.py --episodes 100 --seed 42
```

---

## Citation

If you use this code or simulation environment in your research, please cite:

```bibtex
@software{vec_task_offloading_2025,
  title = {Vehicular Edge Computing Task Offloading System},
  author = {VEC Research Team},
  year = {2025},
  month = {12},
  version = {1.0.0},
  url = {https://github.com/your-repository/vec-task-offloading},
  note = {Implementation of deep reinforcement learning algorithms (A2C, DQN, DDQN) for VEC task offloading optimization}
}
```

**Dataset Citation:**
This project does not use any external public datasets. All experiments are conducted using the custom VEC simulation environment. The simulation can be reproduced using the provided `reproduce_simulation.py` script with the same random seed to ensure reproducibility.

**Related Publications:**
If you publish research using this codebase, please consider citing relevant papers on:
- Vehicular Edge Computing (VEC) and task offloading
- Deep reinforcement learning for resource allocation
- A2C, DQN, and DDQN algorithms

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
- **Issues**: Submit an issue on the GitHub repository

---

## Changelog

### v1.0.0 (2025-11-29)
- Initial release
- Implemented A2C, DQN, DDQN algorithms
- Provided pre-trained models and baseline comparisons
- Complete documentation and usage instructions

---

**Thank you for using this project!**

