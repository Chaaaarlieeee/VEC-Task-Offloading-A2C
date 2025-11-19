#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for Reinforcement Learning Models
"""

import numpy as np
import torch
import os
import sys
import argparse
import time

# Add code path
sys.path.insert(0, os.path.dirname(__file__))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment
from models import A2CAgent, DQNAgent, DDQNAgent
from utils import RLEnvironmentWrapper


def calculate_state_dim():
    """Calculate state space dimension: 88 total"""
    return 6 + 70 + 6 + 5 + 1  # vehicle(6) + RSUs(70) + transmission(6) + new_task(5) + time(1)


def create_environment(seed=None):
    """Create training environment with wrapper"""
    env = RLEnvironmentWrapper(simulation_time=None, seed=seed)
    return env


def train_a2c(args):
    """Train A2C model"""
    print("="*50)
    print("Training A2C Model")
    print("="*50)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    env = create_environment(seed=args.seed)
    state_dim = calculate_state_dim()
    action_dim = 12  # local(0) + RSUs(1-10) + skip(11)
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr,
        lr_critic=args.lr,
        gamma=args.gamma
    )
    
    try:
        print("Training started...")
        start_time = time.time()
        rewards, lengths, losses = agent.train(
            env=env,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            update_frequency=args.update_freq,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f}s")
        
        os.makedirs('../results/models', exist_ok=True)
        model_path_results = f'../results/models/a2c_model.pth'
        agent.save_model(model_path_results)
        print(f"Model saved to {model_path_results}")
        
        model_path_local = 'models/a2c_model.pth'
        agent.save_model(model_path_local)
        print(f"Model saved to {model_path_local}")
        
        return agent
    except Exception as e:
        print(f"Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def train_dqn(args):
    """Train DQN model"""
    print("="*50)
    print("Training DQN Model")
    print("="*50)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    env = create_environment(seed=args.seed)
    state_dim = calculate_state_dim()
    action_dim = 12
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    print("Training started...")
    start_time = time.time()
    rewards, lengths, losses = agent.train(
        env=env,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        update_frequency=args.update_freq
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f}s")
    
    os.makedirs('../results/models', exist_ok=True)
    model_path_results = f'../results/models/dqn_model.pth'
    agent.save_model(model_path_results)
    print(f"Model saved to {model_path_results}")
    
    model_path_local = 'models/dqn_model.pth'
    agent.save_model(model_path_local)
    print(f"Model saved to {model_path_local}")
    
    return agent


def train_ddqn(args):
    """Train DDQN model"""
    print("="*50)
    print("Training DDQN Model")
    print("="*50)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    env = create_environment(seed=args.seed)
    state_dim = calculate_state_dim()
    action_dim = 12
    
    agent = DDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay
    )
    
    print("Training started...")
    print(f"Model saved")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train RL models')
    parser.add_argument('--model', type=str, default='a2c', choices=['a2c', 'dqn', 'ddqn'],
                        help='Model type (default: a2c)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes (default: 100)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Hidden dimension (default: 512)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--update-freq', type=int, default=10,
                        help='Update frequency (default: 10)')
    parser.add_argument('--epsilon-start', type=float, default=0.3,
                        help='Initial epsilon (default: 0.3)')
    parser.add_argument('--epsilon-end', type=float, default=0.01,
                        help='Final epsilon (default: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay (default: 0.995)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.model == 'a2c':
        train_a2c(args)
    elif args.model == 'dqn':
        train_dqn(args)
    elif args.model == 'ddqn':
        train_ddqn(args)
    else:
        print(f"Unknown model type: {args.model}")


if __name__ == "__main__":
    main()

