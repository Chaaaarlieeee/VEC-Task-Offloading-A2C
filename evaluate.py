#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation Script for Reinforcement Learning Models
"""

import numpy as np
import torch
import os
import sys
import argparse
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment
from models import A2CAgent, DQNAgent, DDQNAgent
from baselines import run_local_baseline, run_offload_baseline, run_random_baseline
from utils import RLEnvironmentWrapper


def evaluate_rl_model(model_type='a2c', num_runs=5, seed=None):
    """Evaluate RL model"""
    print(f"\n===== Evaluating {model_type.upper()} Model =====")
    
    model_paths = [
        f'models/{model_type}_model.pth',
        f'../results/models/{model_type}_model.pth',
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Model found: {model_path}")
            break
    
    if model_path is None:
        print(f"Model not found, tried: {model_paths}")
        return None
    
    state_dim = 88
    action_dim = 12
    
    if model_type == 'a2c':
        agent = A2CAgent(state_dim, action_dim, hidden_dim=512)
    elif model_type == 'dqn':
        agent = DQNAgent(state_dim, action_dim, hidden_dim=512)
    elif model_type == 'ddqn':
        agent = DDQNAgent(state_dim, action_dim, hidden_dim=512)
    else:
        print(f"Unknown model type: {model_type}")
        return None
    
    agent.load_model(model_path)
    
    success_rates = []
    
    for run in range(num_runs):
        env = RLEnvironmentWrapper(simulation_time=None, seed=seed)
        
        state = env.reset()
        state = agent._preprocess_state(state)
        done = False
        
        while not done:
            if model_type == 'a2c':
                action, _, _ = agent.model.act(state)
            elif model_type in ['dqn', 'ddqn']:
                action = agent.act(state)
            else:
                action = agent.act(state)
            
            next_state, reward, done, info = env.step(action)
            next_state = agent._preprocess_state(next_state)
            state = next_state
        
        success_rates.append(info['success_rate'])
        print(f"Run {run+1}/{num_runs}: Success rate {info['success_rate']*100:.2f}%")
    
    avg_success_rate = np.mean(success_rates)
    print(f"\nAverage success rate: {avg_success_rate*100:.2f}%")
    
    return avg_success_rate


def compare_all_methods(num_runs=5, seed=42):
    """Compare all methods"""
    print("\n" + "="*60)
    print("Comparing All Methods (RL vs Baselines)")
    print("="*60)
    
    results = {}
    
    print("\n===== Local Processing Baseline =====")
    local_results = run_local_baseline(simulation_time=None, seed=seed, verbose=False)
    results['Local'] = local_results['success_rate']
    print(f"Success rate: {local_results['success_rate']*100:.2f}%")
    
    print("\n===== Full Offloading Baseline =====")
    offload_results = run_offload_baseline(simulation_time=None, seed=seed, verbose=False)
    results['Offload'] = offload_results['success_rate']
    print(f"Success rate: {offload_results['success_rate']*100:.2f}%")
    
    print("\n===== Random Strategy Baseline =====")
    random_results = run_random_baseline(simulation_time=None, seed=seed, verbose=False)
    results['Random'] = random_results['success_rate']
    print(f"Success rate: {random_results['success_rate']*100:.2f}%")
    
    for model_type in ['a2c', 'dqn', 'ddqn']:
        model_paths = [
            f'models/{model_type}_model.pth',
            f'../results/models/{model_type}_model.pth',
        ]
        model_exists = any(os.path.exists(p) for p in model_paths)
        
        if model_exists:
            success_rate = evaluate_rl_model(model_type, num_runs=1, seed=seed)
            if success_rate is not None:
                results[model_type.upper()] = success_rate
    
    plot_comparison(results)
    
    return results


def plot_comparison(results):
    """Plot comparison chart"""
    os.makedirs('../results/figures', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = list(results.keys())
    success_rates = [results[s] * 100 for s in strategies]
    
    colors = ['#4CAF50', '#2196F3', '#FFC107', '#FF5722', '#9C27B0', '#00BCD4']
    bars = ax.bar(strategies, success_rates, color=colors[:len(strategies)])
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Comparison of Task Offloading Strategies', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../results/figures/strategy_comparison.png', dpi=300)
    print(f"\nComparison chart saved to: ../results/figures/strategy_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--model', type=str, default='all', 
                        choices=['a2c', 'dqn', 'ddqn', 'all'],
                        help='Select model to evaluate (default: all)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all methods including baselines')
    
    args = parser.parse_args()
    
    if args.compare or args.model == 'all':
        compare_all_methods(num_runs=args.runs, seed=args.seed)
    else:
        evaluate_rl_model(args.model, num_runs=args.runs, seed=args.seed)


if __name__ == "__main__":
    main()
