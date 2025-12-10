#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproduction Script for Vehicular Edge Computing Simulation Dataset

This script allows researchers to reproduce the simulation dataset used in our experiments.
The simulation generates synthetic task offloading scenarios in a vehicular edge computing environment.

Usage:
    python reproduce_simulation.py --seed 42 --episodes 100 --output ./data/simulation_results.pkl

Author: VEC Task Offloading Research Team
Date: 2025-12-10
"""

import argparse
import os
import sys
import pickle
import numpy as np
import json
from datetime import datetime

# Add code path
sys.path.insert(0, os.path.dirname(__file__))

from environment import Vehicle, RSU, TaskPoolGenerator, Communication, VehicularEdgeEnvironment
from utils import RLEnvironmentWrapper


def create_default_environment(seed=42):
    """
    Create the default VEC environment with standard configuration
    
    Args:
        seed (int): Random seed for reproducibility
    
    Returns:
        VehicularEdgeEnvironment: Configured environment instance
    """
    np.random.seed(seed)
    
    # Create RSUs (Road Side Units)
    rsus = []
    for i in range(10):
        rsus.append(RSU(rsu_id=i+1, position=250 + i*500))
    
    # Create task generator
    task_generator = TaskPoolGenerator(seed=seed)
    task_generator.generate_task_pool(100)
    
    # Create vehicle
    vehicle = Vehicle(position=0, cpu=2.5, speed=20, use_idm=True)
    
    # Create communication module
    communication = Communication()
    
    # Create environment
    env = VehicularEdgeEnvironment(
        vehicle=vehicle,
        rsus=rsus,
        task_generator=task_generator,
        communication=communication,
        simulation_time=None
    )
    
    return env


def run_simulation_episode(env, episode_id, seed):
    """
    Run a single simulation episode and collect data
    
    Args:
        env: Environment instance
        episode_id (int): Episode number
        seed (int): Random seed
        
    Returns:
        dict: Episode data including states, actions, and outcomes
    """
    np.random.seed(seed + episode_id)
    
    state = env.reset()
    episode_data = {
        'episode_id': episode_id,
        'seed': seed + episode_id,
        'states': [],
        'actions': [],
        'rewards': [],
        'task_outcomes': [],
        'statistics': {}
    }
    
    step = 0
    done = False
    total_reward = 0
    
    while not done:
        # Record state
        episode_data['states'].append(state)
        
        # Random action for data collection (can be replaced with policy)
        action = np.random.randint(0, 12)
        episode_data['actions'].append(action)
        
        # Execute action
        next_state, reward, done, info = env.step(action)
        
        # Record reward and outcome
        episode_data['rewards'].append(reward)
        total_reward += reward
        
        state = next_state
        step += 1
        
        # Prevent infinite loops
        if step > 10000:
            break
    
    # Record final statistics
    episode_data['statistics'] = {
        'total_steps': step,
        'total_reward': total_reward,
        'total_tasks': info['total_tasks'],
        'successful_tasks': info['successful_tasks'],
        'failed_tasks': info['failed_tasks'],
        'success_rate': info['success_rate']
    }
    
    # Record task outcomes
    episode_data['task_outcomes'] = [
        {
            'task_id': task.id,
            'size': task.size,
            'complexity': task.complexity,
            'deadline': task.deadline,
            'success': task.success,
            'processing_time': task.processing_time if task.processing_time else None,
            'location': task.location
        }
        for task in env.completed_tasks
    ]
    
    return episode_data


def generate_simulation_dataset(num_episodes=100, seed=42, output_dir='./data'):
    """
    Generate complete simulation dataset
    
    Args:
        num_episodes (int): Number of episodes to simulate
        seed (int): Base random seed
        output_dir (str): Output directory for dataset
    
    Returns:
        dict: Complete dataset with all episodes
    """
    print("="*70)
    print("Vehicular Edge Computing Simulation Dataset Generation")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Random seed: {seed}")
    print(f"  Output directory: {output_dir}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = {
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'num_episodes': num_episodes,
            'base_seed': seed,
            'environment_config': {
                'num_rsus': 10,
                'rsu_coverage': 250,  # meters
                'vehicle_speed': 20,  # m/s
                'vehicle_cpu': 2.5,   # GHz
                'task_pool_size': 100,
                'time_slot': 0.1,     # seconds
                'task_generation_prob': 0.15
            }
        },
        'episodes': []
    }
    
    # Run episodes
    for episode_id in range(num_episodes):
        print(f"Running episode {episode_id + 1}/{num_episodes}...", end=' ')
        
        # Create fresh environment for each episode
        env = create_default_environment(seed=seed)
        
        # Run episode
        episode_data = run_simulation_episode(env, episode_id, seed)
        dataset['episodes'].append(episode_data)
        
        # Print statistics
        stats = episode_data['statistics']
        print(f"Steps: {stats['total_steps']}, "
              f"Success Rate: {stats['success_rate']*100:.2f}%, "
              f"Reward: {stats['total_reward']:.2f}")
    
    # Calculate overall statistics
    overall_stats = {
        'total_episodes': len(dataset['episodes']),
        'avg_success_rate': np.mean([ep['statistics']['success_rate'] 
                                     for ep in dataset['episodes']]),
        'avg_total_tasks': np.mean([ep['statistics']['total_tasks'] 
                                    for ep in dataset['episodes']]),
        'avg_reward': np.mean([ep['statistics']['total_reward'] 
                              for ep in dataset['episodes']])
    }
    dataset['overall_statistics'] = overall_stats
    
    print("\n" + "="*70)
    print("Overall Statistics:")
    print(f"  Average Success Rate: {overall_stats['avg_success_rate']*100:.2f}%")
    print(f"  Average Total Tasks: {overall_stats['avg_total_tasks']:.2f}")
    print(f"  Average Reward: {overall_stats['avg_reward']:.2f}")
    print("="*70)
    
    return dataset


def save_dataset(dataset, output_dir, format='pickle'):
    """
    Save dataset to disk
    
    Args:
        dataset (dict): Dataset to save
        output_dir (str): Output directory
        format (str): File format ('pickle' or 'json')
    """
    if format == 'pickle':
        output_path = os.path.join(output_dir, 'simulation_dataset.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nDataset saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    elif format == 'json':
        # Save metadata and statistics only (JSON doesn't handle numpy arrays well)
        output_path = os.path.join(output_dir, 'simulation_metadata.json')
        metadata = {
            'metadata': dataset['metadata'],
            'overall_statistics': dataset['overall_statistics'],
            'episode_statistics': [ep['statistics'] for ep in dataset['episodes']]
        }
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to: {output_path}")
    
    # Always save a summary
    summary_path = os.path.join(output_dir, 'dataset_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Vehicular Edge Computing Simulation Dataset Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Creation Date: {dataset['metadata']['creation_date']}\n")
        f.write(f"Number of Episodes: {dataset['metadata']['num_episodes']}\n")
        f.write(f"Base Random Seed: {dataset['metadata']['base_seed']}\n\n")
        f.write("Environment Configuration:\n")
        for key, value in dataset['metadata']['environment_config'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\nOverall Statistics:\n")
        for key, value in dataset['overall_statistics'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "="*70 + "\n")
    
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Reproduce simulation dataset for VEC task offloading experiments'
    )
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to simulate (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='./data',
                        help='Output directory for dataset (default: ./data)')
    parser.add_argument('--format', type=str, default='pickle',
                        choices=['pickle', 'json', 'both'],
                        help='Output format (default: pickle)')
    
    args = parser.parse_args()
    
    # Generate dataset
    dataset = generate_simulation_dataset(
        num_episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output
    )
    
    # Save dataset
    if args.format == 'both':
        save_dataset(dataset, args.output, format='pickle')
        save_dataset(dataset, args.output, format='json')
    else:
        save_dataset(dataset, args.output, format=args.format)
    
    print("\n" + "="*70)
    print("Dataset generation completed successfully!")
    print("="*70)
    print("\nYou can now use this dataset to:")
    print("  1. Train reinforcement learning models")
    print("  2. Evaluate baseline strategies")
    print("  3. Analyze task offloading patterns")
    print("  4. Reproduce experimental results")
    print("\nFor more information, see the README.md file.")
    print("="*70)


if __name__ == "__main__":
    main()
