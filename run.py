#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Execution Script for Code Ocean
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("="*70)
    print("Vehicular Edge Computing Task Offloading System")
    print("="*70)
    print()
    
    # Create results directories
    results_base = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(os.path.join(results_base, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_base, 'figures'), exist_ok=True)
    
    print("\n" + "="*70)
    print("Step 1: Training A2C Model")
    print("="*70)
    
    train_cmd = [
        sys.executable, 'train.py',
        '--model', 'a2c',
        '--episodes', '50',
        '--hidden-dim', '512',
        '--lr', '0.0005',
        '--seed', '42'
    ]
    
    result = subprocess.run(train_cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print("\nWarning: Training encountered issues, continuing to evaluation...")
    
    print("\n" + "="*70)
    print("Step 2: Evaluating All Methods")
    print("="*70)
    
    eval_cmd = [
        sys.executable, 'evaluate.py',
        '--compare',
        '--runs', '3',
        '--seed', '42'
    ]
    
    result = subprocess.run(eval_cmd, cwd=os.path.dirname(__file__))
    
    if result.returncode != 0:
        print("\nWarning: Evaluation encountered issues...")
    
    print("\n" + "="*70)
    print("Execution completed!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Models: results/models/")
    print("  - Figures: results/figures/")
    print("="*70)


if __name__ == "__main__":
    main()

