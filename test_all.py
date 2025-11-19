#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Test Script
Verify all modules and functionality
"""

import sys
import os

# Add code path
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("Quick Test Script")
print("="*70)

# Test 1: Environment module import
print("\n[1/5] Testing environment module import...")
try:
    from environment import Vehicle, RSU, Task, TaskPoolGenerator, Communication, VehicularEdgeEnvironment
    print("[OK] Environment module imported successfully")
except Exception as e:
    print(f"[FAIL] Environment module import failed: {e}")
    sys.exit(1)

# Test 2: Model module import
print("\n[2/5] Testing model module import...")
try:
    from models import A2CAgent, DQNAgent, DDQNAgent
    print("[OK] Model module imported successfully")
except Exception as e:
    print(f"[FAIL] Model module import failed: {e}")
    sys.exit(1)

# Test 3: Baseline module import
print("\n[3/5] Testing baseline module import...")
try:
    from baselines import run_local_baseline, run_offload_baseline, run_random_baseline
    print("[OK] Baseline module imported successfully")
except Exception as e:
    print(f"[FAIL] Baseline module import failed: {e}")
    sys.exit(1)

# Test 4: Utility module import (RLEnvironmentWrapper)
print("\n[4/5] Testing utility module import (RLEnvironmentWrapper)...")
try:
    from utils import RLEnvironmentWrapper
    print("[OK] Utility module imported successfully")
except Exception as e:
    print(f"[FAIL] Utility module import failed: {e}")
    sys.exit(1)

# Test 5: Create environment and test
print("\n[5/5] Testing environment creation and functionality...")
try:
    # Create environment wrapper
    env = RLEnvironmentWrapper(simulation_time=10, seed=42)
    
    # Test reset
    state = env.reset()
    print(f"  [+] Environment reset successful")
    
    # Test get action space
    action_space = env._get_action_space()
    print(f"  [+] Action space size: {action_space}")
    
    # Test step
    next_state, reward, done, info = env.step(0)
    print(f"  [+] Step execution successful (reward: {reward:.2f})")
    
    print("[OK] Environment functionality test passed")
except Exception as e:
    print(f"[FAIL] Environment functionality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check model files
print("\n[6/6] Checking model files...")
model_dir = 'models'
model_files = ['a2c_model.pth', 'dqn_model.pth', 'ddqn_model.pth']

found_models = []
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  [+] Found: {model_file} ({size_mb:.2f} MB)")
        found_models.append(model_file)
    else:
        print(f"  [!] Not found: {model_file}")

if found_models:
    print(f"[OK] Found {len(found_models)} model file(s)")
else:
    print("[!] No trained models found (normal if you haven't trained yet)")

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print("[OK] All core functionality tests passed!")
print("\nAvailable commands:")
print("  1. Train model:     python train.py --model a2c --episodes 50")
print("  2. Evaluate model:  python evaluate.py --model a2c --runs 3")
print("  3. Compare all:     python evaluate.py --compare")
print("  4. Full workflow:   python run")
print("="*70)

