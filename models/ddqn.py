#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DDQN using device: {device}")

class DDQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(DDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        

        self._init_weights()
    
    def _init_weights(self):

        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):

        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(device)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

class DDQNAgent:

    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=0.0005, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 target_update_frequency=100, buffer_size=100000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_frequency = target_update_frequency
        

        self.q_network = DDQN(state_dim, action_dim, hidden_dim).to(device)
        self.target_network = DDQN(state_dim, action_dim, hidden_dim).to(device)
        

        self.update_target_network()
        

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        

        self.replay_buffer = deque(maxlen=buffer_size)
        

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.update_count = 0
        
        print(f"DDQN agent initialized:")
        print(f"  - State dim: {state_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Learning rate: {lr}")
        print(f"  - Gamma: {gamma}")
        print(f"  - Buffer size: {buffer_size}")
    
    def _preprocess_state(self, state_dict):


        if isinstance(state_dict, np.ndarray):
            return np.array(state_dict, dtype=np.float32)
        

        vehicle_info = [
            state_dict['vehicle']['cpu'] / 2.5,  
            state_dict['vehicle']['speed'] / 25.0,  
            state_dict['vehicle']['cache_available'] / (10 * 1024 * 1024), 
            min(state_dict['vehicle']['queue_length'] / 5.0, 1.0),  
            min(state_dict['vehicle']['processing_length'] / 1.0, 1.0), 
            state_dict['vehicle'].get('transmission_rate', 0.0) / (25 * 1024 * 1024) 
        ]


        rsu_info = []
        for rsu in state_dict['rsus']:
           
            distance = abs(rsu.get('distance', rsu['position'] - state_dict['vehicle']['position']))
            coverage_radius = rsu.get('coverage_radius', 250) 
            in_coverage = 1.0 if distance <= coverage_radius else 0.0
            rsu_info.extend([
                min(distance / coverage_radius, 1.0),  
                in_coverage, 
                rsu.get('cpu', rsu.get('computing_capability', 0.0)) / 7.0,  
                rsu['cache_available'] / (25 * 1024 * 1024), 
                min(rsu.get('queue_length', 0) / 10.0, 1.0),  
                 min(rsu.get('processing_length', rsu.get('current_tasks_count', 0)) / 2.0, 1.0),
                1.0 if rsu.get('failure', 0) == 0 else 0.0 
            ])
        

        transmitting_info = [] 
        
        if 'transmitting_tasks' in state_dict:

            if state_dict['transmitting_tasks'].get('current_task'):
                task = state_dict['transmitting_tasks']['current_task']
                transmitting_info.extend([
                    min(task['size'] / (3 * 1024 * 1024), 1.0),
                    min(task['complexity'] / 500.0, 1.0),  
                    float(task['priority']), 
                    min(task.get('time_to_deadline', 0.0) / 10.0, 1.0), 
                    min(task.get('transmitted_data', 0.0) / (task['size'] + 1e-6), 1.0)  
                ])
            else:
               
                transmitting_info.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                

            queue_length = state_dict['transmitting_tasks'].get('queue_length', 0)
            transmitting_info.append(min(queue_length / 5.0, 1.0)) 
        else:
          
            transmitting_info.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        

        task_info = [0.0, 0.0, 0.0, 0.0, 0.0]  
        if state_dict['new_task']:
            task = state_dict['new_task']
            task_info = [
                min(task['size'] / (10 * 1024 * 1024), 1.0),  
                min(task['complexity'] / 500.0, 1.0),
                float(task['priority']),  
                min(task.get('deadline', task.get('time_to_deadline', 0.0)) / 10.0, 1.0), 
                min(max(task.get('time_to_deadline', 0.0) / 10.0, 0.0), 1.0)  
            ]
        

        state_vector = (
            vehicle_info + 
            rsu_info + 
            transmitting_info + 
            task_info + 
            [min(state_dict['current_time'] / 100.0, 1.0)] 
        )
        

        for i, val in enumerate(state_vector):
            if np.isnan(val) or np.isinf(val):
                state_vector[i] = 0.0  
        
        return np.array(state_vector, dtype=np.float32)
    
    def act(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state = self._preprocess_state(state)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
  
        state = self._preprocess_state(state)
        next_state = self._preprocess_state(next_state)
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update(self, batch_size=32):

        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)
        
  
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
     
        with torch.no_grad():
      
            next_actions = self.q_network(next_states).argmax(1)
      
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
  
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
  
        self.optimizer.zero_grad()
        loss.backward()
        

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        

        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
 
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
   
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath):

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'update_count': self.update_count
        }, filepath)
        print(f"DDQN model saved to: {filepath}")
    
    def load_model(self, filepath):
 
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
            
        
            if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
           
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
                self.episode_rewards = checkpoint.get('episode_rewards', [])
                self.episode_lengths = checkpoint.get('episode_lengths', [])
                self.episode_losses = checkpoint.get('episode_losses', [])
                self.update_count = checkpoint.get('update_count', 0)
                
                print(f"DDQN model loaded from {filepath}")
                print(f"  - Current epsilon: {self.epsilon:.4f}")
                print(f"  - Update count: {self.update_count}")
            else:

                self.q_network.load_state_dict(checkpoint)
                self.target_network.load_state_dict(self.q_network.state_dict())
                print(f"DDQN model loaded from {filepath} (simple format)")
            
            self.q_network.to(device)
            self.target_network.to(device)
            return True
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_training_stats(self):

        if not self.episode_rewards:
            return None
        
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'final_reward': self.episode_rewards[-1],
            'avg_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_loss': np.mean(self.episode_losses) if self.episode_losses else 0,
            'current_epsilon': self.epsilon
        }
    
    def plot_training_results(self):

        if not self.episode_rewards:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DDQN Training Results', fontsize=16)
        

        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        

        if len(self.episode_rewards) > 10:
            window_size = min(50, len(self.episode_rewards) // 5)
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(self.episode_rewards)), moving_avg)
            axes[0, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Reward')
            axes[0, 1].grid(True)
        

        if self.episode_lengths:
            axes[1, 0].plot(self.episode_lengths)
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True)
        

        if self.episode_losses:
            axes[1, 1].plot(self.episode_losses)
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        

        stats = self.get_training_stats()
        if stats:
            print("\n" + "="*50)
            print("DDQN Training Statistics")
            print("="*50)
            print(f"Total episodes: {stats['episodes']}")
            print(f"Average reward: {stats['avg_reward']:.2f}")
            print(f"Max reward: {stats['max_reward']:.2f}")
            print(f"Min reward: {stats['min_reward']:.2f}")
            print(f"Final reward: {stats['final_reward']:.2f}")
            print(f"Average episode length: {stats['avg_length']:.2f}")
            print(f"Average training loss: {stats['avg_loss']:.4f}")
            print(f"Current epsilon: {stats['current_epsilon']:.4f}")
            print("="*50)