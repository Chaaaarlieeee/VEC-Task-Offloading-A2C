import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),  
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),  
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),  
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, action_dim) 
        )
        

        self._init_weights()
        
       
        self.to(device)
        
    def _init_weights(self):
      
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):

        if len(state.shape) == 1:
            state = state.unsqueeze(0)  
        

        q_values = self.q_network(state)
        
        return q_values
    
    def act(self, state, epsilon=0.0):
     
        if random.random() < epsilon:
    
            action_dim = self.q_network[-1].out_features
            return random.randint(0, action_dim - 1)
        else:
          
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.forward(state)
                action = q_values.argmax(dim=1).item()
            return action

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
 
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
 
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99, 
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
        
        self.q_network = DQN(state_dim, action_dim, hidden_dim)
        self.target_network = DQN(state_dim, action_dim, hidden_dim)
        
   
        self.target_network.load_state_dict(self.q_network.state_dict())
        
       
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
    
        self.replay_buffer = ReplayBuffer(buffer_size)
        

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        
  
        self.max_grad_norm = 0.5
        

        self.update_count = 0

    def act(self, state):

        return self.q_network.act(state, self.epsilon)
    
    def remember(self, state, action, reward, next_state, done):

        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=32):
       
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
       
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch
        
   
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
       
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
      
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
       
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
      
        self.optimizer.zero_grad()
        loss.backward()
        
    
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        

        self.update_count += 1
        
 
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
  
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train(self, env, num_episodes, batch_size=1024, update_frequency=4):
       
        total_steps = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            state = self._preprocess_state(state)
            episode_reward = 0
            episode_loss = 0
            done = False
            step = 0
            
            while not done:
    
                action = self.act(state)
                
          
                next_state, reward, done, _ = env.step(action)
                next_state = self._preprocess_state(next_state)
                
            
                self.remember(state, action, reward, next_state, done)
                
               
                state = next_state
                episode_reward += reward
                step += 1
                total_steps += 1
                
      
                if total_steps % update_frequency == 0:
                    loss = self.update(batch_size)
                    episode_loss += loss
            
           
            self.decay_epsilon()
            
       
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            self.episode_losses.append(episode_loss / max(step // update_frequency, 1))
            
          
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.episode_losses[-10:])
                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.2f}")
        
 
        self._plot_training_results()
        
        return self.episode_rewards, self.episode_lengths, self.episode_losses
    
    def _preprocess_state(self, state_dict):
       

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
    
    def _plot_training_results(self):

        plt.figure(figsize=(15, 5))
        

        plt.subplot(1, 3, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
 
        plt.subplot(1, 3, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        

        plt.subplot(1, 3, 3)
        plt.plot(self.episode_losses)
        plt.title('Episode Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('image/dqn_training_results.png', dpi=300)
        plt.show()
    
    def save_model(self, path):

        torch.save(self.q_network.state_dict(), path)
    
    def load_model(self, path):

 
        map_location = device if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
    
        if isinstance(checkpoint, dict) and 'q_network_state_dict' in checkpoint:
    
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            if 'target_network_state_dict' in checkpoint:
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
        else:
      
            self.q_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.q_network.to(device)
        self.target_network.to(device) 