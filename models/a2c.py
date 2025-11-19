import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
import random
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    """
    Actor-Critic model with policy network (Actor) and value network (Critic)
    Enhanced version: deeper and wider architecture with layer normalization
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim*2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, 1)
        )
        
        self._init_weights()
        self.to(device)
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):


        if len(state.shape) == 1:
            state = state.unsqueeze(0)  
        

        features = self.feature_layer(state)
        action_probs = self.actor(features)
        

        action_probs = action_probs + 1e-10
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_probs, state_value = self.forward(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), action_probs[0, action.item()].item(), state_value.item()

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

class A2CAgent:

    def __init__(self, state_dim, action_dim, hidden_dim=128, lr_actor=0.001, lr_critic=0.001, gamma=0.99):

        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr_actor)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.max_grad_norm = 0.5
        
    def update(self, states, actions, rewards, next_states, dones):

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        action_probs, state_values = self.model(states)
        
        with torch.no_grad():
            _, next_state_values = self.model(next_states)
            next_state_values = next_state_values.squeeze()
        
        targets = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = targets - state_values.squeeze()
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-10)
        
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(state_values.squeeze(), targets.detach())
        loss = actor_loss + critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, env, num_episodes, batch_size=32, update_frequency=10, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):

        replay_buffer = ReplayBuffer(100000)
        epsilon = epsilon_start
        total_steps = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            state = self._preprocess_state(state)
            episode_reward = 0
            episode_loss = 0
            done = False
            step = 0
            
            while not done:
                if random.random() < epsilon:
                    action = random.randint(0, env._get_action_space() - 1)
                    _, _, _ = self.model.act(state)
                else:
                    action, _, _ = self.model.act(state)
                
                next_state, reward, done, _ = env.step(action)
                next_state = self._preprocess_state(next_state)
                
                replay_buffer.push(state, action, reward, next_state, float(done))
                
                state = next_state
                episode_reward += reward
                step += 1
                total_steps += 1
                
                if len(replay_buffer) >= batch_size and total_steps % update_frequency == 0:
                    batch = replay_buffer.sample(batch_size)
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
                    loss = self.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
                    episode_loss += loss
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            self.episode_losses.append(episode_loss / step if step > 0 else 0)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                avg_loss = np.mean(self.episode_losses[-10:])
                print(f"Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.2f}")
        
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
        plt.savefig('image/a2c_training_results.png', dpi=300)
        plt.show()
    
    def save_model(self, path):

        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):

        map_location = device if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:

            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:

            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)