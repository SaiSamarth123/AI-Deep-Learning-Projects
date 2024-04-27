# Deep Deterministic Policy Gradient (DDPG)

# Check out this link for the complete model explanation: https://spinningup.openai.com/en/latest/algorithms/ddpg.html

# Import necessary libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set hyperparameters for the DDPG model
state_dim = 96 * 96 * 3  # Dimension of state space
action_dim = 3           # Dimension of action space
hidden_dim = 256         # Number of nodes in hidden layers
batch_size = 128         # Size of batch taken from replay buffer
learning_rate = 1e-4     # Learning rate for optimizers
gamma = 0.99             # Discount factor for future rewards
tau = 0.005              # Soft update parameter
buffer_size = int(1e5)   # Maximum size of buffer
min_buffer_size = 1000   # Minimum buffer size before starting training
num_episodes = 500       # Number of episodes to train for
max_timesteps = 1000     # Max steps in one episode

# Actor network defines policy function from state to action
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Use tanh to output values between -1 and 1
        return action

# Critic network estimates the value of state-action pairs
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))  # Combine state and action
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# ReplayBuffer stores experience tuples and samples from them for training
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# DDPG Agent: combines the actor and critic into a model for interacting with the environment
class DDPG:
    def __init__(self):
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(buffer_size)

    def update(self, batch_size):
        if len(self.replay_buffer) < min_buffer_size:
            return  # Wait until buffer is filled to the minimum size
        
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = map(torch.tensor, zip(*samples))
        
        # Update critic by minimizing the loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor using sampled policy gradient
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    @staticmethod
    def soft_update(local_model, target_model):
        for
