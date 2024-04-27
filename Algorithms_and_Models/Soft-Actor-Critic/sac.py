# Soft Actor-Critic

# Check out this link for the complete model explanation: https://spinningup.openai.com/en/latest/algorithms/sac.html

# Import necessary libraries
import gymnasium as gym  # Used for the environment
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque  # For storing steps in the replay buffer

# Set hyperparameters for the learning model
state_dim = env.observation_space.shape[0]  # State space dimensionality
action_dim = env.action_space.shape[0]  # Action space dimensionality
hidden_dim = 256  # Number of neurons in hidden layers
actor_lr = 3e-4  # Learning rate for actor
critic_lr = 3e-4  # Learning rate for critic
alpha_lr = 3e-4  # Learning rate for adjusting the entropy coefficient
gamma = 0.99  # Discount factor for future rewards
tau = 0.005  # Smoothing factor for soft update of target networks
buffer_size = 1e6  # Size of the replay buffer
batch_size = 128  # Number of samples per minibatch
alpha = 0.2  # Initial entropy coefficient


# Implementing the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch from the buffer
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# Building the Actor Network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(
            log_std, min=-20, max=2
        )  # Clamp the values of log standard deviation to avoid numerical instability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()  # Standard deviation
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)  # Squash gaussian to be between -1 and 1
        return action


# Building the Critic Network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Building the SAC Agent
class SACAgent:
    def __init__(self):
        self.actor = Actor()
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor.sample(state)
        return action.detach().numpy
