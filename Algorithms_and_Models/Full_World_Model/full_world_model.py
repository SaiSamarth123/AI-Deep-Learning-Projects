# Full World Model

# Check out this link for the complete model explanation: https://worldmodels.github.io/

# Import necessary libraries
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define hyperparameters for the model
vision_model_hidden_dim = 256  # Hidden dimensions for the CNN output layer
memory_model_hidden_dim = 256  # Hidden dimensions for the LSTM layer
controller_hidden_dim = 256  # Not used explicitly in this implementation
action_dim = 3  # Number of actions for the environment
learning_rate = 1e-4  # Learning rate for optimizers
num_episodes = 500  # Number of episodes to train


# Define the Vision Model (using a Convolutional Neural Network)
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Define a fully connected layer to reduce dimensionality to the hidden dimension
        self.fc = nn.Linear(64 * 7 * 7, vision_model_hidden_dim)

    def forward(self, x):
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten and pass through the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


# Define the Memory Model (using a Recurrent Neural Network)
class MemoryModel(nn.Module):
    def __init__(self):
        super(MemoryModel, self).__init__()
        # LSTM layer to incorporate previous state memory
        self.lstm = nn.LSTM(
            input_size=vision_model_hidden_dim, hidden_size=memory_model_hidden_dim
        )

    def forward(self, x, hidden_state):
        # LSTM expects input of shape (seq_len, batch, input_size), hence unsqueeze
        x, hidden_state = self.lstm(x.unsqueeze(0), hidden_state)
        # Squeeze to remove sequence length dimension added by unsqueeze
        return x.squeeze(0), hidden_state


# Define the Controller Model (using a Fully Connected Neural Network)
class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        # Fully connected layer to produce action dimensions
        self.fc = nn.Linear(memory_model_hidden_dim, action_dim)

    def forward(self, x):
        # Linear layer to map from hidden state to action space
        x = self.fc(x)
        return x


# Function to preprocess states from the environment
def preprocess_state(state):
    # Normalize pixel values and adjust dimensions for PyTorch: NCHW format
    state = torch.from_numpy(state).float() / 255.0  # Normalize pixel values
    state = state.permute(2, 0, 1)  # Adjust dimensions to NCHW
    return state.unsqueeze(0)  # Add a batch dimension


# Initialize the environment
env = gym.make("CarRacing-v2")

# Create model instances
vision_model = VisionModel()
memory_model = MemoryModel()
controller = Controller()

# Create optimizers for each model component
vision_optimizer = optim.Adam(vision_model.parameters(), lr=learning_rate)
memory_optimizer = optim.Adam(memory_model.parameters(), lr=learning_rate)
controller_optimizer = optim.Adam(controller.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    hidden_state = None  # Initialize LSTM hidden state
    done = False
    while not done:
        state = preprocess_state(state)
        with torch.no_grad():
            vision_output = vision_model(state)
            # Pass the output of the vision model and hidden state to memory model
            memory_output, hidden_state = memory_model(
                vision_output,
                hidden_state
                or (
                    torch.zeros(1, 1, memory_model_hidden_dim),
                    torch.zeros(1, 1, memory_model_hidden_dim),
                ),
            )
            action = controller(memory_output)
        # Execute action in the environment
        next_state, reward, done, _ = env.step(action.numpy()[0])
        state = next_state
        # Note: Add training backpropagation and loss computation here if implementing training cycles
