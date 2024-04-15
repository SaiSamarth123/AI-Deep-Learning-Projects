# Deep Q-Learning for Lunar Landing

## Project Overview
This project demonstrates the implementation of a Deep Q-Learning (DQL) model to train an agent to land on the moon in the "Lunar Lander" simulation from Gymnasium. The agent is rewarded for successful landings and penalized for crashes, guiding it to learn optimal landing strategies.

## Deep Q-Learning Model
### Components
- **Experience Replay**: Stores transitions in a replay buffer to break the correlation between sequential experiences, using them to train the network in batches.
- **Q-Network**: A neural network that approximates the Q-value function, predicting the value of action-state pairs.
- **Target Network**: A clone of the Q-network, providing stable targets for training updates, helping to prevent divergence.
- **Exploration vs. Exploitation**: Utilizes an ε-greedy strategy that balances exploring new actions with exploiting known ones, where ε decreases over time.

## Implementation
### Setup
- **Environment**: The Gymnasium 'LunarLander-v2' environment.
- **Neural Network**: Defined with three linear layers.
- **Training**: Uses the Adam optimizer, with a learning rate of 5e-4 and a replay buffer for experience replay.
- **Agent Actions**: Decisions are made based on the ε-greedy policy, with training occurring every four steps.

### Training Process
- **Episodes**: Runs for up to 2000 episodes or until the agent consistently achieves a high score.
- **Epsilon Decay**: Reduces exploration rate from 1.0 to 0.01 to encourage exploitation of the learned policy over time.

## Results and Observations
The training progress is monitored by the average scores per episode, aiming for an average score of 200 over 100 episodes to consider the environment solved. Successful training sessions save the model parameters.



### Here's a video demonstration of the lunar landing using Deep Q Learning:

#### First Iteration of the model
![Video](<https://raw.githubusercontent.com/SaiSamarth123/AI-Deep-Learning-Projects/main/Deep_Q_Learning/Lunar_Landing(1).gif>)

#### Second Iteration of the model
![Video](<https://raw.githubusercontent.com/SaiSamarth123/AI-Deep-Learning-Projects/main/Deep_Q_Learning/Lunar_Landing(2).gif>)

#### Third Iteration of the model
![Video](<https://raw.githubusercontent.com/SaiSamarth123/AI-Deep-Learning-Projects/main/Deep_Q_Learning/Lunar_Landing(3).gif>)
