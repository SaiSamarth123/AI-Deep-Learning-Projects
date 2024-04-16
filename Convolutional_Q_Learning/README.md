# Deep Convolutional Q-Learning for Pac-Man

## Project Overview

This project applied Deep Convolutional Q-Learning (DCQL) to the "MsPacmanDeterministic-v0" environment from Gymnasium. The goal was to train an agent using visual inputs to master the game of Pac-Man, making strategic decisions based on the current game state represented as pixel data.

## Deep Convolutional Q-Learning Model

### Components

- **Convolutional Neural Network**: Utilizes layers of convolution to process state images, extracting features that are crucial for decision-making.
- **Experience Replay**: Stores transitions between states to minimize correlations and improve learning stability.
- **Target Network**: Provides a stable target for the Q-learning updates, updating its weights slowly compared to the primary network.
- **Epsilon-Greedy Strategy**: Balances exploration and exploitation, gradually favoring the best-known actions as training progresses.

## Implementation

### Setup

- **Environment Initialization**: Configuration of the Gym environment to provide visual states.
- **Preprocessing**: Frames are processed and scaled down to reduce computational requirements and focus learning on relevant features.
- **Network Training**: The agent learns by interacting with the environment, storing experiences, and periodically learning from mini-batches of these experiences.

### Training Process

- **Episodes**: Configured to run for a maximum of 2000 episodes, with an epsilon decay strategy to reduce the exploration rate over time.
- **Learning**: Updates the policy network using the loss between predicted Q-values and the target Q-values derived from the target network.

## Results and Observations

Training is assessed by tracking the score per episode, aiming for a consistently high score to demonstrate mastery over the environment.

## Conclusion

The DCQL model showcases an advanced application of reinforcement learning, using visual inputs directly to make decisions. This project highlights the potential of convolutional neural networks in understanding and acting in complex, dynamic environments like video games.

#### Here's a video demonstration of Pacman using Convolutional Learning:

![Video](https://raw.githubusercontent.com/SaiSamarth123/AI-Deep-Learning-Projects/main/Convolutional_Q_Learning/Pacman.gif)
