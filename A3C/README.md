# A3C for Kung Fu

## Project Overview

This project implements the Asynchronous Advantage Actor-Critic (A3C) algorithm in the "KungFuMasterDeterministic-v0" environment from Gymnasium. A3C is a type of reinforcement learning algorithm that uses multiple copies of the environment to speed up training and improve stability.

## A3C Model

### Components

- **Neural Network**: Comprises convolutional layers to process state inputs, followed by dense layers that output both action probabilities and state value estimates.
- **Parallel Environments**: Multiple instances of the environment run concurrently to gather diverse experience and accelerate learning.
- **Advantage Estimation**: Calculates the advantage of actions taken, based on the difference between the predicted state values and the actual rewards received.

## Implementation

### Setup

- **Environment Wrapping**: Processes and normalizes the input frames from the game for more efficient learning by the neural network.
- **Action and Training Loops**: The agent selects actions based on policy output, interacts with the environment, and updates the network weights using gradients computed from the advantage.

### Training Process

- **Episodes and Updates**: Runs for a specified number of episodes or until a performance criterion is met, with network updates occurring at each step based on batches of experience.

## Results and Observations

- **Performance Metrics**: Monitors the average rewards and episode lengths to evaluate the agent's performance over time.
- **Model Saving**: Saves the model parameters periodically or when the agent achieves a performance threshold.

## Conclusion

This A3C implementation demonstrates the use of advanced deep reinforcement learning techniques to efficiently solve complex control tasks like the Kung Fu game, highlighting the benefits of parallel computation and policy gradient methods.

### Here's a video demonstration of Kung Fu using an A3C model:

![Video](https://raw.githubusercontent.com/SaiSamarth123/AI-Deep-Learning-Projects/main/A3C/KungFu.gif)
