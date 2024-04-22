# Google Stock Price Prediction with RNN

## Project Overview

This project involves building a Recurrent Neural Network (RNN) using LSTM (Long Short-Term Memory) units to predict Google's stock price. The model is trained on historical stock prices to forecast future values, demonstrating the application of deep learning in financial time series analysis.

## Key Concepts

- **RNN and LSTM**: Utilizes Recurrent Neural Networks and Long Short-Term Memory units to handle sequences of data, ideal for time series like stock prices.
- **Data Preprocessing**: Includes scaling features and creating time-step structured input sequences necessary for LSTM models.
- **Dropout Regularization**: Implemented to prevent overfitting by randomly omitting subset features and outputs during training.
- **Visualization**: Plotting actual versus predicted prices to assess the model's performance.

## Implementation

- The dataset comprises historical stock prices, processed to create training sequences.
- A four-layer LSTM model is constructed with dropout regularization between layers to enhance model robustness.
- The model is compiled and trained using the mean squared error loss function and Adam optimizer, then used to predict stock prices for visualization.

## Usage

This model can serve as a foundation for financial analysts and data scientists interested in applying deep learning to predict market movements.

## Conclusion

The project showcases the effectiveness of LSTM networks in modeling complex sequences and provides insights into stock price trends based on learned patterns from historical data.
