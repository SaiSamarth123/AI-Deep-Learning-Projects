# Bank Customer Churn Prediction using Artificial Neural Network (ANN)

## Project Overview

This project develops an ANN to predict bank customer churn based on their profile and banking activity. The model is trained on historical data to identify patterns that indicate the likelihood of customers leaving the bank.

## Key Concepts

- **Artificial Neural Network**: A computing system inspired by biological neural networks that constitute animal brains.
- **Data Preprocessing**: Involves encoding categorical variables and scaling features to prepare data for effective model training.
- **Model Training and Evaluation**: The ANN is trained with features like geography, credit score, and account details, and evaluated using accuracy metrics and a confusion matrix.

## Implementation

- The data is first preprocessed using `LabelEncoder` and `OneHotEncoder` for categorical variables and `MinMaxScaler` for scaling.
- The ANN structure includes multiple dense layers with ReLU activation for hidden layers and a sigmoid activation for the output layer to predict churn probability.
- The model is compiled with the Adam optimizer and binary cross-entropy loss function, suitable for binary classification tasks.

## Usage

The model can predict individual churn risk and evaluate overall model accuracy against unseen test data. It helps in making informed decisions on retaining valuable customers.

## Conclusion

This project illustrates the application of neural networks in predicting customer behavior, providing insights into customer retention strategies.
