# Part 1 - Data Preprocessing

# Import essential libraries for data handling and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the training dataset
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# Extract the 'Open' stock prices and convert to numpy array for easier manipulation
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling using Min-Max normalization to help the neural network learn efficiently
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
# This means at each time t, the RNN will look at the stock prices at 60 previous times (t-60) to make a prediction at time t
X_train = []
y_train = []
for i in range(60, 1258):  # 1258 is the total number of training data points
    X_train.append(training_set_scaled[i - 60 : i, 0])  # Input: 60 previous days
    y_train.append(training_set_scaled[i, 0])  # Output: the next day
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the data to fit the model input format (batch_size, timesteps, input_dim)
# Here, input_dim is 1 since we are only using one feature (stock price)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN as a sequence of layers
regressor = Sequential()

# Adding the first LSTM layer with Dropout regularisation to avoid overfitting
# 'units' is the number of neurons in the layer
# 'return_sequences=True' because we will add more LSTM layers after this one
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding more LSTM layers with Dropout regularisation
# It's important to note that we do not need to specify the input shape in these layers
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# Adding the fourth LSTM layer (we do not return sequences here because it is the final LSTM layer)
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
# 'units' is 1 because we want to predict a single value (the stock price)
regressor.add(Dense(units=1))

# Compiling the RNN
# We use the Adam optimizer and mean squared error as the loss function since it's a regression problem
regressor.compile(optimizer="adam", loss="mean_squared_error")

# Fitting the RNN to the Training set
# We use X_train as both inputs and outputs because it's a self-regressive model
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Part 3 - Making predictions and visualizing the results

# Getting the real stock price of 2017 to compare with our predictions
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:, 1:2].values

# Preparing the input for the model to make predictions
# We need to concatenate the training set and test set before scaling to avoid information leakage
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60 :].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(
    60, 80
):  # Using the last 60 entries from the total dataset to predict the following 20 entries
    X_test.append(inputs[i - 60 : i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions on the test data
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(
    predicted_stock_price
)  # Inverse the scaling

# Visualizing the results by plotting the predicted and the real stock price
plt.plot(real_stock_price, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
