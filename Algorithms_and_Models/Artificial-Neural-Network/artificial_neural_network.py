# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Display TensorFlow version
print(tf.__version__)

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")  # Load data from CSV file
X = dataset.iloc[:, 3:-1].values  # Independent variables (input features)
y = dataset.iloc[:, -1].values  # Dependent variable (output/target)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])  # Convert gender text to 0 or 1

# One Hot Encoding the "Geography" column
# This is necessary because the model cannot interpret textual data.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))  # Apply one hot encoding and convert to numpy array

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)  # 20% data for testing, 80% for training

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit and transform training set
X_test = sc.transform(X_test)  # Transform test set using same scaler

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()  # Sequential model for stacking layers

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))  # 6 neurons, ReLU activation

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))  # Another layer with ReLU

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))  # Sigmoid for output

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Adam optimizer and binary crossentropy for binary classification

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)  # 100 epochs, batch size of 32

# Part 4 - Making the predictions and evaluating the model

# Predicting the result for a new observation
new_customer = np.array(ct.transform([["France"]]), dtype=np.float64)
# Preprocessing the single new data point
new_customer = np.append(new_customer, [[600, 1, 40, 3, 60000, 2, 1, 1, 50000]], axis=1)
# Include new customer data, fit with the model's expected input structure
new_customer_scaled = sc.transform(new_customer)
# Scale the data similarly to the training data

# Predicting the churn for the new customer
new_customer_churn = ann.predict(new_customer_scaled) > 0.5
# Using the ANN to make a prediction and threshold it
print("Will the customer leave the bank? ", "Yes" if new_customer_churn else "No")

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5  # Convert probabilities to binary output

# Display combined real and predicted values
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)  # Compare actual outcomes to predictions
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))  # Print the accuracy of the model
