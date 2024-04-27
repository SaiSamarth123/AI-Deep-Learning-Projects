# Recommender Systems (Keras)

# Import necessary libraries for data handling and machine learning
import numpy as np
import pandas as pd
from keras.layers import Embedding, Dot, Reshape, Dense
from keras.models import Sequential

# Load movie and user datasets
movies = pd.read_csv("movies.csv")
users = pd.read_csv("users.csv")

# Factorize the movie titles and user IDs to get unique numeric identifiers for each
movie_ids = movies["title"].factorize()[0]
user_ids = users["user_id"].factorize()[0]

# Create an embedding layer for movie IDs
# This layer will learn an embedding for all movies in the dataset
movie_embedding_size = 8  # Size of the embedding vector for each movie
movie_embedding = Embedding(
    input_dim=len(np.unique(movie_ids)), output_dim=movie_embedding_size, input_length=1
)

# Create an embedding layer for user IDs
# Similarly, this layer learns an embedding for all users
user_embedding_size = 8  # Size of the embedding vector for each user
user_embedding = Embedding(
    input_dim=len(np.unique(user_ids)), output_dim=user_embedding_size, input_length=1
)

# Define a model to combine these embeddings and predict ratings
model = Sequential()
model.add(movie_embedding)  # Embedding layer for movies
model.add(Reshape((movie_embedding_size,)))  # Flatten movie embeddings
model.add(user_embedding)  # Embedding layer for users
model.add(Reshape((user_embedding_size,)))  # Flatten user embeddings
model.add(
    Dot(axes=1)
)  # Dot product of movie and user embeddings to get a scalar rating
model.add(
    Dense(1, activation="sigmoid")
)  # Sigmoid layer to ensure the final rating is between 0 and 1

# Compile the model with binary crossentropy loss and the adam optimizer
model.compile(loss="binary_crossentropy", optimizer="adam")

# Load ratings data
ratings = pd.read_csv("ratings.csv")
# Map user IDs and movie IDs in ratings data to the corresponding indices used in embeddings
user_indices = (
    ratings["user_id"].apply(lambda x: np.where(users["user_id"] == x)[0][0]).values
)
movie_indices = (
    ratings["movie_id"].apply(lambda x: np.where(movies["title"] == x)[0][0]).values
)

# Train the model on user ratings
model.fit(x=[user_indices, movie_indices], y=ratings["rating"], epochs=10)

# Make recommendations for a specific user
# Get the user index from user ID
user_index = np.where(users["user_id"] == "<user_id>")[0][
    0
]  # Replace '<user_id>' with actual user ID
# Predict ratings for all movies for this user
predictions = model.predict(
    [np.array([user_index] * len(movie_ids)), np.array(range(len(movie_ids)))]
)
# Get indices of the top 5 recommended movies
recommended_movie_ids = predictions.flatten().argsort()[-5:][::-1]
# Fetch the recommended movies details
recommended_movies = movies.iloc[recommended_movie_ids]

# Display recommended movies
print(recommended_movies)
