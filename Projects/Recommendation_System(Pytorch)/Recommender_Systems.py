# Recommender Systems (PyTorch)

# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load the movie and user data from CSV files
movies = pd.read_csv("movies.csv")
users = pd.read_csv("users.csv")

# Create mappings from movie titles and user IDs to numerical indices
movie_ids = movies["title"].factorize()[
    0
]  # This assigns a unique index to each unique movie title
user_ids = users["user_id"].factorize()[
    0
]  # This assigns a unique index to each unique user ID


# Define the recommendation neural network model
class RecommendationModel(nn.Module):
    def __init__(
        self, num_users, num_movies, movie_embedding_size, user_embedding_size
    ):
        super(RecommendationModel, self).__init__()
        self.movie_embedding = nn.Embedding(
            num_movies, movie_embedding_size
        )  # Embedding layer for movies
        self.user_embedding = nn.Embedding(
            num_users, user_embedding_size
        )  # Embedding layer for users
        self.dense = nn.Linear(
            movie_embedding_size + user_embedding_size, 1
        )  # Output layer

    def forward(self, user_index, movie_index):
        # Embeddings for the user and movie are retrieved and concatenated
        movie_embedding = self.movie_embedding(movie_index)
        user_embedding = self.user_embedding(user_index)
        concatenated = torch.cat(
            [movie_embedding, user_embedding], dim=1
        )  # Concatenate embeddings
        return self.dense(concatenated).squeeze(
            1
        )  # Pass through dense layer and remove extra dimensions


# Instantiate the model and optimizer
model = RecommendationModel(
    len(user_ids), len(movie_ids), movie_embedding_size=8, user_embedding_size=8
)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer

# Load ratings data
ratings = pd.read_csv("ratings.csv")
# Apply the mappings to convert user IDs and movie IDs to indices
user_indices = (
    ratings["user_id"].apply(lambda x: np.where(users["user_id"] == x)[0][0]).values
)
movie_indices = (
    ratings["movie_id"].apply(lambda x: np.where(movies["title"] == x)[0][0]).values
)

# Training loop
for epoch in range(10):  # Iterate over the dataset multiple times
    for user_index, movie_index, rating in zip(
        user_indices, movie_indices, ratings["rating"]
    ):
        # Convert data to tensors
        user_index = torch.tensor([user_index], dtype=torch.long)
        movie_index = torch.tensor([movie_index], dtype=torch.long)
        rating = torch.tensor([rating], dtype=torch.float)

        # Forward pass: Compute predicted output by passing inputs to the model
        prediction = model(user_index, movie_index)
        # Compute loss
        loss = nn.MSELoss()(prediction, rating)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Predicting ratings for a specific user
user_index = np.where(users["user_id"] == "<user_id>")[0][
    0
]  # Replace '<user_id>' with actual user ID
predictions = []
for movie_index in range(len(movie_ids)):
    movie_index = torch.tensor([movie_index], dtype=torch.long)
    prediction = model(torch.tensor([user_index]), movie_index)
    predictions.append(prediction.item())

# Print predicted ratings for the user
print(predictions)
