# Natural Language Processing

# Import essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset from a TSV (Tab Separated Values) file
dataset = pd.read_csv(
    "Restaurant_Reviews.tsv", delimiter="\t", quoting=3
)  # quoting=3 ignores double quotes in the file

# Text preprocessing
import re
import nltk

nltk.download("stopwords")  # Download stopwords from NLTK
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # Import Porter stemmer for stemming

# Prepare to clean the texts
corpus = []  # Initialize an empty list to hold the cleaned texts
for i in range(
    0, 1000
):  # Loop over each review (assuming there are 1000 reviews in the dataset)
    # Remove all punctuation and numbers from text
    review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
    # Convert text to lowercase to maintain uniformity
    review = review.lower()
    # Split text into words
    review = review.split()
    # Stemming (reducing to root form) and removal of stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove(
        "not"
    )  # Remove 'not' from stopwords list because it's crucial for sentiment analysis
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    # Join words back to form the cleaned up review
    review = " ".join(review)
    # Append each cleaned review to the corpus
    corpus.append(review)
print(corpus)

# Convert text data into numerical data using Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=1500
)  # Limit number of columns to 1500 to consider the top 1500 most frequent words
X = cv.fit_transform(corpus).toarray()  # Transform the corpus to a matrix of features
y = dataset.iloc[
    :, -1
].values  # Extract target values (assumed to be the last column of the dataset)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)  # 20% data for testing

# Train a Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = classifier.predict(X_test)
print(
    np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1)
)

# Evaluate the model's performance
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))
