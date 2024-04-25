# Sentiment Analysis of Restaurant Reviews

## Project Overview

This project applies Natural Language Processing (NLP) techniques to analyze and classify restaurant reviews. The goal is to use a dataset of textual reviews to predict whether a review is positive or negative. This involves preprocessing text, creating a Bag of Words model, and employing a Naive Bayes classifier for sentiment classification.

## Key Concepts

### Text Preprocessing

Text preprocessing is a critical step in any NLP task. In this project:

- **Cleaning Text**: Non-letter characters are removed, and all text is converted to lowercase to standardize inputs.
- **Tokenization**: Text is split into words or tokens. This allows for the application of further cleaning techniques.
- **Stop Words Removal**: Common words that typically don't carry much meaning in the context of sentiment analysis (like 'the', 'is', 'in') are removed to reduce noise in the input data. However, words such as 'not' are kept because they can significantly alter the sentiment of a phrase.
- **Stemming**: Words are reduced to their root form to decrease the size of the vocabulary and simplify the model without losing significant meaning.

### Bag of Words Model

The Bag of Words (BoW) model is used to convert text documents into numerical feature vectors:

- **Vectorization**: Each document is represented as a vector within a common vector space where each dimension corresponds to a word from the entire vocabulary of the training set.
- **Frequency Counts**: Each word's frequency in the document is used as the value in the corresponding dimension.

### Naive Bayes Classification

Naive Bayes is a probabilistic classifier that is particularly effective for classification with discrete features (like word counts for text classification):

- **Probability Estimation**: It uses Bayes' Theorem to predict the probability that a given feature set belongs to a particular class.
- **Assumption of Independence**: It assumes that all features are independent of each other, simplifying the computation, and often still performs well even when this assumption is violated in practice.

## Implementation

### Dataset

- The dataset consists of 1000 restaurant reviews that have been labeled as either positive or negative.

### Libraries

- This project uses Python libraries such as NumPy for handling numerical operations, pandas for data manipulation, matplotlib for plotting, and scikit-learn for machine learning workflows.

### Model Training and Evaluation

- The Naive Bayes model is trained on the processed data. Its performance is evaluated using metrics such as a confusion matrix and accuracy score to understand the model's effectiveness in classifying sentiments correctly.

## Usage

This model can serve businesses by automating the categorization of customer feedback, helping them to identify customer sentiment trends and improve their services accordingly.

## Conclusion

This project demonstrates the integration of NLP techniques and machine learning to efficiently classify textual data, providing valuable insights into consumer behavior and preferences. Such applications can be critical in areas ranging from customer service to product development.
