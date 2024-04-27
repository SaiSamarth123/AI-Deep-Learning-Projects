# Linear Discriminant Analysis (LDA) for Wine Classification

## Project Overview

This project utilizes Linear Discriminant Analysis (LDA), a dimensionality reduction technique, followed by a logistic regression model to classify types of wine based on various chemical properties. The dataset used includes measurements from different wine instances and a classification label for each wine.

## Key Concepts

### Linear Discriminant Analysis (LDA)

LDA is a statistical method used for dimensionality reduction while maintaining the class-discriminatory information. It is particularly useful in preprocessing for classification tasks. The key steps involved in LDA include:

1. **Computing the Within-Class and Between-Class Scatter Matrices**: These matrices encapsulate the variability within each class and the differences between classes, respectively.
2. **Calculating Eigenvalues and Eigenvectors**: These are computed from the scatter matrices to determine the new feature space that maximizes the class separability.
3. **Projecting Data to Lower Dimensions**: The original features are projected to this new feature space, reducing the number of dimensions while attempting to retain the most significant class-distinctive information.

### Logistic Regression

After reducing dimensionality, logistic regression is applied to perform classification. Logistic regression is a predictive analysis used to describe data and the relationship between one dependent binary variable and one or more nominal, ordinal, interval
