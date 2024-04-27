# Linear Discriminant Analysis (LDA)

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the dataset
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:, :-1].values  # Features: all columns except the last
y = dataset.iloc[:, -1].values  # Target: the last column

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardizing the features (important for LDA)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit and transform training set
X_test = sc.transform(X_test)  # Transform test set

# Applying LDA for dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)  # Projecting down to 2 dimensions
X_train = lda.fit_transform(
    X_train, y_train
)  # Fitting LDA and transforming training data
X_test = lda.transform(X_test)  # Transforming test data

# Training a Logistic Regression model on the transformed data
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)  # Fitting model

# Evaluating the model using a confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)  # Predicting test results
cm = confusion_matrix(y_test, y_pred)  # Creating confusion matrix
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Function to visualize decision boundaries for the training set
def visualize_results(X_set, y_set, set_description):
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
    )
    plt.contourf(
        X1,
        X2,
        classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75,
        cmap=ListedColormap(("red", "green", "blue")),
    )
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0],
            X_set[y_set == j, 1],
            c=ListedColormap(("red", "green", "blue"))(i),
            label=j,
        )
    plt.title(f"Logistic Regression ({set_description} set)")
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.legend()
    plt.show()


# Visualizing the Training set results
visualize_results(X_train, y_train, "Training")

# Visualizing the Test set results
visualize_results(X_test, y_test, "Test")
