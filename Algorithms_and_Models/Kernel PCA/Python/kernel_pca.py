# Kernel PCA

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:, :-1].values  # Independent variables (all columns except the last)
y = dataset.iloc[:, -1].values  # Dependent variable (last column)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale features to have mean zero and variance one
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply Kernel PCA for dimensionality reduction
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(
    n_components=2, kernel="rbf"
)  # Using Radial Basis Function (RBF) kernel
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Train a logistic regression model on the transformed training set
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Function to visualize decision boundaries
def plot_decision_boundary(X_set, y_set, title):
    from matplotlib.colors import ListedColormap

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
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()


# Visualize the results on the training set
plot_decision_boundary(X_train, y_train, "Logistic Regression (Training set)")

# Visualize the results on the test set
plot_decision_boundary(X_test, y_test, "Logistic Regression (Test set)")
