print("Script is running...")

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

######
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create and train a Logistic Regression model
logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
logreg_predictions = logreg_model.predict(X_test)

# Evaluate the model
logreg_accuracy = accuracy_score(y_test, logreg_predictions)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")

# Print other evaluation metrics if needed
print(confusion_matrix(y_test, logreg_predictions))
print(classification_report(y_test, logreg_predictions))

######
from sklearn.neighbors import KNeighborsClassifier

# Create and train a k-NN model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Make predictions on the test set
knn_predictions = knn_model.predict(X_test)

# Evaluate the model
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"k-NN Accuracy: {knn_accuracy}")
##############################
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Print the description of the dataset
print(iris.DESCR)

# Print feature names
print("Feature names:", iris.feature_names)

# Print target names
print("Target names:", iris.target_names)

# Print the first few rows of the data
print("Data samples:")
print(iris.data[:5])

# Print the first few target values
print("Target values:")
print(iris.target[:5])
