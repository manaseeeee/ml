# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define the dataset
X = np.array([
    [3, 7], [4, 6], [1, 8], [5, 5], [6, 5],
    [7, 6], [2, 7], [8, 4], [9, 4], [10, 3]
])
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1])  # Pass (1) or Fail (0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Output model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Show prediction probabilities (optional)
print("\nPrediction Probabilities:")
print(model.predict_proba(X_test))
