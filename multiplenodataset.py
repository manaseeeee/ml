# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define the dataset
X = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
    [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]
])
y = np.array([5, 8, 11, 14, 17, 20, 23, 26, 29, 32])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a multiple linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Output model performance
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Visualize the relationship between actual and predicted values
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual values")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted values")
plt.xlabel("Test sample index")
plt.ylabel("Target value")
plt.legend()
plt.show()
