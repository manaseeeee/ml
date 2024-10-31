# Linear Regression from Scratch

# Sample data
# Independent variable (X) and dependent variable (Y)
X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

# Calculate means of X and Y
n = len(X)
mean_x = sum(X) / n
mean_y = sum(Y) / n

# Calculate the slope (m) and intercept (c)
numerator = sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
denominator = sum((X[i] - mean_x) ** 2 for i in range(n))

m = numerator / denominator
c = mean_y - m * mean_x

# Display slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

# Function to predict y given x
def predict(x):
    return m * x + c

# Testing the function with a sample input
test_x = 6
print(f"Predicted value for x = {test_x}: {predict(test_x)}")

# Calculate Mean Squared Error (MSE) for evaluation
mse = sum((Y[i] - predict(X[i])) ** 2 for i in range(n)) / n
print(f"Mean Squared Error: {mse}")
