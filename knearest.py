import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

test_point = X_test[0]
neighbors_indices = model.kneighbors([test_point], 3, return_distance=False)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', alpha=0.5)
plt.scatter(X_test[:, 0], X_test[:, 1], c='green', marker='x', s=100)
plt.scatter(test_point[0], test_point[1], c='red', marker='*', s=500)

for index in neighbors_indices[0]:
    plt.scatter(X_train[index, 0], X_train[index, 1], edgecolor='black', s=150)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('KNN Visualization on Iris Dataset')
plt.legend(['Training Data', 'Test Data', 'Test Point', 'Nearest Neighbors'])
plt.grid()
plt.show()
