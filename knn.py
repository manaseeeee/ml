import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.knn import KNN
from pyod.utils.data import generate_data, get_outliers_inliers

# Generating a random dataset with two features
X_train, y_train = generate_data(n_train=300, train_only=True, n_features=2)

# Setting the percentage of outliers
outlier_fraction = 0.1

# Storing the outliers and inliers in different numpy arrays
X_outliers, X_inliers = get_outliers_inliers(X_train, y_train)
n_inliers = len(X_inliers)
n_outliers = len(X_outliers)

# Separating the two features
f1 = X_train[:, [0]].reshape(-1, 1)
f2 = X_train[:, [1]].reshape(-1, 1)

# Visualizing the dataset
# Create a meshgrid
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

# Scatter plot of the data points
plt.scatter(f1, f2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Points')
plt.show()

# Training the KNN outlier detector
clf = KNN(contamination=outlier_fraction)
clf.fit(X_train)

# Prediction scores and number of errors
scores_pred = clf.decision_function(X_train) * -1  # Raw anomaly scores
y_pred = clf.predict(X_train)  # 1 for outliers, 0 for inliers
n_errors = (y_pred != y_train).sum()
print('The number of prediction errors are:', n_errors)

# Threshold to consider a data point as an inlier or outlier
threshold = stats.scoreatpercentile(scores_pred, 100 * outlier_fraction)

# Decision function calculates the raw anomaly score for every point
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and data points
plt.figure(figsize=(10, 5))

# Contour plot for decision boundary
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 10), cmap=plt.cm.Blues_r)
# Draw red contour line where anomaly score equals threshold
plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

# Fill orange where anomaly score is above threshold
plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

# Scatter plot for inliers and outliers
plt.scatter(X_inliers[:, 0], X_inliers[:, 1], c='white', s=20, edgecolor='k', label='Inliers')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='black', s=20, edgecolor='k', label='Outliers')

# Configure plot
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Nearest Neighbors Outlier Detection')
plt.legend(loc='lower right')
plt.xlim((-10, 10))
plt.ylim((-10, 10))
plt.show()
