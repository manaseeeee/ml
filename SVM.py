import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = datasets.load_iris(return_X_y=True)
X, y = X[:100, :2], y[:100]

svc = SVC(kernel="linear")
svc.fit(X, y)

y_pred = svc.predict(X)

accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100), 
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.xlabel("Sepal length")
plt.ylabel("Sepal length")
plt.title("Linear SVM on Iris (First 100 samples)")
