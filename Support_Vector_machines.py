from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load data
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Plot decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.show()
