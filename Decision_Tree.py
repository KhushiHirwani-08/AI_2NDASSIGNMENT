from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Model
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

# Plot tree
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, filled=True)
plt.show()
