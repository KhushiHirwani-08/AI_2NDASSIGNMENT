import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 2, 3, 4, 5])

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, predictions, color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
