import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Tao bo du lieu
X_data = np.random.random((10000, 2))
sample_weights = np.array([3, 4]).reshape(2, 1)

y_data = np.matmul(X_data, sample_weights)

y_data = np.add(y_data, np.random.uniform(-0.5, 0.5))

print(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)


