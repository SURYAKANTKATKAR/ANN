import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_fn(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_fn(summation)

    def train(self, training_inputs, labels):
        self.weights = np.zeros(1 + training_inputs.shape[1])
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

# Generate random data points for two classes
np.random.seed(0)
class1 = np.random.randn(50, 2) + [2, 2]
class2 = np.random.randn(50, 2) + [-2, -2]

# Concatenate the data points and labels
data = np.vstack((class1, class2))
labels = np.hstack((np.ones(50), np.zeros(50)))

# Initialize and train the perceptron
perceptron = Perceptron()
perceptron.train(data, labels)

# Plot the data points
plt.figure(figsize=(8, 6))
plt.scatter(class1[:, 0], class1[:, 1], color='blue', label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color='red', label='Class 2')

# Plot the decision boundary
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, alpha=0.5)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Regions')
plt.legend()
plt.show()
