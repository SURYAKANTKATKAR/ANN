import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def step(x):
    return np.where(x >= 0, 1, 0)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate y values for each activation function
y_step = step(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_softmax = softmax(x)

'''''# Plot the activation functions
#plt.figure(figsize=(12, 8))

#plt.subplot(2,3,1)
plt.plot(x,y_step)
#plt.title('Step')

#plt.subplot(2, 3, 2)
plt.plot(x, y_sigmoid)
#plt.title('Sigmoid')

#plt.subplot(2, 3, 3)
plt.plot(x, y_tanh)
#plt.title('Tanh')

#plt.subplot(2, 3, 4)
plt.plot(x, y_relu)
#plt.title('ReLU')

#plt.subplot(2, 3, 5)
plt.plot(x, y_leaky_relu)
#plt.title('Leaky ReLU')

#plt.subplot(2, 3, 6)
plt.plot(x, y_softmax)
#plt.title('Softmax')

plt.tight_layout()
plt.show()'''
plt.plot(x, y_step, label='Step', color='red')
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.plot(x, y_relu, label='ReLU', color='orange')
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='purple')
plt.plot(x, y_softmax, label='Softmax', color='brown')

plt.legend()
'''plt.xlabel('Input')
plt.ylabel('Output')'''
plt.title('Activation Functions')
plt.show()
