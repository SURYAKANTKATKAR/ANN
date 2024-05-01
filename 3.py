import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_fn(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def convert_to_ascii(number):
    return [ord(digit) for digit in str(number)]

# Define training data
training_data = [
    (convert_to_ascii(0), 1), # Even
    (convert_to_ascii(1), 0), # Odd
    (convert_to_ascii(2), 1), # Even
    (convert_to_ascii(3), 0), # Odd
    (convert_to_ascii(4), 1), # Even
    (convert_to_ascii(5), 0), # Odd
    (convert_to_ascii(6), 1), # Even
    (convert_to_ascii(7), 0), # Odd
    (convert_to_ascii(8), 1), # Even
    (convert_to_ascii(9), 0)  # Odd
]

# Initialize and train perceptron
input_size = len(training_data[0][0])
perceptron = Perceptron(input_size)
inputs = np.array([data[0] for data in training_data])
labels = np.array([data[1] for data in training_data])
perceptron.train(inputs, labels)

# Test the perceptron
test_number = 8
test_input = np.array(convert_to_ascii(test_number))
prediction = perceptron.predict(test_input)
if prediction == 1:
    print(f"The number {test_number} is predicted to be even.")
else:
    print(f"The number {test_number} is predicted to be odd.")
 
