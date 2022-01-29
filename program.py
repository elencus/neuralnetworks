import network2 as nn
import numpy as np
import os

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist.npz')
with np.load(path) as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    testing_images = data['test_images']
    testing_labels = data['test_labels']

layer_sizes = (784, 16, 16, 10)

net = nn.NeuralNetwork(layer_sizes)
training_data = list(zip(training_images, training_labels))
testing_data = list(zip(testing_images, testing_labels))
print(net.accuracy(training_data))
net.gradient_descent(training_data, 30, 500, testing_data)
print(net.accuracy(training_data))
