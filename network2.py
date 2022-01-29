#### Libraries
# Standard library
from ast import Num
import json
import random
import sys

# Third-party libraries
import numpy as np

#### Main Network class
class NeuralNetwork(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/x**0.5
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.z = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.a = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.aps = [np.zeros((y, x)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.deltas = [np.zeros(y) for y in self.sizes[1:]]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gradient_descent(self, training_data, epochs, mini_batch_size, test_data=None):
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_params(mini_batch, len(training_data))
            if test_data:
                print("{}%".format(self.accuracy(test_data)))
            print("Completed epoch {}".format(j))

    def update_params(self, mini_batch, n):
        """Update networks weights and biases based on mini batch"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(1/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(1/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Input: a (array of vectors): activations,
        labels (array): desired outputs,
        zs (array of vectors): z corresponding to activations,
        w (array of vectors): weights"""
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = [np.multiply((a - y), ap) for a, y, ap in zip(activations[-1], y, sigmoid_prime(activations[-1]))]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(activations[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy / len(data) * 100

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))