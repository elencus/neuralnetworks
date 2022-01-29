from pickletools import decimalnl_short
import numpy as np
import random

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = [np.random.randn(y, x) / x **.5 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.sizes = layer_sizes

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(np.matmul(w, a) + b)
        return a

    def cost(self, a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self, z, a, y):
        return (a-y)*self.sigmoid_prime(z)

    def print_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        num_correct = sum([int(x == y) for (x, y) in results])
        print("{}/{} accuracy: {}".format(num_correct, len(data), num_correct / len(data) * 100))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def SGD(self, training_data, epochs, mini_batch_size, test_data=None):
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, len(training_data), mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data:
                self.print_accuracy(test_data)

    def update_mini_batch(self, mini_batch):
        print(self.biases[-1][-1])
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # Update weights and biases according to whatever we learn from the gradient (delta_nabla_w and delta_nabla_b) of the cost function during backpropagation
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w-(1/len(self.sizes))*nw for w, nw in zip(self.weights, nabla_w)]
        self.weights = [b-(1/len(self.sizes))*nb for b, nb in zip(self.biases, nabla_b)]
        print(self.biases[-1][-1])

    def backprop(self, x, y):
        """Take in outputs and training labels, return nabla_w and nabla_b"""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []
        # feed forward
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_prime(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.sizes)):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            nabla_b = delta
        return (nabla_w, nabla_b)

    def sigmoid_prime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def cost_prime(self, a, y):
        return (a - y)
