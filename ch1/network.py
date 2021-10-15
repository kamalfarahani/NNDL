import random

import numpy as np

from functools import reduce


class Network():
    def __init__(self, layers_sizes):
        self.number_of_layers = len(layers_sizes)
        self.biases = [ np.random.randn(s, 1) for s in layers_sizes[1:] ]
        self.weights = [
            np.random.randn(y, x) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])
        ]
    
    def feedforward(self, input_vector):
        def go(act, i = 0):
            new_act = sigmoid(self.weights[i] @ act + self.biases[i])
            if i == len(self.weights) - 1:
                return new_act
            
            return go(new_act, i + 1)
        
        return go(input_vector)
    
    def SGD(self, training_data, epochs, batch_size, learning_rate, test_data=None):
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[j : j + batch_size] for j in range(0, n, batch_size)]
            for batch in batches:
                self.update_with_batch(batch, learning_rate)
            
            if test_data:
                performance = self.evaluate(test_data)
                print(f'For { i }th epoch performance is { performance }')
            else:
                print(f'Epoch { i } completed.')

    def update_with_batch(self, batch, learning_rate):
        batch_size = len(batch)

        nabla_b, nabla_w =  reduce(
            lambda acc, x: (list_sum(acc[0], x[0]), list_sum(acc[1], x[1])),
            [ self.backprop(x, y) for x, y in batch ]
        )

        eta = learning_rate / batch_size
        self.weights = list( np.array(self.weights) - (eta * np.array(nabla_w)) )
        self.biases = list( np.array(self.biases) - (eta * np.array(nabla_b)) )

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.number_of_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, data):
        eval_result = [ (np.argmax(self.feedforward(x)), y) for x, y in data ]
        return sum([ int(x == y) for x, y in eval_result ]) / len(data)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)



def list_sum(l1: list, l2: list) -> list:
    return [x1 + x2 for x1 ,x2 in zip(l1, l2)]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))