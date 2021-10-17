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
        def calculate_activations_and_zs(a, i=0):
            z = self.weights[i] @ a + self.biases[i]
            act = sigmoid(z)

            if i == len(self.weights) - 1:
                return [act], [z]
            
            acts, zs = calculate_activations_and_zs(act, i + 1)
            return [act] + acts, [z] + zs
        
        activations, zs = calculate_activations_and_zs(x)

        def calculate_deltas(l=0):
            if l == len(activations) - 1:
                delta = self.cost_derivative(activations[l], y) * sigmoid_prime(zs[l])
                return [delta]
            
            deltas = calculate_deltas(l + 1)
            delta = ( self.weights[l + 1].T @ deltas[0] ) * sigmoid_prime(zs[l])
            
            return [delta] + deltas
        
        deltas = calculate_deltas()
        nabla_b = deltas
        nabla_w = [
            deltas[i] @ activations[i - 1].T if i != 0 else deltas[i] @ x.T
            for i in range(len(deltas))
        ]

        return nabla_b, nabla_w

    
    def evaluate(self, data):
        eval_result = [ (np.argmax(self.feedforward(x)), y) for x, y in data ]
        return sum([ int(x == y) for x, y in eval_result ]) / len(data)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


def list_sum(l1: list, l2: list) -> list:
    return [x1 + x2 for x1 ,x2 in zip(l1, l2)]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))