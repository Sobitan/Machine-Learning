#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'homeway'
__email__ = 'xiaocao.grasses@gmail.com'
__copyright__   = 'Copyright © 2017/04/30, homeway'

import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork(object):
    '''
    :param layers: A list containing the number of units in each layer. Should be at least two values.
    :param activation: The activation function to be used. Can be "logistic" or "tanh".
    '''
    def __init__(self, layers, activation = 'tanh'):
        if activation == 'logistic':
            self.activation = self.logistic
            self.activation_deriv = self.logistic_derivative
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_deriv = self.tanh_deriv
        '''
        generate weight matrix with random float
        '''
        self.layers = layers
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_deriv(x):
        return 1.0 - np.tanh(x) * np.tanh(x)

    @staticmethod
    def logistic(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logistic_derivative(x):
        return NeuralNetwork.logistic(x) * (1 - NeuralNetwork.logistic(x))

    '''
    :param X        numpy.array     train matrix
    :param y        numpy.array     result label
    :param learning_rate    float
    :param epochs   int             backprobagation times
    '''
    def fit(self, X, y, learning_rate = 0.2, epochs = 10000):
        X = np.atleast_2d(X)
        # temp.shape=(X.shape[0], X.shape[1] + 1) `+1` is for bais, so X[*][-1] = 1 => numpy.dot(x, weights) + numpy.dot(1 * bais)
        temp = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0:-1] = X
        X = temp
        y = np.array(y)

        '''
        loop operation for epochs times
        '''
        for k in range(epochs):
            # select a random line from X for training
            i = np.random.randint(X.shape[0])
            x = [X[i]]

            # going forward network, for each layer
            for l in range(len(self.weights)):
                # computer the node value for each layer (O_i) using activation function
                x.append(self.activation(np.dot(x[l], self.weights[l])))

            # computer the error at the top layer
            error = y[i] - x[-1]
            deltas = [error * self.activation_deriv(x[-1])]  # For output layer, Err calculation (delta is updated error)

            # start backprobagation
            for l in range(len(x) - 2, 0, -1):  # we need to begin at the second to last layer
                # compute the updated error (i,e, deltas) for each node going from top layer to input layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(x[l]))
            deltas.reverse()

            for i in range(len(self.weights)):
                layer = np.atleast_2d(x[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':
    nn = NeuralNetwork(layers = [2, 3, 1], activation='tanh')
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])

    nn.fit(X, y)
    for e in X:
        print(e, nn.predict(e))