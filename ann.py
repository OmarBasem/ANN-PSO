#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:25:25 2020

@author: Omar Basem
"""

# -*- coding: utf-8 -*-

import numpy as np

class ANN:
    # initialize ANN
    def __init__(self, shape, weights):
        self.shape = shape
        self.num_layers = len(shape)
        self.weights = weights

    def evaluate(self, data):
        for i in range(self.num_layers - 1): # loop over the layers
            w = self.weights[i].flatten()[:-len(self.weights[i])] # extract the weights from the particle vector
            b = self.weights[i].flatten()[-len(self.weights[i]):] # extract the biases from the particle vector
            w = w.reshape(self.shape[i], len(b)) # reshape the weights
            o = data.dot(w) + b # pre-activation function

            # ACTIVATION FUNCTIONS
            data =  np.tanh(o) # tan
            # data = np.exp(-(o ** 2) / 2) # gaussian
            # data = 1 / (1 + np.exp(-o)) # sigmoid
            # data = np.cosh(o) # cosine
            # data = np.zeros((o.shape[0], o.shape[1])) # NULL
            # data = np.maximum(0, o) # RELU

        return data
