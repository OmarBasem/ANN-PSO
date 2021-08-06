#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:54:56 2020

@author: Omar Basem
"""

''' start of helper methods '''


import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection

from ann import ANN

# function to calculate the number of the dimensions of an ANN for the PSO
def calculate_dimensions(shape):
    dimensions = 0
    for i in range(len(shape) - 1):
        dimensions += (shape[i] + 1) * shape[i + 1]
    return dimensions


# function to convert a particle's vector into weights
def convert_vector_to_weights(vector, shape):
    weights = []
    index = 0
    index_min = index
    for i in range(len(shape) - 1):
        rows = shape[i + 1]
        cols = shape[i] + 1
        index_max = index_min + rows * cols
        W = vector[index_min:index_max].reshape(rows, cols)
        weights.append(W)
        index_min = index_max
    return weights

# Evaluation function of ANN, used as the cost function for the PSO, returns the MSE
def evaluate_ann(weights, shape, X, y):
    mse = np.asarray([])
    y = y.reshape(y.shape[1], 1)
    for w in weights:
        weights = convert_vector_to_weights(w, shape)
        ann = ANN(shape, weights)
        y_pred = ann.evaluate(X)

        if np.all(np.isinf(y_pred).any()) or np.all(np.isnan(y_pred).any()):  # handles infinity and nan values
            mse = np.append(mse, np.inf)
            continue

        mse = np.append(mse, sklearn.metrics.mean_squared_error(y, y_pred))
    return mse


def output_new_best_particle(score, iteration):
    print("Best particle has been found at iteration #{i} with MSE of: {score}".format(i=iteration, score=score))

''' end of helper methods '''