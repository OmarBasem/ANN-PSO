#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:25:25 2020

@author: Omar Basem
"""

# -*- coding: utf-8 -*-

import functools
import numpy as np
import sklearn.metrics
import sklearn.datasets
import sklearn.model_selection
import matplotlib.pyplot as plt

from pso import PSO
from ann import ANN
import time

from helper_methods import calculate_dimensions, convert_vector_to_weights, evaluate_ann, output_new_best_particle


''' read data with 1 input'''
with open('./Data/1in_cubic.txt') as f:
    num_inputs = 1
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]
''' end of read data with 1 input'''

''' read data with 2 inputs'''
# with open('./Data/2in_complex.txt') as f:
#     num_inputs = 2
#     lines = f.readlines()
#     x = [line.split()[:2] for line in lines]
#     y = [line.split()[2:] for line in lines]
#     a = [line.split()[0] for line in lines]
#     b = [line.split()[1] for line in lines]
#     c = [line.split()[2] for line in lines]
#
# a = np.array(a)
# a = a.astype(np.float)
# b = np.array(b)
# b = b.astype(np.float)
# c = np.array(c)
# c = c.astype(np.float)
''' end of read data with 2 inputs'''

x = np.array(x)
x = x.astype(np.float)
y = np.array(y)
y = y.astype(np.float)

# reshape arrays into into rows and cols
X = x.reshape((len(x), num_inputs))
y = y.reshape((len(y), 1))

# Setting up
num_classes = 1
shape = (num_inputs, 3, 3, num_classes)
cost_function = functools.partial(evaluate_ann, shape=shape, X=X, y=y.T)
dimensions_count = calculate_dimensions(shape)
particles_count = 100
informants_count = 8
pso = PSO(cost_function, dimensions_count, particles_count, informants_count)
i = 0
best_scores = [pso.best_score]
iterations = [0]
output_new_best_particle(best_scores[-1], 0)
start_time = time.time()
epochs = 5
A = epochs  # iteration (epoch) at which MSE reaches below 0.0025

# start training
while i < epochs:
    pso.step()
    i = i + 1
    if pso.best_score < best_scores[-1]:
        output_new_best_particle(best_scores[-1], i)
        if pso.best_score < 0.0025 and A == epochs:
            A = i
    best_scores.append(pso.best_score)
    iterations.append(i)
end_time = time.time() - start_time

# Test the results
best_weights = convert_vector_to_weights(pso.global_best, shape)
best_ann = ANN(shape, best_weights)
P = best_ann.evaluate(X)


''' Start of 1 input plot result '''
plt.scatter(x, y, label='Actual')
plt.scatter(x, P, label='Predicted')
plt.title('Input (x) versus Output (y)')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()
plt.close()
''' End of 1 input plot result '''

''' Start of 2 inputs plot result '''
# ax = plt.axes(projection='3d')
# ax.scatter(a, b, c)
# ax.scatter(a, b, P.flatten())
''' End of 2 inputs plot result '''


# Plot MSE against epochs
fig = plt.figure()
plt.plot(iterations, best_scores, label='MSE x epochs')
plt.title('MSE x epochs')
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.close()

# output the results
print("\nBest MSE: ", "{:.3e}".format(best_scores[-1]))
print("Iteration at which MSE goes below 0.0025: ", A)
print("Execution time: %s seconds ---\n" % "{:.3f}".format(end_time))


