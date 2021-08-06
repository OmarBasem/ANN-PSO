#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:25:25 2020

@author: Omar Basem
"""

# -*- coding: utf-8 -*-

import functools
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt


from pso import PSO
from ann import ANN
import time
from helper_methods import calculate_dimensions, convert_vector_to_weights, evaluate_ann, output_new_best_particle

''' This is an example test file for variable number of informants. It test informants count of
2, 8, 14 and 20 informants. Each test is run for 10 times, and then the average is taken. At the end of all the tests
a graph is plotted for the 4 different values of informants count. All the other tests has been carried out
in a similar way.'''

''' read data '''
with open('./Data/1in_tanh.txt') as f:
    num_inputs = 1
    lines = f.readlines()
    x = [line.split()[0] for line in lines]
    y = [line.split()[1] for line in lines]


x = np.array(x)
x = x.astype(np.float)
y = np.array(y)
y = y.astype(np.float)

# reshape arrays into into rows and cols
X = x.reshape((len(x), num_inputs))
y = y.reshape((len(y), 1))

# Setting up
num_classes = 1
mnist = sklearn.datasets.load_digits(num_classes)
informants_list = [2, 8, 14, 20]
all_P = []
all_mse = []
all_epochs = []
test_outcomes = []




''' Start of testing '''
# loop over the test variable
for t in range(len(informants_list)):
    print('Test Number: ', t + 1)

    # init lists for different averages
    best_mse = []
    mse_A = []
    exec_time = []
    results = []
    avg_mse_list = []
    avg_epochs_list = []
    shape = (num_inputs, 3, 3, num_classes)
    cost_func = functools.partial(evaluate_ann, shape=shape, X=X, y=y.T)

    # repeat test 10 times
    for num in range(10):
        print("Run number: ", num + 1)        
        particles_count = 100
        epochs = 500
        # Initialize PSO
        dimensions_count = calculate_dimensions(shape)
        informants_count = informants_list[t]
        swarm = PSO(cost_func, dimensions_count, particles_count, informants_count)
        i = 0
        best_scores = [swarm.best_score]
        iterations = [0]
        output_new_best_particle(best_scores[-1], 0)
        start_time = time.time()
        A = epochs  # iteration (epoch) at which MSE reaches below 0.0025

        # start training
        while i < epochs:
            swarm.step()
            i = i + 1
            if swarm.best_score < best_scores[-1]:
                output_new_best_particle(best_scores[-1], i)
                if swarm.best_score < 0.0025 and A == epochs:
                    A = i
            best_scores.append(swarm.best_score)
            iterations.append(i)
        end_time = time.time() - start_time


        # Test the results
        best_weights = convert_vector_to_weights(swarm.global_best, shape)
        best_ann = ANN(shape, best_weights)
        P = best_ann.evaluate(X)

        # add results to lists to calculate average after all tests done
        results.append(P)
        avg_mse_list.append(best_scores)
        avg_epochs_list.append(iterations)

        # plot graphs of current test
        plt.scatter(x, y, label='Actual')
        plt.scatter(x, P, label='Predicted')
        plt.title('Input (x) versus Output (y)')
        plt.xlabel('Input Variable (x)')
        plt.ylabel('Output Variable (y)')
        plt.legend()
        plt.show()
        plt.close()

        fig = plt.figure()
        plt.plot(iterations, best_scores, label='MSE x epochs')
        plt.title('MSE x epochs')
        plt.ylabel('MSE')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
        plt.close()

        best_mse.append(best_scores[-1])
        if A != epochs:
            mse_A.append(A)
        exec_time.append(end_time)

        # output results of current test
        print("Best MSE: ", "{:.3e}".format(best_scores[-1]))
        print("Iteration at which MSE goes below 0.0025: ", A)
        print("---Execution time: %s seconds ---\n" % "{:.3f}".format(end_time))

    # Calculate the averages of the 10 runs
    test_outcomes.append([np.mean(best_mse), round(np.mean(mse_A)), np.mean(exec_time)])
    avg_P = np.average(results, axis=0)
    all_P.append(avg_P)
    avg_epochs = np.average(avg_epochs_list, axis=0)
    avg_mse = np.average(avg_mse_list, axis=0)
    all_mse.append(avg_mse)
    all_epochs.append(avg_epochs)

''' End of testing '''

# plot the different graphs for the different tests
plt.scatter(x, y, label='Actual')
for t in range(4):
    plt.scatter(x, all_P[t], label=str(informants_list[t]) + ' informants')
plt.title('Input (x) versus Output (y) for variable number of informants')
plt.xlabel('Input Variable (x)')
plt.ylabel('Output Variable (y)')
plt.legend()
plt.show()
plt.close()

# plot MSE against the epochs graphs
fig = plt.figure()
for t in range(4):
    print('Results for ' + str(informants_list[t]) + ' informants')
    print("Average MSE:", "{:.2e}".format(test_outcomes[t][0]))
    print("Average iteration at which MSE goes below 0.0025:", test_outcomes[t][1])
    print("Average execution time:", "{:.1f}".format(test_outcomes[t][2]))
    plt.plot(all_epochs[t], all_mse[t], label=str(informants_list[t]) + ' informants')
plt.title('MSE X Epochs for variable number of informants')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.close()
