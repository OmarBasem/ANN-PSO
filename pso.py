#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 10:25:25 2020

@author: Omar Basem
"""

# -*- coding: utf-8 -*-

import numpy as np
from random import choice


class PSO(object):
    # initializations function
    def __init__(self, cost_function, dimensions_count, particles_count, informants_count, inertia=0.75, p_coef=1, i_coef=1, g_coef=1,
                 jump_size=0.75):

        # initialize PSO with paramters
        self.cost_function = cost_function
        self.dimensions_count = dimensions_count
        self.particles_count = particles_count
        self.inertia = inertia
        self.p_coef = p_coef
        self.i_coef = i_coef
        self.g_coef = g_coef
        self.jump_size = jump_size
        self.informants_count = informants_count

        self.X = np.random.uniform(size=(self.particles_count, self.dimensions_count)) # init particles position
        self.V = np.random.uniform(size=(self.particles_count, self.dimensions_count)) # inti particles velocity

        # find the current best
        self.personal_best = self.X.copy()
        self.scores = self.cost_function(self.X)
        self.global_best = self.personal_best[self.scores.argmin()]
        self.best_score = self.scores.min()

    # update function
    def step(self):

        # ASSIGN INFORMANTS
        informants_best = []
        for i in range(self.particles_count):
            l = [i]
            s = [self.scores[i]]
            for j in range(self.informants_count - 1):
                num = choice([x for x in range(self.particles_count) if x not in l])
                l.append(num)
                s.append(self.scores[num])
            informants_best.append(self.personal_best[l[np.asarray(s).argmin()]])
        self.informants_best = np.asarray(informants_best)

        # Random noise vectors
        p_noise = np.random.uniform(self.p_coef, size=(self.particles_count, self.dimensions_count))
        i_noise = np.random.uniform(self.i_coef, size=(self.particles_count, self.dimensions_count))
        g_noise = np.random.uniform(self.g_coef, size=(self.particles_count, self.dimensions_count))

        # velocity update
        self.V = (self.inertia * self.V) + (p_noise * (self.personal_best - self.X)) + (i_noise * (self.informants_best - self.X)) + (
                    g_noise * (self.global_best - self.X))

        # position update
        self.X = self.X + (self.jump_size * self.V)

        # Update global best and best scores
        scores = self.cost_function(self.X)
        better_scores_index = scores < self.scores
        self.personal_best[better_scores_index] = self.X[better_scores_index]
        self.scores[better_scores_index] = scores[better_scores_index]
        self.global_best = self.personal_best[self.scores.argmin()]
        self.best_score = self.scores.min()
