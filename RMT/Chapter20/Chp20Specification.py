"""
This file sets up the model parameters and specification
as described in 20.2.
"""
import numpy as np
import matplotlib.pyplot as plt

from numba import jit, vectorize


def get_primitives(beta=0.92, gamma=0.8, ymin=6, ymax=15, ny=10, lamb=0.66):

    # Set up ybar
    ybar = np.linspace(ymin, ymax, ny)
    pi_y = (1-lamb)/(1-lamb**ny) * lamb**(np.arange(ny))

    # Set up utility function
    u = lambda c: np.exp(-gamma*c)/(-gamma)
    up = lambda c: np.exp(-gamma*c)

    return beta, gamma, ybar, pi_y, u, up

