"""
This file sets up the model parameters and specification
as described in 20.2.
"""
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
from numba import jit, vectorize


def get_primitives(beta=0.92, gamma=0.8, ymin=6, ymax=15, ny=10, lamb=0.66):

    # Set up ybar
    ybar = np.linspace(ymin, ymax, ny)
    pi_y = (1-lamb)/(1-lamb**ny) * lamb**(np.arange(ny))

    # Set up utility function
    u = lambda c: np.exp(-gamma*c)/(-gamma)
    uinv = lambda _u: np.log(-gamma*_u) / (-gamma)
    up = lambda c: np.exp(-gamma*c)
    upinv = lambda _up: np.log(_up) / (-gamma)

    return beta, gamma, ybar, pi_y, u, uinv, up, upinv


class Chp20_Sec3_Economy(object):
    """
    This class builds the economy described by Chapter 20 section 3 of
    "Recursive Macroeconomic Dynamics. This economy is one in which there
    is a risk-averse agent and a risk-neutral money lender with one-sided
    commitment. It is one-sided in the sense that the money lender honors
    all of his contracts, but the agent is free to walk away if he so
    chooses.
    """
    def __init__(self, beta=0.92, gamma=0.2, ymin=6, ymax=15, ny=10, lamb=0.66):
        # Generate the primitives of the model
        primitives = get_primitives(beta, gamma, ymin, ymax, ny, lamb)
        self.beta, self.gamma = primitives[0], primitives[1]
        self.ybar, self.pi_y = primitives[2], primitives[3]
        self.u, self.uinv = primitives[4], primitives[5]
        self.up, self.upinv = primitives[6], primitives[7]

        # Get the autarky value for the agent
        self.v_aut = (1/(1-beta)) * np.dot(self.u(self.ybar), self.pi_y)
        self.c_complete_markets = np.dot(self.pi_y, self.ybar)

    def v(self, c, w):
        return self.u(c) + self.beta*w

    def participation_constraint(self, c, w, s):
        """
        This function evaluates whether the incentive constraint is binding
        for a specific state (s) given consumption (c) and continuation
        promise (w)
        """
        # Pull out information we need
        ybars = self.ybar[s]
        v_aut = self.v_aut

        # Evaluate policy under the contract and exiting contract
        v_stay = self.v(c, w)
        v_exit = self.v(ybars, v_aut)

        return (v_stay >= v_exit)

    def solve(self):

        # Unpack some parameters
        beta, gamma = self.beta, self.gamma
        ybar, pi_y = self.ybar, self.pi_y
        nstates = ybar.size
        v_aut = self.v_aut
        v, u, up = self.v, self.u, self.up
        uinv, upinv = self.uinv, self.upinv

        # Allocate space for policies when participation constraint binds
        # We call this "amnesia"
        g1 = np.empty(nstates)  # Policy for consumption when hit with amnesia
        l1 = np.empty(nstates)  # Policy for cont value when hit with amnesia

        # First step is to solve equilibrium for an agent who has seen the
        # maximum income level
        l1[-1] = v(ybar[-1], v_aut)
        g1[-1] = uinv((1-beta)*l1[-1])

        for s in range(nstates-2, -1, -1):
            # Get the value of exit
            v_exit = v(ybar[s], v_aut)

            # All the j+1 to S terms
            jp1_S_terms = np.dot(pi_y[s+1:], v(g1[s+1:], l1[s+1:]))

            # Sum of 1 to j values of pi times beta
            pi1j = np.sum(pi_y[:s+1])

            # Closed form for barc_j and barw_j
            l1[s] = pi1j*v_exit + jp1_S_terms
            g1[s] = uinv((l1[s]*(1-beta*pi1j) - jp1_S_terms)/(pi1j))

        # Allocate space for money lender vf
        P = np.empty(nstates)

        # Solve for values of money lender
        P[-1] = 1/(1-beta) * np.dot(pi_y, ybar - g1[-1])
        for s in range(nstates-2, -1, -1):
            # Pull out cvalues for current states
            ck = g1[s]

            # Sum of 1 to j values of pi times beta
            pi1j = np.sum(pi_y[:s+1])

            # Solve for what happens if you have low/high shocks relative
            # to the state you bring into period
            low_flow = np.dot(pi_y[:s+1], ybar[:s+1] - ck)
            high_flow = np.dot(pi_y[s+1:], ybar[s+1:] - g1[s+1:])

            # Give continuation values of high shocks
            high_cont = np.dot(pi_y[s+1:], P[s+1:])
            P[s] = ((low_flow + high_flow) + beta*high_cont)/(1 - beta*pi1j)

        return g1, l1, P

    def simulate(self, g1, l1, T):
        """
        Given a policy for consumption (g1) and a policy for continuation
        values (l1) simulate for T periods.
        """
        # Pull out information from class
        ybar, pi_y = self.ybar, self.pi_y
        ns = self.ybar.size

        # Draw random simulation of iid income realizations
        d = qe.DiscreteRV(pi_y)
        y_indexes = d.draw(T)

        # Draw appropriate indexes for policy.
        # We do this by making sure that the indexes are weakly increasing
        # by changing any values less than the previous max to the previous max
        pol_indexes = np.empty(T, dtype=int)
        fix_indexes(ns, T, y_indexes, pol_indexes)

        # Pull off consumption and continuation value sequences
        c = g1[pol_indexes]
        w = l1[pol_indexes]
        y = ybar[y_indexes]

        return c, w, y


@jit(nopython=True)
def fix_indexes(nstates, T, realized, fixed):
    # Current values of everything
    prev_max = 0

    # loop through values and "fix" them as described above
    for t in range(T):
        cind = realized[t]
        if cind<=prev_max:
            fixed[t] = prev_max
        else:
            fixed[t] = cind
            prev_max = cind

    return None
