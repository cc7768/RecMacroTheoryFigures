"""
This file produces a figure similar to the one at
as chapter 20 section 3 figure 1 in Recursive
Macroeconomic Theory.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from Chp20Specification import Chp20_Sec3_Economy

def Fig_20_3_1(nw=50):

    # Create the economy described
    m = Chp20_Sec3_Economy()
    v_aut = m.v_aut
    nstates = m.ybar.size

    # Create a linspace of c values
    minw = m.u(m.ybar[0]) / (1-m.beta)
    maxw = m.u(m.ybar[-1]) / (1-m.beta)
    w_ls = np.linspace(minw, maxw, nw)

    # Make plot
    fig, ax = plt.subplots()

    for i_s in range(nstates):
        # Pull out current income realization
        ybars = m.ybar[i_s]
        v_exit = m.u(ybars) + m.beta*v_aut
        cvals = np.empty(nw)

        # For each consumption find the continuation
        # value necessary for equality
        for i_w in range(nw):
            cvals[i_w] = m.uinv(v_exit - m.beta*w_ls[i_w])

        ax.plot(cvals, w_ls)

    return fig

