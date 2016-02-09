"""
This file generates a pictures that is similar to the one
shown in Recursive Macroeconomic Theory Chapter 20 Section
2 figure 1a. That figure shows one path while this will
show many paths.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Chp20Specification import Chp20_Sec3_Economy


def Figure_20_2_1a(npaths=250, T=150):
    m = Chp20_Sec3_Economy()
    g1, l1 = m.solve()

    csims = np.empty((npaths, T))
    for i in range(npaths):
        c, w, y = m.simulate(g1, l1, T)
        csims[i, :] = c

    fig, ax = plt.subplots()

    ax.plot((csims[1:, :]).T, alpha=0.25, color="r", linewidth=0.75)
    ax.plot(csims[0, :], color="k", linewidth=2)
    ax.hlines(m.c_complete_markets, 0, T, alpha=0.75, colors="k", linestyle="--")

    ax.set_title("Consumption for one-sided commitment")
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption")

    return fig

