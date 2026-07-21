"""Statistical plot constructors."""

import matplotlib.pyplot as plt
import numpy as np


def boxplot_figure(groups, title="Objective distribution"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.boxplot(list(groups.values()), tick_labels=list(groups))
    ax.set_title(title)
    ax.set_ylabel("Objective")
    return fig, ax


def ranking_figure(ranks, title="Average algorithm rank"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = list(ranks)
    vals = [ranks[k] for k in names]
    order = np.argsort(vals)
    ax.barh([names[i] for i in order], [vals[i] for i in order])
    ax.set_title(title)
    ax.set_xlabel("Average rank")
    return fig, ax
