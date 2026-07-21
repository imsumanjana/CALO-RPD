"""Convergence plot constructors."""

import matplotlib.pyplot as plt


def convergence_figure(series, title="Convergence comparison"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for label, values in series.items():
        ax.plot(range(1, len(values) + 1), values, label=label)
    ax.set_title(title)
    ax.set_xlabel("Recorded iteration")
    ax.set_ylabel("Best objective")
    return fig, ax
