import math

import numpy as np
import numpy.typing as npt
import pylab as pl
from matplotlib.figure import Figure

import config.types as T
import scripts.plots.mapping as MP

# red, green, blue, yellow, orange, pink, purple, gray
colors = [
    (1, 0.639, 0.639),
    (0.647, 1, 0.639),
    (0.639, 0.894, 1),
    (1, 0.996, 0.639),
    (1, 0.82, 0.639),
    (1, 0.639, 0.839),
    (0.937, 0.639, 1),
    (0.678, 0.678, 0.678),
]

# uncertainty maps


def create_uncertainty_fig(
    config: T.mappingConfig,
    prob_map: np.ndarray,
    n_comp: int,
    title: str,
) -> Figure:
    """
    Purpose
    --------------

    Creates a plot for each cluster with the probability of that cluster for each pixel

    Parameters

    """
    ncols = 3
    nrows = math.ceil(n_comp / ncols)
    fig, ax = pl.subplots(nrows, ncols, constrained_layout=True)
    ax = np.atleast_1d(ax).flatten()

    for i in range(n_comp):
        add_cbar = True if i == n_comp - 1 else False

        MP.plot_patch(
            config, prob_map[i, :], "Prob", ax[i], fig, add_cbar, "probability"
        )
        ax[i].set_title(f"Cluster {i + 1}")

    for a in ax[n_comp:]:
        fig.delaxes(a)
    fig.suptitle(f"{title} posterior probability of clusters", fontsize=10)
    return fig


def create_max_prob_map(
    config: T.mappingConfig,
    prob_map: np.ndarray,
    title: str,
) -> Figure:
    """
    Purpose
    --------------

    Creates a map with the maximum probability for each pixel
    """
    fig, ax = pl.subplots(1, 1)
    max_prob_map: npt.NDArray[np.float64] = np.max(prob_map, axis=0)

    mean = round(np.mean(max_prob_map), 3)

    # cmap.set_under("black")?

    add_cbar = True
    MP.plot_patch(config, max_prob_map, "Prob", ax, fig, add_cbar, "probability")
    fig.suptitle(f"{title}maximum probability map", fontsize=10)
    ax.set_title(f"Mean Probability: {mean}")
    return fig
