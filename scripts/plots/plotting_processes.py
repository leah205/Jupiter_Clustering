from typing import cast

import numpy as np
import numpy.typing as npt
import pylab as pl
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

import config.types as T
import scripts.plots.mapping as MP
import scripts.plots.plots as PL
import scripts.preprocessing.preprocessing as pre
from config.config import cf
from scripts.helpers.io import get_dir_path, get_fits_files

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


"""
Helper functions to create mapping and plotting figures


Parameters
    --------------------------------
   
    config, REQUIRED
        - instance of pipelineConfig
    pred
        - one dimensional array of cluster assignments
    arr
        - original processed data np array for clustering with indices as last column
    subset_shape
        - dimensions of original spatial subset pixel data 
    param_ranges
        - list of length two lists with desired values ranges for each dimension in clustering, same order as keywords
    

"""


def create_map_comp_figure(
    config: T.mappingConfig,
    reshaped_pred: npt.NDArray[np.float64],
    title: str,
    n_comp: int,
) -> Figure:
    """
    Creates spatial map comparing radiances in different filters with cluster outputs
    """
    dim1, dim2 = config.keywords[0], config.keywords[1]

    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1, constrained_layout=True)
    fig = cast(Figure, fig)

    dir_path = get_dir_path(config)
    file1 = get_fits_files(dir_path, [dim1])[0]
    file2 = get_fits_files(dir_path, [dim2])[0]

    map1 = pre.get_patch(dim1, config.latRng, config.lngRng, config.cm_num, file1)
    map2 = pre.get_patch(dim2, config.latRng, config.lngRng, config.cm_num, file2)

    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis[0], n_comp, fig)

    MP.plot_patch(config, map1, dim1, axis[1], fig)
    MP.plot_patch(config, map2, dim2, axis[2], fig)
    _ = fig.suptitle(f"{title}", fontsize=10)

    return fig


def create_plot_figure(
    config: T.mappingConfig,
    arr: npt.NDArray[np.float64],
    reshaped_pred: npt.NDArray[np.float64],
    title: str,
    cluster_obj: dict,
    n_comp: int,
) -> Figure:
    """
    Creates scatter plot of clusters in two parameters and spatial cluster map
    """
    # creates and saves cluster map and cluster scatter plot
    pred = cluster_obj["pred"]
    cmap = ListedColormap(colors[:n_comp])

    fig, axis = pl.subplots(2, 1, figsize=(8, 8))
    fig = cast(Figure, fig)
    pl.tight_layout()
    _ = fig.subplots_adjust(top=0.95, left=0.1, right=0.9, bottom=0.1)
    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis[0], n_comp, fig)
    PL.create_cluster_plot(axis[1], config.keywords, 0, 1, arr, pred, n_comp, cmap)
    PL.plot_gmm_ellipsoids(axis[1], cluster_obj, cmap)
    _ = fig.suptitle(f"{title}", fontsize=10)
    # STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    return fig


def create_plots_figure(
    config: T.mappingConfig,
    pred: np.ndarray,
    input_arr: np.ndarray,
    reshaped_pred: np.ndarray,
    title: str,
    n_comp: int,
) -> Figure:
    """
    Creates two scatter plots each showing clusters with two parameters and a spatial cluster map
    """
    # creates and saves cluster map and cluster scatter plot
    n_comp = np.unique(pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1)
    fig = cast(Figure, fig)

    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis[0], n_comp, fig)
    PL.create_cluster_plot(
        axis[1], config.keywords, 0, 1, input_arr, pred, n_comp, cmap
    )
    PL.create_cluster_plot(
        axis[2], config.keywords, 2, 3, input_arr, pred, n_comp, cmap
    )
    fig.suptitle(f"{title}", fontsize=10)

    return fig


def create_cluster_map(
    config: T.mappingConfig, reshaped_pred: np.ndarray, title: str, n_comp: int
) -> Figure:
    """
    Plots clusters spatially on region
    """
    n_comp = np.unique(reshaped_pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(1, 1)
    fig = cast(Figure, fig)
    axis = cast(Axes, axis)

    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis, n_comp, fig)
    fig.suptitle(f"{title} cluster map", fontsize=10)
    return fig


def create_plot_title(
    c: T.clusterConfig, m: T.mappingConfig, description: str = ""
) -> str:
    """
    Creates descriptive plot title for output plots
    """
    # date = pre.get_date(m.keywords)
    print(m.lngRng)
    prob_str = f"Threshold {c.threshold}" if cf["soft_clustering"] else ""
    keyword_str = "_".join(m.keywords)
    return f"{m.name} Lat: {m.latRng[0]} - {m.latRng[1]}, Lon: {360 - m.lngRng[1]} - {360 - m.lngRng[0]}, {c.n_comp} components \n {prob_str} {keyword_str} {description}"
