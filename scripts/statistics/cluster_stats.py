from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy._typing._array_like import NDArray
from sklearn.linear_model import LinearRegression

import config.types as T

keyword_dict = {
    "NH3": "Ammonia Mole Fraction (ppm)",
    "PCld": "Cloud Pressure (mb)",
    "AOI": "Altitude Opacity Index",
    "CI": "Color Index",
}


def get_cluster_regressions(
    data: np.ndarray, probs: np.ndarray, n_clusters: int
) -> dict:
    """
    Gets object containing  cluster regressions for all clusters

     Parameters
     ----------------
     data: np array
         - radiance data for pixels
     probs: np array
         - two d array of probabilities that each pixel belongs to each cluster (pixels on axis 0)
     n_cluster: int
         - number of clusters

     Returns
     ----------------
     dictionary containing entries of form coeff_n and b_n, characterizing linear regressions for each cluster n


    """
    res = {}
    for i in range(n_clusters):
        res = res | run_cluster_regression(data, probs, i)
    return res


def run_cluster_regression(data: np.ndarray, probs: np.ndarray, i: int) -> dict:
    """

    Runs linear regression on specific cluster

    Parameters
        ----------------
        data: np array
            - radiance data for pixels
        probs: np array
            - two d array of probabilities that each pixel belongs to each cluster (pixels on axis 0)
        i: int
            - cluster to run regression
    Returns
    ------------------

    dict with keys coeff_i and b_i for linear regression outputs


    """

    # covariances, and linear regression for NH3/pcld
    # get correlation
    # assumes nh3 is index 0, pcld at index 1
    # only runs if two dimensional

    res = {}
    x, y = data[:, 0].reshape(-1, 1), data[:, 1]

    m = LinearRegression()
    m.fit(x, y, sample_weight=probs[:, i])

    # coeff_list = list(np.round(m.coef_, 3))
    # coeoff_strs = [f"{:.3f}".format(coeff) for coeff in coeff_list]
    return {f"coeff_{i + 1}": f"{m.coef_[0]:.3f}", f"b_{i + 1}": f"{m.intercept_:.3f}"}


def get_all_stats(
    cluster_obj,
    indices: np.ndarray,
    n_comp: int,
    map_config: T.mappingConfig,
    data: np.ndarray,
) -> dict:
    """
    Purpose: Preprocesses data and passes it into a function to generate list of means and standard deviations for the clusters

    Parameters
    ----------------
    cluster_arr, TYPE python list of clusters
    dim_arr, TYPE 1D numpy array
    indices, TYPE 1D numpy array of floats
    n_clusters, TYPE Integer

    Returns
    --------------
    List of Lists of mean and standard deviation pairs for each cluster for each dimension

    """

    pred, probs, covariances, means = (
        cluster_obj["pred"],
        cluster_obj["probs"],
        cluster_obj["covariances"],
        cluster_obj["means"],
    )

    # stds = [np.sqrt(np.diag(cov_m)) for cov_m in covariances]
    stds: NDArray[np.float64] = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    cluster_arr = np.array(pred)
    indices = indices.astype(np.int64)
    if not (cluster_arr.shape == indices.shape) and cluster_arr.ndim == 1:
        raise TypeError("Index array length must match cluster array length")

    res = {}

    for i, key in enumerate(map_config.keywords):
        cluster_means: npt.NDArray[np.float64] = means[:, i]
        cluster_stds: npt.NDArray[np.float64] = stds[:, 1]
        res[key] = {}
        for cluster in range(n_comp):
            res[key][f"{cluster + 1}"] = {
                "mean": str(cluster_means[i]),
                "standard deviation": str(cluster_stds[i]),
            }

    # first keyword is predictor, second is predictee
    if len(map_config.keywords) == 2:
        res = res | get_cluster_regressions(data, probs, n_comp)

    res["covariances"] = f"{covariances.tolist()}"

    print(res)

    return res


def reassign_clusters(
    pred: np.ndarray, stats: np.ndarray, param_ranges: list[float]
) -> np.ndarray:
    """
    Parameters
    -----------
    pred
        np array of cluster assignments
    stats
        np array of means with axis 0 as clusters and axis 1 as dimension


    """

    cluster_means = quantify_clusters(stats, param_ranges)
    res = np.full(pred.shape, -1)

    unique, counts = np.unique(pred, return_counts=True)

    mask = ~(pred == -1)
    indices = np.argsort(cluster_means)
    res[mask] = indices[pred[mask]]

    return res


def quantify_clusters(stats: np.ndarray, param_ranges: list[float]) -> float:
    # convert param_ranges to numpy array
    """
    Purpose:  Find the mean of all means of clusters in all dimensions

    Parameters:
    stats - nd array, first axis is dimension, second axis, is cluster means for that dimension
    param_ranges - similiarly ordered cross-dimensional array with min and max value for each dimension
    """

    param_swapped = np.swapaxes(param_ranges, 0, 1)
    normed_stats = (stats - param_swapped[0]) / (param_swapped[1] - param_swapped[0])
    return np.mean(normed_stats, axis=1)


def get_centroids_figure(
    keywords: list[str], cluster_centroids: np.ndarray, title: str
) -> Figure:
    """
    Purpose
    --------------
    Gets a table of the means of each cluster in each original dimension

    Parameters
    --------------
    keywords
        array of dimension names
    cluster_centroid
        2D np array with axis 0 as clusters, axis 1 as dimensions (same order as keywords)

    Returns
    --------------
    Centroids table figure for each cluster and dimension

    """
    fig, ax = plt.subplots(1, 1)
    fig = cast(Figure, fig)
    ax = cast(Axes, ax)

    # check if dimension is wavelength (valid number)
    if keywords[0].isdigit():
        ax.set_title("Mean Cluster Reflectance per Wavelength")
    else:
        ax.set_title("Mean Cluster Value per Parameter")
    c_num = cluster_centroids.shape[0]
    cluster_labels = np.arange(1, c_num + 1).tolist()
    sns.heatmap(
        cluster_centroids,
        ax=ax,
        annot=True,
        cmap="coolwarm",
        xticklabels=keywords,
        yticklabels=cluster_labels,
    )
    fig.suptitle(title, fontsize=10)
    return fig
