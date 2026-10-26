from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_pca_comp(X: np.ndarray, var_threshold=0.98) -> list:
    """
    Purpose
    ----------
    Finds significant principal component axes and returns reduced dimensional data

    Parameters
    --------
    X,
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances

    var_threshold
    --------
    threshold for cumulative explained variance to choose number of components

    Returns
    ---------
    python list
        First element: numpy array with axis 0 as pixels within lon/lat range and axis 1 as pixel radiances in new dimensions
        Second Element: fitted PCA object

    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=var_threshold)
    X_transformed = pca.fit_transform(X_scaled)
    # gets array of percentage of variance explained by each component
    print("variances")
    print(pca.explained_variance_ratio_)
    return [X_transformed, pca, scaler]


def get_loadings_heatmap(pca_obj, keywords: list[str], title: str) -> Figure:
    """
    Purpose
    ----------------


    Parameters
    ---------------
    pca_obj
        fitted pca object

    """
    loadings = pca_obj.components_.T * np.sqrt(pca_obj.explained_variance_)
    print("pca loadings:")
    print(loadings)
    fig, ax = plt.subplots(1, 1)
    fig = cast(Figure, fig)
    ax: Axes = ax

    ax.set_title("PCA Loadings Heat Map")
    sns.heatmap(loadings, ax=ax, annot=True, cmap="coolwarm", yticklabels=keywords)
    fig.suptitle(title, fontsize=10)
    return fig
