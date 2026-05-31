from astropy.io import fits
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
import scripts.posterior_prob as PP
from sklearn.metrics import silhouette_score
from config.config import cf
import scripts.plots.mapping as MP
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import colormaps
import math
import scripts.cluster_stats as STAT
import pylab as pl
from pathlib import Path
# red, green, blue, yellow, orange, pink, purple, gray
colors =[(1, 0.639, 0.639), (0.647, 1, 0.639), (0.639, 0.894, 1), (1, 0.996, 0.639), (1, 0.82, 0.639), (1, 0.639, 0.839), (0.937, 0.639, 1), (0.678, 0.678, 0.678)]

keyword_dict = {
    "PCld": "Cloud Pressure",
    "AOI": "AOI Index",
    "NH3": "Ammonia Content",
    "CI": "CI Index"
}

color_dict = {
    "PCld": "Blues",
    "NH3": "terrain_r",
    "AOI": "viridis",
    "CI": "cividis"

}

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

def create_map_comp_figure(config, pred, arr, subset_shape, param_ranges, description = ""):
    dim1, dim2 = config.keywords[0], config.keywords[1]


    cmap = ListedColormap(colors[:config.n_comp])
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)
    map1 = pre.get_patch(dim1, config.latRng, config.lngRng, config.cm_num)
    map2 = pre.get_patch(dim2, config.latRng, config.lngRng, config.cm_num)

    MP.create_cluster_map(axis2[0], config.n_comp, pred, arr, subset_shape, config.latRng, config.lngRng, cmap, config.ROI, config.cm_num, fig2)
    MP.plot_patch(map1, config.latRng, config.lngRng, color_dict[dim1], config.ROI, axis2[1], param_ranges[0][0],  param_ranges[0][1],  keyword_dict[dim1], config.cm_num, fig2, True)
    MP.plot_patch(map2, config.latRng, config.lngRng, color_dict[dim2], config.ROI, axis2[2], param_ranges[1][0],  param_ranges[1][1], keyword_dict[dim2], config.cm_num, fig2, True)
    fig2.suptitle(create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, description), fontsize = 10)
    return fig2


def create_plot_figure(config, pred, arr, subset_shape, description = ""):
    # creates and saves cluster map and cluster scatter plot
    cmap = ListedColormap(colors[:config.n_comp])
    fig1, axis1 = pl.subplots(2, 1, figsize = (8,8))
    pl.tight_layout()
    fig1.subplots_adjust(
        top = 0.95,
        left = 0.1,
        right = 0.9,
        bottom = 0.1
    )
    MP.create_cluster_map(axis1[0], config.n_comp, pred, arr, subset_shape, config.latRng, config.lngRng, cmap, config.ROI, config.cm_num, fig1)
    PL.create_cluster_plot(axis1[1],  config.keywords, 0, 1, arr, pred,  config.n_comp, config.cov_type, config.latRng, config.lngRng, cmap, config.cm_num)
    fig1.suptitle(create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, description), fontsize = 10)
    #STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    return fig1


def create_plots_figure(config, pred, input_arr, subset_shape, description = ""):
    # creates and saves cluster map and cluster scatter plot
    cmap = ListedColormap(colors[:config.n_comp])
    fig, axis = pl.subplots(3, 1)
    MP.create_cluster_map(axis[0], config.n_comp, pred, input_arr, subset_shape, config.latRng, config.lngRng, cmap, config.ROI, config.cm_num, fig)
    PL.create_cluster_plot(axis[1],  config.keywords, 0, 1, input_arr, pred,  config.n_comp, config.cov_type, config.latRng, config.lngRng, cmap, config.cm_num)
    PL.create_cluster_plot(axis[2],  config.keywords, 2, 3, input_arr, pred,  config.n_comp, config.cov_type, config.latRng, config.lngRng, cmap, config.cm_num)

    fig.suptitle(create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, description), fontsize = 10)
    return fig

def create_cluster_map(config, pred, input_arr, subset_shape, description = ""):
    cmap = ListedColormap(colors[:config.n_comp])
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, config.n_comp, pred, input_arr, subset_shape, config.latRng, config.lngRng, cmap, config.ROI, config.cm_num, fig)
    fig.suptitle(create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, description), fontsize = 10) 
    return fig

def create_file_prefix(config):
    date = pre.get_date(config.keywords)
    lat_lon_str = f'{90 - config.latRng[1]}-{90 - config.latRng[0]}_{360 - config.lngRng[1]}-{ 360 - config.lngRng[0]}'
    keyword_str = '_'.join(config.keywords)
    #prob_str = ("_p" + str(config.threshold)) if cf["soft_clustering"] else ""
    pca_dir = ("PCA/") if config.isPca else ""
    thresh_dir = (f"{config.threshold_type}_{config.threshold}/") if cf["soft_clustering"] else ""
    save_path = Path(f'{cf["output"]}/{keyword_str}/{pca_dir}{config.n_comp}_cl/{thresh_dir}{date}_{keyword_str}_{lat_lon_str}_{config.n_comp}_sys_{config.cm_num}_')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    return save_path



def create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description = ""):
    date = pre.get_date(keywords)
    prob_str = f'Threshold {threshold}' if cf["soft_clustering"] else ""
    keyword_str = '_'.join(keywords)
    return f'{date} Lat: {90 - latRng[1]} - {90 - latRng[0]}, Lon: {360 - lngRng[1]} - {360 - lngRng[0]}, {n_comp} components \n {prob_str} {keyword_str} {description}'