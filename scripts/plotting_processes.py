from astropy.io import fits
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
import scripts.posterior_prob as PP
from sklearn.metrics import silhouette_score
from config.config import config
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
   
    keywords
        - substrings in radiance array file name to identify file (Ex: PCld)
    latRng
        - two element python list with minimum and maximum latitude specified
    lonRng
        - two element python list with minimum and maximum longitude specified
    n_comp
        - number of clusters for model
    pred
        - one dimensional array of cluster assignments
    input_arr
        - original processed data np array for clustering with indices as last column
    subset_shape
        - dimensions of original spatial subset pixel data 
    param_ranges
        - list of length two lists with desired values ranges for each dimension in clustering, same order as keywords
    threshold, 
        - if soft clustering is enabled, specifies probability threshold for clustering

"""

def create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold,  ROI, description = "", suffix = ""):
   
    cmap = ListedColormap(colors[:n_comp])
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)
    map1 = pre.get_patch(keywords[0], latRng, lngRng, cm_num)
    map2 = pre.get_patch(keywords[1], latRng, lngRng, cm_num)

    MP.create_cluster_map(axis2[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num)
    MP.plot_patch(map1, latRng, lngRng, color_dict[keywords[0]], ROI, axis2[1], param_ranges[0][0],  param_ranges[0][1],  keyword_dict[keywords[0]], cm_num, fig2, True)
    MP.plot_patch(map2, latRng, lngRng, color_dict[keywords[1]], ROI, axis2[2], param_ranges[1][0],  param_ranges[1][1], keyword_dict[keywords[1]], cm_num, fig2, True)
    fig2.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, f"map_comparison_{suffix}", cm_num, threshold)
    fig2.savefig(output_file_name)


def create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold,  cov_type, ROI, description = "", suffix = ""):
    # creates and saves cluster map and cluster scatter plot
    cmap = ListedColormap(colors[:n_comp])
    fig1, axis1 = pl.subplots(2, 1, figsize = (8,8))
    pl.tight_layout()
    fig1.subplots_adjust(
        top = 0.95,
        left = 0.1,
        right = 0.9,
        bottom = 0.1
    )
    MP.create_cluster_map(axis1[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num, fig1, True)
    PL.create_cluster_plot(axis1[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig1.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description), fontsize = 10)
    STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, f"map_plot_{suffix}", cm_num, threshold)
    fig1.savefig(output_file_name)


def create_plots_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cov_type, ROI, description = "", suffix = ""):
    # creates and saves cluster map and cluster scatter plot
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1)
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num, fig, True)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[2],  keywords, 2, 3, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)

    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, f"map_plots_{suffix}", cm_num, threshold)
    fig.savefig(output_file_name) 

def create_cluster_map(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, threshold,  ROI, description = "", suffix = ""):
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num, fig, True)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, f"cluster_map_{suffix}", cm_num, threshold)
    fig.savefig(output_file_name)

def create_cluster_plot(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cov_type, ROI, description = "", suffix = ""):
    cmap = ListedColormap(colors[:n_comp])

    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, f"cluster_plot_{suffix}", cm_num, threshold)
    fig.savefig(output_file_name)

def create_file_name(keywords, latRng, lngRng, n_comp, suffix, cm_num, threshold):
    date = pre.get_date(keywords)
    lat_lon_str = f'{90 - latRng[1]}-{90 - latRng[0]}_{360 - lngRng[1]}-{ 360 - lngRng[0]}'
    keyword_str = '_'.join(keywords)
    prob_str = ("_p" + str(threshold)) if config["soft_clustering"] else ""
    save_path = Path(f'{config["output"]}/{keyword_str}/{n_comp}_cl/{date}_{keyword_str}_{lat_lon_str}_{n_comp}_sys_{cm_num}{prob_str}_{suffix}.png')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    return save_path

def create_plot_title(keywords, latRng, lngRng, n_comp, threshold, description = ""):
    date = pre.get_date(keywords)
    prob_str = f'Threshold {threshold}' if config["soft_clustering"] else ""
    return f'{date} Lat: {90 - latRng[1]} - {90 - latRng[0]}, Lon: {360 - lngRng[1]} - {360 - lngRng[0]}, {n_comp} components \n {prob_str} {description}'