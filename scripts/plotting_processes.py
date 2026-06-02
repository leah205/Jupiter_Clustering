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
import scripts.types as T
# red, green, blue, yellow, orange, pink, purple, gray
colors =[(1, 0.639, 0.639), (0.647, 1, 0.639), (0.639, 0.894, 1), (1, 0.996, 0.639), (1, 0.82, 0.639), (1, 0.639, 0.839), (0.937, 0.639, 1), (0.678, 0.678, 0.678)]



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

def create_map_comp_figure(config: T.mappingConfig, reshaped_pred, title):
    n_comp = np.unique(reshaped_pred).shape[0] - 1
    dim1, dim2 = config.keywords[0], config.keywords[1]
    cmap = ListedColormap(colors[:n_comp])
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)
    map1 = pre.get_patch(dim1, config.latRng, config.lngRng, config.cm_num)
    map2 = pre.get_patch(dim2, config.latRng, config.lngRng, config.cm_num)

    MP.plot_cluster_patch(config, reshaped_pred, cmap,  axis2[0], n_comp,fig2)
    
    MP.plot_patch(config, map1, dim1,  axis2[1], fig2)
    MP.plot_patch(config, map2, dim2, axis2[2], fig2)
    fig2.suptitle(f"{title}", fontsize = 10) 

    return fig2


def create_plot_figure(config: T.mappingConfig, pred, arr, reshaped_pred, title, cluster_obj):
    # creates and saves cluster map and cluster scatter plot
    n_comp = np.unique(pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig1, axis1 = pl.subplots(2, 1, figsize = (8,8))
    pl.tight_layout()
    fig1.subplots_adjust(
        top = 0.95,
        left = 0.1,
        right = 0.9,
        bottom = 0.1
    )
    MP.plot_cluster_patch(config, reshaped_pred,  cmap, axis1[0],  n_comp, fig1)
    PL.create_cluster_plot(axis1[1],  config.keywords, 0, 1, arr, pred, n_comp, cmap)
    PL.plot_gmm_ellipsoids(axis1[1], cluster_obj, cmap)
    fig1.suptitle(f"{title}", fontsize = 10) 
    #STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    return fig1


def create_plots_figure(config: T.mappingConfig, pred, input_arr, reshaped_pred, title):
    # creates and saves cluster map and cluster scatter plot
    n_comp = np.unique(pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1)
    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis[0], n_comp,  fig)
    PL.create_cluster_plot(axis[1],  config.keywords, 0, 1, input_arr, pred,  n_comp, config.latRng, config.lngRng, cmap, config.cm_num)
    PL.create_cluster_plot(axis[2],  config.keywords, 2, 3, input_arr, pred,  n_comp, cmap)
    fig.suptitle(f"{title}", fontsize = 10) 

    return fig

def create_cluster_map(config: T.mappingConfig, reshaped_pred, title):
    n_comp = np.unique(reshaped_pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(1, 1)
    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis, n_comp, fig)
    fig.suptitle(f"{title} cluster map", fontsize = 10) 
    return fig

def create_file_prefix(c: T.clusterConfig, m: T.mappingConfig):
    date = pre.get_date(m.keywords)
    lat_lon_str = f'{90 - m.latRng[1]}-{90 - m.latRng[0]}_{360 - m.lngRng[1]}-{ 360 - m.lngRng[0]}'
    keyword_str = '_'.join(m.keywords)
    #prob_str = ("_p" + str(config.threshold)) if cf["soft_clustering"] else ""
    pca_dir = ("PCA/") if c.isPca else ""
    thresh_dir = (f"{c.threshold_type}_{c.threshold}/") if cf["soft_clustering"] else ""
    save_path = Path(f'{cf["output"]}/{keyword_str}/{pca_dir}{c.n_comp}_cl/{thresh_dir}{date}_{keyword_str}_{lat_lon_str}_{c.n_comp}_sys_{m.cm_num}_')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    return save_path



def create_plot_title(c: T.clusterConfig, m: T.mappingConfig, description = ""):
    date = pre.get_date(m.keywords)
    prob_str = f'Threshold {c.threshold}' if cf["soft_clustering"] else ""
    keyword_str = '_'.join(m.keywords)
    return f'{date} Lat: {90 - m.latRng[1]} - {90 - m.latRng[0]}, Lon: {360 - m.lngRng[1]} - {360 - m.lngRng[0]}, {c.n_comp} components \n {prob_str} {keyword_str} {description}'