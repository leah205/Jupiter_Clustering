
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
from config.config import cf
import scripts.plots.mapping as MP
import pylab as pl
from pathlib import Path
from matplotlib.colors import ListedColormap 
from scripts.helpers import get_dir_path
import config.types as T
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
    """
    Creates spatial map comparing radiances in different filters with cluster outputs
    """
    n_comp = np.unique(reshaped_pred).shape[0] - 1
    dim1, dim2 = config.keywords[0], config.keywords[1]
    cmap = ListedColormap(colors[:n_comp])
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)

    dir_path = get_dir_path(config)
    map1 = pre.get_patch(dim1, config.latRng, config.lngRng, config.cm_num, dir_path)
    map2 = pre.get_patch(dim2, config.latRng, config.lngRng, config.cm_num, dir_path)

    MP.plot_cluster_patch(config, reshaped_pred, cmap,  axis2[0], n_comp,fig2)
    
    MP.plot_patch(config, map1, dim1,  axis2[1], fig2)
    MP.plot_patch(config, map2, dim2, axis2[2], fig2)
    fig2.suptitle(f"{title}", fontsize = 10) 

    return fig2


def create_plot_figure(config: T.mappingConfig, pred, arr, reshaped_pred, title, cluster_obj):
    """
    Creates scatter plot of clusters in two parameters and spatial cluster map
    """
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
    """
    Creates two scatter plots each showing clusters with two parameters and a spatial cluster map
    """
    # creates and saves cluster map and cluster scatter plot
    n_comp = np.unique(pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1)
    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis[0], n_comp,  fig)
    PL.create_cluster_plot(axis[1],  config.keywords, 0, 1, input_arr, pred,  n_comp, cmap)
    PL.create_cluster_plot(axis[2],  config.keywords, 2, 3, input_arr, pred,  n_comp, cmap)
    fig.suptitle(f"{title}", fontsize = 10) 

    return fig

def create_cluster_map(config: T.mappingConfig, reshaped_pred, title):
    """
    Plots clusters spatially on region
    """
    n_comp = np.unique(reshaped_pred).shape[0] - 1
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(1, 1)
    MP.plot_cluster_patch(config, reshaped_pred, cmap, axis, n_comp, fig)
    fig.suptitle(f"{title} cluster map", fontsize = 10) 
    return fig

def create_file_prefix(c: T.clusterConfig, m: T.mappingConfig):
    """
    Creates file path name and prefix for plotting output
    """
    print(c)
    print(m)
    dir_name = get_dir_path(m)
    date = pre.get_date(m.keywords, dir_name)
    lat_lon_str = f'{m.latRng[0]}-{m.latRng[1]}_{m.lngRng[0]}-{m.lngRng[1]}'
    keyword_str = '_'.join(m.keywords)
    pca_dir = ("PCA/") if c.isPca else ""
    thresh_dir = (f"{c.threshold_type}_{c.threshold}/") if cf["soft_clustering"] else ""
    lon_str = f"lon_{m.lngRng[0]}-{m.lngRng[1]}"
    save_path = Path(f'{cf["output"]}/{m.name}/{keyword_str}/{pca_dir}{c.n_comp}_cl/{thresh_dir}{date}_{keyword_str}_{lat_lon_str}_{c.n_comp}_sys_{m.cm_num}_')
    save_path.parent.mkdir(parents = True, exist_ok = True)
    return save_path



def create_plot_title(c: T.clusterConfig, m: T.mappingConfig, description = ""):
    """
    Creates descriptive plot title for output plots
    """
    #date = pre.get_date(m.keywords)
    print(m.lngRng)
    prob_str = f'Threshold {c.threshold}' if cf["soft_clustering"] else ""
    keyword_str = '_'.join(m.keywords)
    return f'{m.name} Lat: {m.latRng[0]} - {m.latRng[1]}, Lon: {m.lngRng[0]} - {m.lngRng[1]}, {c.n_comp} components \n {prob_str} {keyword_str} {description}'