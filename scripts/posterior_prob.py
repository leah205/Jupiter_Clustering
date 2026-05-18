from astropy.io import fits
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config
import scripts.plots.mapping as MP
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import colormaps
import math
import scripts.cluster_stats as STAT
import pylab as pl
import scripts.plotting_processes as PLP
# red, green, blue, yellow, orange, pink, purple, gray
colors =[(1, 0.639, 0.639), (0.647, 1, 0.639), (0.639, 0.894, 1), (1, 0.996, 0.639), (1, 0.82, 0.639), (1, 0.639, 0.839), (0.937, 0.639, 1), (0.678, 0.678, 0.678)]

# uncertainty maps

def create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold):
     """
     Purpose
     --------------

     Creates a plot for each cluster with the probability of that cluster for each pixel

     Parameters
     
     """
     ncols = 3
     nrows = math.ceil(n_comp / ncols)
     fig, ax = pl.subplots(nrows, ncols)
     ax = np.atleast_1d(ax).flatten()

     for i in range(0, n_comp):
        # cmap = LinearSegmentedColormap.from_list("cluster_color", [(1, 1, 1, 0), colors[i]], N = 256)
        cmap = LinearSegmentedColormap.from_list("cluster_color", ["white", "yellow", "orange", "red"], N = 256)

        prob_map = MP.create_cluster_arr(indices, subset_shape, probs[:, i])
        cbar = True if i == n_comp - 1 else False
        annotated = True if i == 0 else False
        MP.plot_patch(prob_map, latRng, lngRng, cmap, {}, ax[i], 0, 1, "", cm_num, fig, cbar, annotated, "probability")
        ax[i].set_title(f"Cluster {i}")
    
     for a in ax[n_comp:]: 
         fig.delaxes(a)
     fig.suptitle(PLP.create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
     output_file_name = PLP.create_file_name(keywords, latRng, lngRng, n_comp, "posterior_probs", cm_num, threshold)
     fig.savefig(output_file_name)

def create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold):
    """
     Purpose
     --------------

     Creates a map with the maximum probability for each pixel 
     """
    fig, ax = pl.subplots(1, 1)
    max_probs = np.max(probs, axis = 1)
    cmap = LinearSegmentedColormap.from_list("cluster_color", ["white", "yellow", "orange", "red"], N = 256)
    max_prob_map = MP.create_cluster_arr(indices, subset_shape, max_probs)
    MP.plot_patch(max_prob_map, latRng, lngRng, cmap, {}, ax, 0, 1, "", cm_num, fig, True, True, "probability")
    fig.suptitle(PLP.create_plot_title(keywords, latRng, lngRng, n_comp, threshold, "maximum probability map"), fontsize = 10)
    output_file_name = PLP.create_file_name(keywords, latRng, lngRng, n_comp, "max_prob_map", cm_num, threshold)
    fig.savefig(output_file_name)


   
    