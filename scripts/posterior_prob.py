
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
import scripts.plots.mapping as MP
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import math
import pylab as pl
import scripts.plotting_processes as PLP
# red, green, blue, yellow, orange, pink, purple, gray
colors =[(1, 0.639, 0.639), (0.647, 1, 0.639), (0.639, 0.894, 1), (1, 0.996, 0.639), (1, 0.82, 0.639), (1, 0.639, 0.839), (0.937, 0.639, 1), (0.678, 0.678, 0.678)]

# uncertainty maps

def create_uncertainty_fig(config, probs, indices, subset_shape):
     """
     Purpose
     --------------

     Creates a plot for each cluster with the probability of that cluster for each pixel

     Parameters
     
     """
     ncols = 3
     nrows = math.ceil(config.n_comp / ncols)
     fig, ax = pl.subplots(nrows, ncols, constrained_layout = True)
     ax = np.atleast_1d(ax).flatten()
    
     for i in range(0, config.n_comp):
        # cmap = LinearSegmentedColormap.from_list("cluster_color", [(1, 1, 1, 0), colors[i]], N = 256)
        cmap = LinearSegmentedColormap.from_list("cluster_color", ["white", "yellow", "orange", "red"], N = 256)

        prob_map = MP.create_cluster_arr(indices, subset_shape, probs[:, i])
        cbar = True if i == config.n_comp - 1 else False
        #annotated = True if i == 0 else False
        annotated = True
        MP.plot_patch(prob_map, config.latRng, config.lngRng, cmap, {}, ax[i], 0, 1, "", config.cm_num, fig, cbar, annotated, "probability")
        ax[i].set_title(f"Cluster {i + 1}")
    
     for a in ax[config.n_comp:]: 
         fig.delaxes(a)
     fig.suptitle(PLP.create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, "posterior probability of clusters"), fontsize = 10)
     return fig

def create_max_prob_map(config, probs, indices, subset_shape):
    """
     Purpose
     --------------

     Creates a map with the maximum probability for each pixel 
     """
    fig, ax = pl.subplots(1, 1)
    max_probs = np.max(probs, axis = 1)
    mean = round(np.mean(max_probs), 3)
    cmap = LinearSegmentedColormap.from_list("cluster_color", ["white", "yellow", "orange", "red"], N = 256)
    max_prob_map = MP.create_cluster_arr(indices, subset_shape, max_probs)
    MP.plot_patch(max_prob_map, config.latRng, config.lngRng, cmap, config.ROI, ax, 0, 1, "", config.cm_num, fig, True, True, "probability")
    fig.suptitle(PLP.create_plot_title(config.keywords, config.latRng, config.lngRng, config.n_comp, config.threshold, f"maximum probability map"), fontsize = 10)
    ax.set_title(f"Mean Probability: {mean}")
    return fig


   
    