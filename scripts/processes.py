from astropy.io import fits
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.plotting_processes as PLP
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
import scripts.plotting_processes as PPL
import scripts.pca as PCA





"""
functions to run various clustering --> visualization pipelines

 Parameters
    --------------------------------
    date
        - date observation was taken
    keywords
        - substrings in radiance array file name to identify file (Ex: PCld)
    param_ranges
        - list of length two lists with desired values ranges for each dimension in clustering, same order as keywords
    latRng, OPTIONAL, default = [85, 95]
        - two element python list with minimum and maximum latitude specified
    lonRng, OPTIONAL, default = [230, 330]
        - two element python list with minimum and maximum longitude specified
    n_comp, OPTIONAL, default = 4
        - number of clusters for model
    cov_type, OPTIONAL, default = "full"
        - type of covariance for model
    cm_num, OPTIONAL, default = 1
        - number 1 - 3 specifying longitudinal mapping system
    threshold, OPTIONAL, default = 0.75
        - if soft clustering is enabled, specifies probability threshold for clustering
"""

def output_comparison_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, ROI = {}, cov_type = "full", cm_num = 1, threshold = 0.75):
    

    #creates two visualization files, one is a map comparison, the other has cluster map and scatter plot


    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
  
    pred,probs, *_ = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    # create scatter plots figure
  
    PLP.create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold,  cov_type, ROI)
    PLP.create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold,  ROI)
    indices = input_arr[:, input_arr.shape[1] - 1]
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)
   


def output_cluster_map_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, ROI = {}, cov_type = "full", cm_num = 1, threshold = 0.75):
    
    # creates cluster spatial map and cluster scatter plot
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred, probs, *_ = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    PLP.create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cov_type, ROI)
     # cluster probability maps
    indices = input_arr[:, input_arr.shape[1] - 1]
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)

def output_cluster_map_and_plots(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 240], 
                           n_comp = 4, ROI = {}, cov_type = "full", cm_num = 1, threshold = 0.75):
    # creates spatial cluster map and two cluster scatter plots
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred, probs, *_ = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
  
    indices = input_arr[:, input_arr.shape[1] - 1]


    stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    #create figures
    PLP.create_plots_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cov_type, ROI)
    
    # cluster probability maps
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)


def output_cluster_map(date, keywords, param_ranges, 
                           latRng = [65, 115], lngRng = [0, 50], 
                            n_comp = 4, ROI = {}, cov_type = "full", cm_num = 3, threshold = 0.75):
    
    # creates spatial cluster map
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    indices = input_arr[:, input_arr.shape[1] - 1]

    pred, probs, *_ = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    PLP.create_cluster_map(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, threshold,  ROI)

    # cluster probability maps
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)
   


def output_cluster_plot(date, keywords, param_ranges, 
                           latRng = [90, 95], lngRng = [330, 330], sys = 3,
                           n_comp = 4, cov_type = "full",  cm_num = 3, threshold = 0.75):
    
    # creates cluster scatter plot 
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    indices = input_arr[:, input_arr.shape[1] - 1]
    pred, probs, *_ = CL.create_clusters(input_arr[:, 0: len(keywords)], cov_type, n_comp, config["soft_clustering"])
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    PLP.create_cluster_plot(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cov_type, ROI)

    # cluster probability maps
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)



def create_map_comparison(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 260], 
                            n_comp = 4, ROI = {},cov_type = "full",  cm_num = 1, threshold = 0.75):
    
    # creates spatial cluster map comparison with dimension maps
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    indices = input_arr[:, input_arr.shape[1] - 1]
   
    pred, probs, *_ = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
   
    PLP.create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, ROI)
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold)
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI)


def pca_pipeline(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 260], 
                            n_comp = 4, ROI = {},cov_type = "full",  cm_num = 1, threshold = 0.75):
    # creates spatial cluster map comparison with dimension maps
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    indices = input_arr[:, input_arr.shape[1] - 1]
    [pca_reduced, pca_obj, scaler] = PCA.get_pca_comp(input_arr[:, 0:len(keywords)])
    pred, probs, means = CL.create_clusters(pca_reduced, cov_type, n_comp, config["soft_clustering"], threshold)
    means = pca_obj.inverse_transform(means)
    means = scaler.inverse_transform(means)
    print(means)
    
    #stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    
    PLP.create_cluster_map(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, threshold, ROI, "PCA Reduced", "pca")
    PP.create_uncertainty_fig(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold,  "PCA Reduced", "pca")
    PP.create_max_prob_map(keywords, probs, indices, subset_shape, latRng, lngRng, cm_num, n_comp, threshold, ROI,  "PCA Reduced", "pca")
    heat_map_fig = PCA.get_loadings_heatmap(pca_obj, keywords)
    heat_map_fig.savefig(PLP.create_file_name(keywords, latRng, lngRng, n_comp, "pca_heat_map", cm_num, threshold))
    centroids_fig = STAT.get_centroids_figure(keywords, means)
    centroids_fig.savefig(PLP.create_file_name(keywords, latRng, lngRng, n_comp, "pca_centroids", cm_num, threshold))

   

# Run processes for longitude 0 - 30, latitude 90 - 105
lon_range = [0, 30]
lat_range = [0, 15]

lon_range = [360 - lon_range[1], 360 - lon_range[0]]
lat_range = [90 - lat_range[1], 90 - lat_range[0]]

#output_comparison_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3100]], lat_range, lon_range, 4)


#First two elements are the north and south colatitudes. Third is the central longitude and the fourth with the longitude halfwidth

ROI={"Hot Spot":[82,83,14.0,2.0],
                     "Gyre":[84,86,15.0,3.0],
                     "Cloud Plume":[82,84,5.0,3.0],
                     "Reference":[76,78,15,4.0]} 

# output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3100], [0.1, 0.4], [0.3, 0.8]], lat_range, lon_range, 5)

#output_cluster_map_and_plot("20251016", ["AOI", "CI"], [[0.1, 0.4], [0.3, 0.8]], lat_range, lon_range, 4, ROI)

#output_cluster_map_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3100]], lat_range, lon_range, 4, ROI)


# output_cluster_map("20251016",
#                        ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
#                       [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], 
#                       lat_range, lon_range, 8, ROI, "full", 1)
                      

pca_pipeline("20251016",
                       ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
                      [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], 
                      lat_range, lon_range, 8, ROI, "full", 1)
                      