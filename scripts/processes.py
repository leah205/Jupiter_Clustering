from astropy.io import fits
import numpy as np
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config
import scripts.plots.mapping as MP
import matplotlib
from matplotlib.colors import ListedColormap
import scripts.cluster_stats as STAT
import pylab as pl
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
    cmap, 
        color map object for clusters
"""

def create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, ROI):
   
  
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)
    map1 = pre.get_patch(keywords[0], latRng, lngRng, cm_num)
    map2 = pre.get_patch(keywords[1], latRng, lngRng, cm_num)

    MP.create_cluster_map(axis2[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num)
    MP.plot_patch(map1, latRng, lngRng, color_dict[keywords[0]], axis2[1], param_ranges[0][0], ROI, param_ranges[0][1],  keyword_dict[keywords[0]], cm_num, fig2, True)
    MP.plot_patch(map2, latRng, lngRng, color_dict[keywords[1]], axis2[2], param_ranges[1][0], ROI, param_ranges[1][1], keyword_dict[keywords[1]], cm_num, fig2, True)
    fig2.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_comparison", cm_num, threshold)
    fig2.savefig(output_file_name)


def create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, cov_type):
    # creates and saves cluster map and cluster scatter plot
    
    fig1, axis1 = pl.subplots(2, 1)
    MP.create_cluster_map(axis1[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis1[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig1.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plot", cm_num, threshold)
    fig1.savefig(output_file_name)


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

    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    # create scatter plots figure
  
    cmap = ListedColormap(colors[:n_comp])
    create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, cov_type)
    create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, ROI)
   


def output_cluster_map_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, ROI = {}, cov_type = "full", cm_num = 1, threshold = 0.75):
    
    # creates cluster spatial map and cluster scatter plot
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(2, 1)
    fig.subplots_adjust(hspace=0.4)
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plot", cm_num, threshold)
    fig.savefig(output_file_name)

def output_cluster_map_and_plots(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 240], 
                           n_comp = 4, ROI = {}, cov_type = "full", cm_num = 1, threshold = 0.75):
    # creates spatial cluster map and two cluster scatter plots
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(3, 1)
  
    indices = input_arr[:, input_arr.shape[1] - 1]


    stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    #create figures
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[2],  keywords, 2, 3, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)

    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plots", cm_num, threshold)
    fig.savefig(output_file_name) 


def output_cluster_map(date, keywords, param_ranges, 
                           latRng = [65, 115], lngRng = [0, 50], 
                            n_comp = 4, ROI = {}, cov_type = "full", cm_num = 3, threshold = 0.75):
    
    # creates spatial cluster map
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
   
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    cmap = ListedColormap(colors[:n_comp])

    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, ROI, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "cluster_map", cm_num, threshold)
    fig.savefig(output_file_name)


def output_cluster_plot(date, keywords, param_ranges, 
                           latRng = [90, 95], lngRng = [330, 330], sys = 3,
                           n_comp = 4, cov_type = "full",  cm_num = 3, threshold = 0.75):
    
    # creates cluster scatter plot 
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0: len(keywords)], cov_type, n_comp, config["soft_clustering"])
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    cmap = ListedColormap(colors[:n_comp])

    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "cluster plot", cm_num, threshold)
    fig.savefig(output_file_name)


def create_file_name(keywords, latRng, lngRng, n_comp, suffix, cm_num, threshold):
    date = pre.get_date(keywords)
    lat_lon_str = f'{90 - latRng[1]}-{90 - latRng[0]}_{360 - lngRng[1]}-{ 360 - lngRng[0]}'
    keyword_str = '_'.join(keywords)
    prob_str = ("_p" + str(threshold)) if config["soft_clustering"] else ""
    return f'{config["output"]}/{date}_{keyword_str}_{lat_lon_str}_{n_comp}_sys_{cm_num}{prob_str}_{suffix}.png'

def create_plot_title(keywords, latRng, lngRng, n_comp, threshold):
    date = pre.get_date(keywords)
    prob_str = f'Threshold {threshold}' if config["soft_clustering"] else ""
    return f'{date} Lat: {90 - latRng[1]} - {90 - latRng[0]}, Lon: {360 - lngRng[1]} - {360 - lngRng[0]}, {n_comp} components \n {prob_str}'

def create_map_comparison(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 260], 
                            n_comp = 4, ROI = {},cov_type = "full",  cm_num = 1, threshold = 0.75):
    
    # creates spatial cluster map comparison with dimension maps
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)

    indices = input_arr[:, input_arr.shape[1] - 1]
   
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    cmap = ListedColormap(colors[: n_comp])
    create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, ROI)
   


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#create_map_comparison("20251016", ["NH3", "PCld", "AOI", "CI"], [[100, 300], [1500,  2500], [0, 1], [0, 1]], [75, 105], [0, 200])#

# physical parameter/Index clustering longitude 0 - 200, latitude 75 - 105

#output_comparison_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3000]], [75, 105], [0, 200], 4)
#create_map_comparison("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3000]], [75, 105], [0, 200], 4)
#output_comparison_and_plot("20251016", ["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], [75, 105], [0, 200], 4)

#output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0.1, 0.4], [0.4, 0.8]], [75, 105], [0, 200], 4)
#output_cluster_map_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3000]], [75, 105], [0, 200], 4)

#RGB clustering longitude 0 - 200, latitude 75-105
# add rgb context (map comparison)
"""output_cluster_map("20251016",
                       ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
                      [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], 
                      [75, 105], [0, 200], 6, "full", 1)"""


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

output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3100], [0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, 5, ROI)

#output_cluster_map_and_plot("20251016", ["AOI", "CI"], [[0.1, 0.4], [0.4, 0.8]], lat_range, lon_range, 4, ROI)

#output_cluster_map_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3100]], lat_range, lon_range, 4, ROI)


# output_cluster_map("20251016",
#                        ["275", "395", "502", "619", "631", "645", "673", "727", "889"], 
#                       [[0, 1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1], [0,  1]], 
#                       lat_range, lon_range, 8, ROI, "full", 1)
                      

