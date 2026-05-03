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
import pylab as pl



def output_cluster_map_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, cov_type = "full", cm_num = 1, threshold = 0.75):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(2, 1)
    print("plotting...")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold))
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plot", cm_num, threshold)
    fig.savefig(output_file_name)

def output_cluster_map_and_plots(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 240], 
                           n_comp = 4, cov_type = "full", cm_num = 1, threshold = 0.95):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    #sil_score = str(round(silhouette_score(input_arr, pred), 3))
    fig, axis = pl.subplots(3, 1)
    print("plotting...")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[2],  keywords, 2, 3, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)

    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold))
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plots", cm_num, threshold)
    fig.savefig(output_file_name) 


def output_cluster_map(date, keywords, param_ranges, 
                           latRng = [65, 115], lngRng = [0, 50], 
                            n_comp = 4, cov_type = "full", cm_num = 3, threshold = 0.75):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    #sil_score = str(round(silhouette_score(input_arr, pred), 3))
    #colors = np.random.rand(n_comp, 3)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    print("plotting...")
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold))
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "cluster_map", cm_num, threshold)
    fig.savefig(output_file_name)


def output_cluster_plot(date, keywords, param_ranges, 
                           latRng = [90, 95], lngRng = [330, 330], sys = 3,
                           n_comp = 4, cov_type = "full",  cm_num = 3, threshold = 0.75):
    
   
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0: len(keywords)], cov_type, n_comp, config["soft_clustering"])
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = ["red", "green", "blue", "yellow"]
    cmap = ListedColormap(colors[:n_comp])
    print("plotting...")
    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold))
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "cluster plot", cm_num, threshold)
    fig.savefig(output_file_name)


def create_file_name(keywords, latRng, lngRng, n_comp, suffix, cm_num, threshold):
    date = pre.get_date(keywords)
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    keyword_str = '_'.join(keywords)
    prob_str = str(threshold) if config["soft_clustering"] else ""
    return f'{config["output"]}/{date}_{keyword_str}_{lat_lon_str}_{n_comp}_sys_{cm_num}_p{threshold}_{suffix}.png'

def create_plot_title(keywords, latRng, lngRng, n_comp, threshold):
    date = pre.get_date(keywords)
    prob_str = f'Threshold {threshold}' if config["soft_clustering"] else ""
    return f'{date} Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components \n {prob_str}'
  #longitude range 230 - 330 for sys 1  
def create_map_comparison(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 260], 
                            n_comp = 4, cov_type = "full",  cm_num = 1, threshold = 0.95):
    
  
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    #print("scoring...")
    #sil_score = str(round(silhouette_score(input_arr, pred), 3))
    #print(sil_score)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[: n_comp])
    print("start fig")
    fig, axis = pl.subplots(3, 1, constrained_layout = True)
    
    fig.tight_layout(rect = [0, 0, 1, 0.95])
    cloud_map = pre.get_patch(keywords[0], latRng, lngRng, cm_num)
    amm_map = pre.get_patch(keywords[1], latRng, lngRng, cm_num)
    print("plotting")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num, fig, True)
    MP.plot_patch(cloud_map, latRng, lngRng, "Blues", axis[1], param_ranges[0][0], param_ranges[0][1],  "Cloud Pressure", cm_num, fig, True)
    MP.plot_patch(amm_map, latRng, lngRng, "terrain_r", axis[2], param_ranges[1][0], param_ranges[1][1], "Ammonia", cm_num, fig, True)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold))
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_comparison", cm_num, threshold)
    fig.savefig(output_file_name)


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#get correct filters
#create_map_comparison("20251016", ["NH3", "PCld", "AOI", "CI"], [[100, 300], [1500,  2500], [0, 1], [0, 1]], [75, 105], [0, 200])#
#output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0, 1], [0, 1]], [75, 105], [0, 200], 4)
#output_cluster_map_and_plots("20251016", ["PCld", "NH3", "AOI", "CI"], [[1000, 3000], [0,  300], [0, 1], [0, 1]], [85, 95], [0, 50], 4)
output_cluster_map_and_plots("20251016", ["PCld", "NH3", "AOI", "CI"], [[1000, 3000], [0,  300], [0, 1], [0, 1]], [75, 105], [50, 100], 4)

#create_map_comparison("20251016", ["NH3", "PCld"], [[100, 300], [1500,  2500]], [75, 105], [0, 200])
#create_map_comparison("20251016", ["PCld", "NH3"], [[1000,  3000], [0, 300]], [75, 105], [0, 200])

#create_map_comparison("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200])
#output_cluster_map_and_plot("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200], 4)


#output_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
    