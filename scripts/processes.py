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

colors =["red", "green", "blue", "yellow", "orange", "pink"]

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

def create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap):
    # create map comparison figure
  
    fig2, axis2 = pl.subplots(3, 1, constrained_layout = True)
    map1 = pre.get_patch(keywords[0], latRng, lngRng, cm_num)
    map2 = pre.get_patch(keywords[1], latRng, lngRng, cm_num)
    MP.create_cluster_map(axis2[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    MP.plot_patch(map1, latRng, lngRng, color_dict[keywords[0]], axis2[1], param_ranges[0][0], param_ranges[0][1],  keyword_dict[keywords[0]], cm_num, fig2, True)
    MP.plot_patch(map2, latRng, lngRng, color_dict[keywords[1]], axis2[2], param_ranges[1][0], param_ranges[1][1], keyword_dict[keywords[1]], cm_num, fig2, True)
    fig2.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_comparison", cm_num, threshold)
    fig2.savefig(output_file_name)


def create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, cov_type):
    fig1, axis1 = pl.subplots(2, 1)
    MP.create_cluster_map(axis1[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis1[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig1.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    STAT.get_all_stats(pred, keywords, input_arr[:, input_arr.shape[1] - 1], n_comp, latRng, lngRng, cm_num)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plot", cm_num, threshold)
    fig1.savefig(output_file_name)


def output_comparison_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, cov_type = "full", cm_num = 1, threshold = 0.75):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    
    # create scatter plots figure
    
    print("plotting...")
    cmap = ListedColormap(colors[:n_comp])
    create_plot_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap, cov_type)
    create_map_comp_figure(keywords, latRng, lngRng, cm_num, n_comp, pred, input_arr, subset_shape, param_ranges, threshold, cmap)
   


def output_cluster_map_and_plot(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, cov_type = "full", cm_num = 1, threshold = 0.75):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(2, 1)
    print("plotting...")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_plot", cm_num, threshold)
    fig.savefig(output_file_name)

def output_cluster_map_and_plots(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 240], 
                           n_comp = 4, cov_type = "full", cm_num = 1, threshold = 0.75):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"], threshold)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    #sil_score = str(round(silhouette_score(input_arr, pred), 3))
    fig, axis = pl.subplots(3, 1)
    print("plotting...")
    indices = input_arr[:, input_arr.shape[1] - 1]


    stats = STAT.get_all_stats(pred, keywords, indices, n_comp, latRng, lngRng, cm_num)
    pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))
    
    #create figures
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[2],  keywords, 2, 3, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)

    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
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
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
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
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
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
    return f'{date} Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]}, {n_comp} components \n {prob_str}'
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
    fig.suptitle(create_plot_title(keywords, latRng, lngRng, n_comp, threshold), fontsize = 10)
    output_file_name = create_file_name(keywords, latRng, lngRng, n_comp, "map_comparison", cm_num, threshold)
    fig.savefig(output_file_name)


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#get correct filters
#create_map_comparison("20251016", ["NH3", "PCld", "AOI", "CI"], [[100, 300], [1500,  2500], [0, 1], [0, 1]], [75, 105], [0, 200])#

# longitude 0 - 200, latitude 75 - 105

#output_comparison_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3000]], [75, 105], [0, 200], 4)
#output_comparison_and_plot("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200], 3)


#output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[0, 300], [1000,  3000], [0, 1], [0, 1]], [75, 105], [0, 200], 3)
output_cluster_map_and_plot("20251016", ["NH3", "PCld"], [[0, 300], [1000,  3000]], [75, 105], [0, 200], 4)
#create_map_comparison("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200])
#output_cluster_map_and_plot("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200], 4)

#output_cluster_map_and_plots("20251016", ["PCld", "NH3", "AOI", "CI"], [[1000, 3000], [0,  300], [0, 1], [0, 1]], [85, 95], [0, 50], 4)
#output_cluster_map_and_plots("20251016", ["PCld", "NH3", "AOI", "CI"], [[1000, 3000], [0,  300], [0, 1], [0, 1]], [75, 105], [0, 300], 4)
#create_map_comparison("20251016", ["PCld", "NH3", "AOI", "CI"], [[1000, 3000], [0,  300], [0, 1], [0, 1]], [75, 105], [0, 300], 4)

#create_map_comparison("20251016", ["NH3", "PCld"], [[100, 300], [1500,  2500]], [75, 105], [0, 200])
#create_map_comparison("20251016", ["PCld", "NH3"], [[1000,  3000], [0, 300]], [75, 105], [0, 200])

#create_map_comparison("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200])
#output_cluster_map_and_plot("20251016", ["AOI", "CI"], [[0, 1], [0, 1]], [75, 105], [0, 200], 4)


#output_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
    