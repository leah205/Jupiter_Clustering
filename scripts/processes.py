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
                           n_comp = 4, cov_type = "full", cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp, config["soft_clustering"])
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    fig, axis = pl.subplots(2, 1)
    print("plotting...")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}_map_and_plot.png')

def output_cluster_map_and_plots(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330], 
                           n_comp = 4, cov_type = "full", cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"])
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])


    fig, axis = pl.subplots(3, 1)
    print("plotting...")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1],  keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[2],  keywords, 2, 3, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)

    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}_map_and_plot.png')



def output_cluster_map(date, keywords, param_ranges, 
                           latRng = [65, 115], lngRng = [0, 50], 
                            n_comp = 4, cov_type = "full", cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"])
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    #colors = np.random.rand(n_comp, 3)
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[:n_comp])
    print("plotting...")
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_maps/spatial_map_{date}_{lat_lon_str}_{n_comp}.png')


def output_cluster_plot(date, keywords, param_ranges, 
                           latRng = [90, 95], lngRng = [330, 330], 
                           n_comp = 4, cov_type = "full",  cm_num = 3):
    
   
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, 0: len(keywords)], cov_type, n_comp, config["soft_clustering"])
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = ["red", "green", "blue", "yellow"]
    cmap = ListedColormap(colors[:n_comp])
    print("plotting...")
    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, keywords, 0, 1, input_arr, pred,  n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_plots/clusters_{date}_{lat_lon_str}_{cov_type}_{n_comp}_.png')




def create_map_comparison(date, keywords, param_ranges, 
                           latRng = [85, 95], lngRng = [230, 330],
                            n_comp = 4, cov_type = "full",  cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
  
    pred = CL.create_clusters(input_arr[:, 0:len(keywords)], cov_type, n_comp, config["soft_clustering"])
    #print("scoring...")
    #sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors =["red", "green", "blue", "yellow", "orange", "pink"]
    cmap = ListedColormap(colors[: n_comp])
    print("start fig")
    fig, axis = pl.subplots(3, 1, constrained_layout = True)
    fig.tight_layout()
    cloud_map = pre.get_patch("PCld", latRng, lngRng)
    amm_map = pre.get_patch("NH3", latRng, lngRng)
    print("plotting")
    MP.create_cluster_map(axis[0], n_comp, pred, input_arr, subset_shape, latRng, lngRng, cmap, cm_num, fig, True)
    MP.plot_patch(cloud_map, latRng, lngRng, "Blues", axis[1], 1000, 3000,  "Cloud Pressure", cm_num, fig, True)
    MP.plot_patch(amm_map, latRng, lngRng, "terrain_r", axis[2], 0, 300, "Ammonia", cm_num, fig, True)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}_map_comparison.png')
    


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#get correct filters
create_map_comparison("20251016", ["NH3", "PCld", "AOI", "CI"], [[100, 300], [1500,  2500], [0, 1], [0, 1]])#
#output_cluster_map_and_plots("20251016", ["NH3", "PCld", "AOI", "CI"], [[100, 300], [1500,  2500], [0, 1], [0, 1]], [85, 95], [230,330], 4)


#output_cluster_map_and_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#output_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
    