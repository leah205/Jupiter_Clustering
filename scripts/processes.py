from astropy.io import fits
import numpy as np
import pylab as pl
import scripts.preprocessing.preprocessing as pre
import scripts.plots.plots as PL
import scripts.clustering.clusters as CL 
from sklearn.metrics import silhouette_score
from config.config import config
import scripts.plots.mapping as MP
from matplotlib.colors import ListedColormap


def output_cluster_map_and_plot(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [75, 105], lngRng = [0, 360], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)

    fig, axis = pl.subplots(2, 1)
    MP.create_cluster_map(axis[0], date, n_comp, pred, input_arr, subset_shape, sil_score, latRng, lngRng, cmap, cm_num)
    PL.create_cluster_plot(axis[1], date, keywords, input_arr, pred, sil_score, n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/{date}_{lat_lon_str}_{n_comp}.png')

def output_cluster_map(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)
    fig, axis = pl.subplots(1, 1)
    MP.create_cluster_map(axis, date, n_comp, pred, input_arr, subset_shape, sil_score, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_maps/spatial_map_{date}_{lat_lon_str}_{n_comp}.png')


def output_cluster_plot(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                           latRng = [65, 115], lngRng = [0, 50], cm_num = 3):
    
   
    [input_arr, subset_shape] = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    print(pred)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    colors = np.random.rand(n_comp, 3)
    cmap = ListedColormap(colors)
    fig, axis = pl.subplots(1, 1)
    PL.create_cluster_plot(axis, date, keywords, input_arr, pred, sil_score, n_comp, cov_type, latRng, lngRng, cmap, cm_num)
    fig.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}, {n_comp} components, silhouette: {sil_score}')
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    fig.savefig(f'{config["output"]}/cluster_plots/clusters_{date}_{lat_lon_str}_{cov_type}_{n_comp}_.png')


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
output_cluster_map_and_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 4)#


#output_cluster_map("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
#output_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
    