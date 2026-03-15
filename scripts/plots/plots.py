import matplotlib.pyplot as plt
import scripts.preprocessing.preprocessing as pre
from config.config import config
from sklearn.metrics import silhouette_score
import scripts.clustering.clusters as CL
import numpy as np


def create_cluster_plot(date, keywords, param_ranges, n_comp = 5, cov_type = "full", 
                        latRng = [65, 105], lngRng = [0, 50], cm_num = 3
                        ):
 
    input_arr = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)[0]
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    sil_score = str(round(silhouette_score(input_arr, pred), 3))
    #print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    fig, ax = plt.subplots()
    ax.yaxis.set_inverted(True)
    plt.scatter(input_arr[:, 0], input_arr[:, 1],c = pred,s = 1)
    plt.suptitle(f' Lat: {latRng[0]} - {latRng[1]}, Lon: {lngRng[0]} - {lngRng[1]} {date}')
    plt.title(f'{n_comp} components, silhouette: {sil_score} ')
    plt.xlabel(keywords[0])
    plt.ylabel(keywords[1])
    lat_lon_str = f'{latRng[0]}-{latRng[1]}_{lngRng[0]}-{lngRng[1]}'
    plt.savefig(f'{config["output"]}/cluster_plots/clusters_{date}_{lat_lon_str}_{cov_type}_{n_comp}_.png')
    return 0

create_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 5)
create_cluster_plot("20251214", ["NH3", "PCld"], [[0, 300], [1000, 3000]], 4)