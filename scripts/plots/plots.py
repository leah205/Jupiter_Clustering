import matplotlib.pyplot as plt
import scripts.preprocessing.preprocessing as pre
from config.config import config
from sklearn.metrics import silhouette_score
import scripts.clustering.clusters as CL
import numpy as np


def create_cluster_plot(keywords, param_ranges, n_comp = 5, cov_type = "full", 
                        latRng = [65, 105], lngRng = [0, 50], cm_num = 3
                        ):
 
    input_arr = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)[0]
    print(input_arr)
    pred = CL.create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    fig, ax = plt.subplots()
    ax.yaxis.set_inverted(True)
    plt.scatter(input_arr[:, 0], input_arr[:, 1],c = pred,s = 1)
    plt.title(f'clustering with {n_comp} components and {cov_type} covariances')
    plt.xlabel(keywords[0])
    plt.ylabel(keywords[1])
    plt.savefig(f'{config["output"]}/cluster_plots/clusters_{cov_type}_{n_comp}_.png')
    return 0

create_cluster_plot(["NH3", "PCld"], [[30, 250], [1000, 2500]])