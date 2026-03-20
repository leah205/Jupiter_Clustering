import matplotlib.pyplot as plt
import scripts.preprocessing.preprocessing as pre
from config.config import config
from sklearn.metrics import silhouette_score
import scripts.clustering.clusters as CL
import numpy as np


def create_cluster_plot(ax, keywords, input_arr, pred, n_comp, cov_type, 
                        latRng, lngRng, cmap, cm_num = 3
                        ):
 
 
    #print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    #fig, ax = plt.subplots()
    ax.yaxis.set_inverted(True)
    ax.scatter(input_arr[:, 0], input_arr[:, 1],c = pred,s = 1, cmap = cmap)
    
    ax.set_xlabel(keywords[0])
    ax.set_ylabel(keywords[1])
    
    return 0

