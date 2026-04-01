import matplotlib.pyplot as plt
import scripts.preprocessing.preprocessing as pre
from config.config import config
from sklearn.metrics import silhouette_score
import scripts.clustering.clusters as CL
import numpy as np
from scipy.stats import gaussian_kde


def create_cluster_plot(ax, keywords, index_x, index_y, input_arr, pred, n_comp, cov_type, 
                        latRng, lngRng, cmap, cm_num = 3
                        ):
 
 
    #print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    #fig, ax = plt.subplots()
    ax.yaxis.set_inverted(True)
    ##xy = np.vstack(input_arr[:,0], input_arr[:, 1])
   # kde = gaussian_kde(xy)
    print("shape:" + str(input_arr.shape))
    ax.scatter(input_arr[:, index_x], input_arr[:, index_y],c = pred,s = 1, cmap = cmap, alpha = 0.01)
    
    ax.set_xlabel(keywords[index_x])
    ax.set_ylabel(keywords[index_y])
    
    return 0

