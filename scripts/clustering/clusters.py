from sklearn.mixture import GaussianMixture as GMM
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import numpy as np
#from scripts.preprocessing.preprocessing import get_filtered_pix_arr, subset_map, get_parameter_2d_array, get_map_shape
import scripts.preprocessing.preprocessing as pre
#import scripts.preprocessing.mapping as mp
matplotlib.use('Agg')
from sklearn.metrics import silhouette_score
from config.config import config


def create_clusters(pix_arr, cov_type, n_components):
    gmm_model = GMM(n_components=n_components, covariance_type=cov_type)
    print("fitting model...")
    gmm_model.fit(pix_arr)
    predictions = gmm_model.predict(pix_arr)

    return predictions

def create_cluster_plot(keywords, param_ranges, n_comp = 5, cov_type = "full", 
                        latRng = [65, 105], lngRng = [0, 360], cm_num = 3
                        ):
 
    input_arr = pre.get_input_array(keywords, param_ranges, latRng, lngRng, cm_num)[0]
    print(input_arr)
    pred = create_clusters(input_arr[:, [0,1]], cov_type, n_comp)
    print("silhouette score: " + str(silhouette_score(input_arr, pred)))
    fig, ax = plt.subplots()
    ax.yaxis.set_inverted(True)
    plt.scatter(input_arr[:, 0], input_arr[:, 1],c = pred,s = 1)
    plt.title(f'clustering with {n_comp} components and {cov_type} covariances')
    plt.xlabel(keywords[0])
    plt.ylabel(keywords[1])
    plt.savefig(f'{config["output"]}/cluster_plots/clusters_{cov_type}_{n_comp}_.png')
    return 0


  
    
  
   

#pix_arr =  get_pix_arr([[60, 160], [1400, 2200]], ["NH3", "PCld"])
#create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "full", 28)
#create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "diag", 7)




#create_cluster_map_arr( ["NH3", "PCld", "AOI", "CI"], [[60, 250], [1400, 2500], [0.1, 0.4], [0.35, 0.75]])

#create_cluster_map_arr(["NH3", "PCld"], [[30, 250], [1000, 2500]])
#create_cluster_plot(["NH3", "PCld"], [[30, 250], [1000, 2500]])
#print(silhouette_score(hst_pix_arr, pred))