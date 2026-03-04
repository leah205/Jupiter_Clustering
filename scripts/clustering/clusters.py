from sklearn.mixture import GaussianMixture as GMM
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import numpy as np
from scripts.preprocessing.preprocessing import get_pix_arr, get_parameter_2d_array, get_map_shape
matplotlib.use('Agg')
from sklearn.metrics import silhouette_score


def create_clusters(pix_arr, cov_type, n_components):
    gmm_model = GMM(n_components=n_components, covariance_type=cov_type)
    print("fitting model...")
    gmm_model.fit(pix_arr)
    predictions = gmm_model.predict(pix_arr)

    return predictions


def create_cluster_plot(pix_arr,  param1, param2, cov_type, n_comp):
    predictions = create_clusters(pix_arr, cov_type, n_comp)
    print("plotting")
    plt.scatter(pix_arr[:, 0], pix_arr[:, 1],c = predictions,s = 1)
    plt.title(f'clustering with {n_comp} components and {cov_type} covariances')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.savefig(f'./data/visualizations/clusters_{cov_type}_{n_comp}_.png')
    return 0
    
def create_cluster_map_arr(keywords, latRng = [45, 135], lngRng = [210, 300]):
    latLen, lonLen = latRng[1] - latRng[0], lngRng[1] - lngRng[0]
    #do subset map here for reshape later
    param_arr = get_parameter_2d_array(keywords, latRng, lngRng)
    print(param_arr.shape)
    pix_arr = get_pix_arr(param_arr, [[60, 160], [1400, 2200], [0.1, 0.4], [0.35, 0.75]])
    pred = create_clusters(pix_arr[:, [0,1]], "full", 5)
    #silhouette score:
    print("silhouette score: " + str(silhouette_score(pix_arr, pred)))
    
    indices = pix_arr[:, pix_arr.shape[1] - 1]
    #concatenat indexed and cluster array
    indexed_clusters = np.column_stack((indices, pred))
    oned_mapped_clusters = np.full(shape = param_arr.shape[1], fill_value = np.nan)
    #np.empty(param_arr.shape[1])
    #print(latLen * lonLen)
    for r in range(indexed_clusters.shape[0]):
        index, cluster = int(indexed_clusters[r][0]), int(indexed_clusters[r][1])
        oned_mapped_clusters[index] = cluster
    map_shape = (get_map_shape(keywords[0], latRng, lngRng))
    #print(oned_mapped_clusters[-50:])
    mapped_clusters = oned_mapped_clusters.reshape(map_shape)
    #print(mapped_clusters)
    #print(map_shape)
    hdul = fits.HDUList()
    hdul.append(fits.PrimaryHDU())
    hdul[0].data = mapped_clusters
    hdul.writeto('output.fits', overwrite= True)
        #print(indexed_clusters[r][0])
        #print(indexed_clusters[r][1])
    #check for 30
    #print(oned_mapped_clusters)
    #create clusters 
    #print(mapped_clusters)
  
   

#pix_arr =  get_pix_arr([[60, 160], [1400, 2200]], ["NH3", "PCld"])
#create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "full", 28)
#create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "diag", 7)




create_cluster_map_arr( ["NH3", "PCld", "AOI", "CI"],  [45, 135], [210, 300])
#print(silhouette_score(hst_pix_arr, pred))