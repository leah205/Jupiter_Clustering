from sklearn.mixture import GaussianMixture as GMM
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scripts.preprocessing.preprocessing import get_pix_arr
matplotlib.use('Agg')


def create_clusters(pix_arr, cov_type, n_components):
    gmm_model = GMM(n_components=n_components, covariance_type=cov_type)
    print("fitting model...")
    gmm_model.fit(pix_arr)
    predictions = gmm_model.predict(pix_arr)
    print(predictions)
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
    


pix_arr =  get_pix_arr([[60, 160], [1400, 2200]], ["NH3", "PCld"])


create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "full", 28)
#create_cluster_plot(pix_arr, "ammonia content", "cloud pressure", "diag", 7)