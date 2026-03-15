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




  
    
  
  





