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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_clusters(pix_arr, cov_type, n_components, is_soft_clustering, threshold):
    if(is_soft_clustering):
        print("Clustering with probability threshold " + str(threshold))
    gmm_model = GMM(n_components=n_components, covariance_type=cov_type)
    pipe = Pipeline([('scaler', StandardScaler()), ('gmm', gmm_model)])
  
    pipe.fit(pix_arr)
    scaled = pipe.named_steps["scaler"].transform(pix_arr)
    gm = pipe.named_steps["gmm"]
    #P = gm.eval(pix_arr)[0]
    #print(P)
    #p = posterior(gm, pix_ar)
    #bic = pipe.named_steps["gmm"].bic(scaled)
    
    predictions = []
    if(not is_soft_clustering):
        predictions = pipe.predict(pix_arr)
    if(is_soft_clustering):
        pixel_probs = pipe.predict_proba(pix_arr)
        print("soft clustering.....")
        for pixel in pixel_probs:
            index = next((i for i, x in enumerate(pixel) if x > threshold), - 1)
            predictions.append(index)
        
    


    return predictions




  
    
  
  





