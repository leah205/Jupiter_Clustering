from sklearn.mixture import GaussianMixture as GMM
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
#from scripts.preprocessing.preprocessing import get_filtered_pix_arr, subset_map, get_parameter_2d_array, get_map_shape
import scripts.preprocessing.preprocessing as pre
#import scripts.preprocessing.mapping as mp
matplotlib.use('Agg')
from sklearn.metrics import silhouette_score
from config.config import cf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import scripts.cluster_stats as STAT
import numpy as np
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

import scripts.pca as PCA


def create_clusters(pix_arr, cov_type, n_components, is_soft_clustering, threshold, threshold_type):

    """
    Parameters
    --------
    pix_arr, 
    numpy array with axis 0 as pixels within lon/lat range and axis 1 as parameter pixel radiances
    
    """
    if(is_soft_clustering):
        print("Clustering with probability threshold " + str(threshold))
    gmm_model = GMM(n_components=n_components, covariance_type=cov_type)
    pipe = Pipeline([('scaler', StandardScaler()), ('gmm', gmm_model)])
  
  
    pipe.fit(pix_arr)
    scaler = pipe.named_steps["scaler"]
    scaled = scaler.transform(pix_arr)
    gm = pipe.named_steps["gmm"]
  
    
    pixel_probs = np.array(pipe.predict_proba(pix_arr))

    predictions = pipe.predict(pix_arr)
    
    means = gm.means_
    cov = gm.covariances_
     
    if(is_soft_clustering):
        if(threshold_type == "posterior"):
            threshold_mask = get_posterior_threshold(pixel_probs, threshold)
        else:
            threshold_mask = get_mahalanobis_threshold(predictions, scaled, means, cov, threshold)
     
        print(threshold_mask)
        predictions = np.where(threshold_mask, -1, predictions)
        cl_means = []
        for cl in range(0, n_components):
            mask = ~threshold_mask & predictions == cl
            cl_mean = np.mean(pix_arr[mask], axis = 0)
            cl_means.append(cl_mean)
        means = np.array(cl_means)
      
    means = scaler.inverse_transform(gm.means_)
 
    return [predictions, pixel_probs, means]

def get_posterior_threshold(probs, threshold):
    return np.all(probs < threshold, axis = 1)

def get_mahalanobis_threshold(predictions, pix_arr, means, covariances, prob):
    threshold_mask = np.zeros(predictions.shape[0])
    inv_cov = np.linalg.inv(covariances)
     # number of variables is degrees of freedom
    d_freedom = means.shape[1]
    cutoff = chi2.ppf(prob, d_freedom)

    for cl in range(0, means.shape[0]):
        mask = predictions == cl
       
        cl_indices = np.where(mask)[0]
        cl_mean = means[cl, :]
        # adjust if not only full
        cl_invcov = inv_cov[cl, :, :]
        
        cl_points = pix_arr[cl_indices, :]
        
        def mahalanobis(d):
            return (d-cl_mean).T.dot(cl_invcov).dot(d-cl_mean)
       
        
        m_dists = np.apply_along_axis(mahalanobis, 1, cl_points)

        m_mask = m_dists > cutoff
        cl_indices = cl_indices[m_mask]
       
        threshold_mask[cl_indices] = 1
   
    return threshold_mask.astype(bool)






def run_raw_pipeline(data, cov_type, n_comp, threshold, threshold_type):
    pred,probs, means = create_clusters(data, cov_type, n_comp, cf["soft_clustering"], threshold, threshold_type)
    #stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))

    #means = np.swapaxes(stats[:,:, 0], 0, 1)
    return {
        "pred": pred,
        "probs": probs,
        "means": means,
       
    }
    

def run_pca_pipeline(data, cov_type, n_comp, threshold, threshold_type):
    [pca_reduced, pca_obj, scaler] = PCA.get_pca_comp(data)
    pred, probs, means = create_clusters(pca_reduced, cov_type, n_comp, cf["soft_clustering"], threshold, threshold_type)
    means = pca_obj.inverse_transform(means)
    means = scaler.inverse_transform(means)
    return {
        "pred": pred,
        "probs": probs,
        "means": means,
        "pca_obj": pca_obj
    }
   


  
    
  
  





