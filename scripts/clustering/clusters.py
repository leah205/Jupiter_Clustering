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

import scripts.pca as PCA


def create_clusters(pix_arr, cov_type, n_components, is_soft_clustering, threshold):

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
    #P = gm.eval(pix_arr)[0]
    #print(P)
    #p = posterior(gm, pix_ar)
    #bic = pipe.named_steps["gmm"].bic(scaled)
    
    pixel_probs = pipe.predict_proba(pix_arr)

    predictions = []
    if(not is_soft_clustering):
        predictions = pipe.predict(pix_arr)
    if(is_soft_clustering):
        
        print("soft clustering.....")
        for pixel in pixel_probs:
            index = next((i for i, x in enumerate(pixel) if x > threshold), - 1)
            predictions.append(index)
        
    
    means = scaler.inverse_transform(gm.means_)
 
    return [predictions, pixel_probs, means]


def run_raw_pipeline(data, cov_type, n_comp, threshold):
    pred,probs, means = create_clusters(data, cov_type, n_comp, cf["soft_clustering"], threshold)
    #stats = STAT.get_all_stats(pred, keywords, input_arr[:, len(keywords)], n_comp, latRng, lngRng, cm_num)
    #pred = STAT.reassign_clusters(np.array(pred), np.swapaxes(stats[:, :, 0], 0, 1), np.array(param_ranges))

    #means = np.swapaxes(stats[:,:, 0], 0, 1)
    return {
        "pred": pred,
        "probs": probs,
        "means": means,
       
    }
    

def run_pca_pipeline(data, cov_type, n_comp, threshold):
    [pca_reduced, pca_obj, scaler] = PCA.get_pca_comp(data)
    pred, probs, means = create_clusters(pca_reduced, cov_type, n_comp, cf["soft_clustering"], threshold)
    means = pca_obj.inverse_transform(means)
    means = scaler.inverse_transform(means)
    return {
        "pred": pred,
        "probs": probs,
        "means": means,
        "pca_obj": pca_obj
    }
   


  
    
  
  





